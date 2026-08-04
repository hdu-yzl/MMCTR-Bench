"""Versionable user/item cold-start split constraints."""

import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Sequence, Tuple, Union

from mmctr.config import load_yaml_mapping
from mmctr.core import ContractError
from mmctr.core.registry import RegistryError
from mmctr.experiments import ExperimentTask, load_task_matrix, save_task_matrix
from mmctr.models.registry import MODEL_REGISTRY


COLD_START_STUDY_MATRIX_SCHEMA = "cold-start-study-matrix-v1"
_CONFIG_KEYS = frozenset(
    {
        "dataset",
        "data_fingerprint",
        "data",
        "model_configs",
        "models",
        "seeds",
        "audit_manifest",
    }
)
PathLike = Union[str, Path]


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(name): _plain(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ColdStartAudit:
    target: str
    regime: str
    target_count: int
    train_events: int
    evaluation_events: int
    maximum_support_interactions: int
    support_counts: Mapping[int, int]
    fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "support_counts", MappingProxyType(dict(self.support_counts)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "regime": self.regime,
            "target_count": self.target_count,
            "train_events": self.train_events,
            "evaluation_events": self.evaluation_events,
            "maximum_support_interactions": self.maximum_support_interactions,
            "support_counts": {
                str(target_id): count for target_id, count in sorted(self.support_counts.items())
            },
            "protocol_fingerprint": self.fingerprint,
        }


class ColdStartProtocol:
    """Validate event disjointness and target exposure for one versioned split."""

    def __init__(
        self,
        target: str,
        regime: str,
        max_support_interactions: int = 0,
    ) -> None:
        if target not in {"user", "item"}:
            raise ContractError("cold-start target must be user or item")
        if regime not in {"cold_start", "zero_shot", "few_shot"}:
            raise ContractError("cold-start regime must be cold_start, zero_shot, or few_shot")
        if isinstance(max_support_interactions, bool) or int(max_support_interactions) < 0:
            raise ContractError("max_support_interactions must be non-negative")
        support = int(max_support_interactions)
        if regime == "few_shot" and support <= 0:
            raise ContractError("few-shot protocol requires a positive support limit")
        if regime != "few_shot" and support != 0:
            raise ContractError("zero/cold-start protocols cannot contain support interactions")
        self.target = target
        self.regime = regime
        self.max_support_interactions = support

    @staticmethod
    def _validate_partition(
        name: str,
        event_ids: Sequence[str],
        user_ids: Sequence[int],
        item_ids: Sequence[int],
    ) -> Tuple[Tuple[str, ...], Tuple[int, ...], Tuple[int, ...]]:
        events = tuple(str(value) for value in event_ids)
        users = tuple(int(value) for value in user_ids)
        items = tuple(int(value) for value in item_ids)
        if not events or len(events) != len(users) or len(events) != len(items):
            raise ContractError(
                "{} event/user/item arrays must be non-empty and aligned".format(name)
            )
        if len(set(events)) != len(events):
            raise ContractError("{} event IDs must be unique".format(name))
        if any(value < 0 for value in users) or any(value <= 0 for value in items):
            raise ContractError("cold-start user/item IDs violate canonical ID rules")
        return events, users, items

    def audit(
        self,
        train_event_ids: Sequence[str],
        train_user_ids: Sequence[int],
        train_item_ids: Sequence[int],
        evaluation_event_ids: Sequence[str],
        evaluation_user_ids: Sequence[int],
        evaluation_item_ids: Sequence[int],
    ) -> ColdStartAudit:
        train_events, train_users, train_items = self._validate_partition(
            "train", train_event_ids, train_user_ids, train_item_ids
        )
        evaluation_events, evaluation_users, evaluation_items = self._validate_partition(
            "evaluation", evaluation_event_ids, evaluation_user_ids, evaluation_item_ids
        )
        overlap = set(train_events).intersection(evaluation_events)
        if overlap:
            raise ContractError("cold-start train/evaluation event sets must be disjoint")
        train_targets = train_users if self.target == "user" else train_items
        evaluation_targets = evaluation_users if self.target == "user" else evaluation_items
        evaluation_target_set = set(evaluation_targets)
        train_counts = Counter(train_targets)
        support_counts = {
            target_id: int(train_counts.get(target_id, 0))
            for target_id in sorted(evaluation_target_set)
        }
        if self.regime in {"zero_shot", "cold_start"}:
            if any(support_counts.values()):
                raise ContractError("zero-shot target IDs must be absent from training")
        else:
            invalid = {
                target_id: count
                for target_id, count in support_counts.items()
                if count < 1 or count > self.max_support_interactions
            }
            if invalid:
                raise ContractError(
                    "few-shot targets must have 1..{} support interactions: {}".format(
                        self.max_support_interactions, invalid
                    )
                )
        payload = {
            "target": self.target,
            "regime": self.regime,
            "max_support_interactions": self.max_support_interactions,
            "train": list(zip(train_events, train_users, train_items)),
            "evaluation": list(zip(evaluation_events, evaluation_users, evaluation_items)),
            "support_counts": support_counts,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return ColdStartAudit(
            target=self.target,
            regime=self.regime,
            target_count=len(evaluation_target_set),
            train_events=len(train_events),
            evaluation_events=len(evaluation_events),
            maximum_support_interactions=max(support_counts.values()),
            support_counts=support_counts,
            fingerprint=hashlib.sha256(encoded).hexdigest(),
        )


def _manifest_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save_cold_start_audit(path: Path, audit: ColdStartAudit) -> Path:
    """Atomically persist an integrity-protected cold-start audit manifest."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "schema": "mmctr-cold-start-audit-v1",
        "audit": audit.to_dict(),
    }
    payload["manifest_fingerprint"] = _manifest_digest(payload)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}-".format(destination.name), dir=str(destination.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, str(destination))
    except BaseException:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
        raise
    return destination


def load_cold_start_audit(path: Path) -> ColdStartAudit:
    """Load a versioned audit and reject incomplete or modified manifests."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != "mmctr-cold-start-audit-v1":
        raise ContractError("unsupported cold-start audit manifest schema")
    fingerprint = payload.pop("manifest_fingerprint", None)
    if not isinstance(fingerprint, str) or fingerprint != _manifest_digest(payload):
        raise ContractError("cold-start audit manifest fingerprint mismatch")
    values = payload.get("audit")
    if not isinstance(values, dict):
        raise ContractError("cold-start audit manifest is missing audit values")
    try:
        support_counts = {
            int(target_id): int(count) for target_id, count in values["support_counts"].items()
        }
        audit = ColdStartAudit(
            target=str(values["target"]),
            regime=str(values["regime"]),
            target_count=int(values["target_count"]),
            train_events=int(values["train_events"]),
            evaluation_events=int(values["evaluation_events"]),
            maximum_support_interactions=int(values["maximum_support_interactions"]),
            support_counts=support_counts,
            fingerprint=str(values["protocol_fingerprint"]),
        )
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise ContractError("invalid cold-start audit manifest values") from error
    if (
        audit.target not in {"user", "item"}
        or audit.regime not in {"cold_start", "zero_shot", "few_shot"}
        or audit.target_count != len(audit.support_counts)
        or audit.train_events <= 0
        or audit.evaluation_events <= 0
        or len(audit.fingerprint) != 64
        or audit.maximum_support_interactions != max(audit.support_counts.values(), default=0)
    ):
        raise ContractError("invalid cold-start audit manifest values")
    return audit


def build_cold_start_study_tasks(
    dataset: str,
    data_fingerprint: str,
    data_config: Mapping[str, Any],
    model_configs: Mapping[str, Mapping[str, Any]],
    models: Sequence[str],
    seeds: Sequence[int],
    audit: ColdStartAudit,
    audit_manifest: Path,
) -> Tuple[ExperimentTask, ...]:
    """Build evaluation tasks anchored to one verified split-audit manifest."""

    if not isinstance(data_config, Mapping) or not isinstance(model_configs, Mapping):
        raise ContractError("cold-start study data/model configs must be mappings")
    if not isinstance(audit, ColdStartAudit):
        raise ContractError("cold-start study requires a verified audit")
    manifest_path = Path(audit_manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise ContractError("cold-start audit manifest does not exist")
    try:
        model_names = tuple(MODEL_REGISTRY.canonical_name(name) for name in models)
    except RegistryError as error:
        raise ContractError(str(error)) from error
    if not model_names or len(set(model_names)) != len(model_names):
        raise ContractError("cold-start study models must be non-empty and unique")
    seed_values = tuple(int(seed) for seed in seeds)
    if (
        not seed_values
        or any(isinstance(seed, bool) for seed in seeds)
        or len(set(seed_values)) != len(seed_values)
    ):
        raise ContractError("cold-start study seeds must be non-empty unique integers")
    if not isinstance(data_fingerprint, str) or not data_fingerprint:
        raise ContractError("cold-start data_fingerprint must be non-empty")

    plain_data = _plain(data_config)
    tasks = []
    for model_name in model_names:
        try:
            model_config = model_configs[model_name]
        except KeyError as error:
            raise ContractError(
                "cold-start study is missing config for model {!r}".format(model_name)
            ) from error
        if not isinstance(model_config, Mapping):
            raise ContractError("cold-start study model config must be a mapping")
        for seed in seed_values:
            tasks.append(
                ExperimentTask(
                    task_id="{}-{}-{}-seed-{}".format(
                        model_name, audit.target, audit.regime.replace("_", "-"), seed
                    ),
                    dataset=str(dataset),
                    model=model_name,
                    seed=seed,
                    resolved_config={
                        "data_fingerprint": data_fingerprint,
                        "data": plain_data,
                        "model": _plain(model_config),
                        "analysis": {
                            "protocol": "cold-start-evaluation-v1",
                            "target": audit.target,
                            "regime": audit.regime,
                            "maximum_support_interactions": audit.maximum_support_interactions,
                            "audit_fingerprint": audit.fingerprint,
                            "audit_manifest": manifest_path.as_posix(),
                            "audit_manifest_sha256": _file_sha256(manifest_path),
                        },
                    },
                )
            )
    return tuple(tasks)


def load_cold_start_study_config(path: PathLike) -> Tuple[ExperimentTask, ...]:
    """Load strict YAML and verify its audit before constructing tasks."""

    config_path = Path(path).expanduser().resolve()
    values = load_yaml_mapping(config_path)
    keys = set(values)
    missing = sorted(_CONFIG_KEYS - keys)
    unknown = sorted(keys - _CONFIG_KEYS)
    if missing or unknown:
        problems = []
        if missing:
            problems.append("missing keys: {}".format(", ".join(missing)))
        if unknown:
            problems.append("unknown keys: {}".format(", ".join(unknown)))
        raise ContractError("invalid cold-start study config ({})".format("; ".join(problems)))
    audit_manifest = Path(values["audit_manifest"]).expanduser()
    if not audit_manifest.is_absolute():
        audit_manifest = config_path.parent / audit_manifest
    audit_manifest = audit_manifest.resolve()
    audit = load_cold_start_audit(audit_manifest)
    return build_cold_start_study_tasks(
        dataset=values["dataset"],
        data_fingerprint=values["data_fingerprint"],
        data_config=values["data"],
        model_configs=values["model_configs"],
        models=values["models"],
        seeds=values["seeds"],
        audit=audit,
        audit_manifest=audit_manifest,
    )


def save_cold_start_study_matrix(tasks: Sequence[ExperimentTask], path: PathLike) -> Path:
    """Atomically save a canonical cold-start task matrix."""

    return save_task_matrix(tasks, path, COLD_START_STUDY_MATRIX_SCHEMA)


def load_cold_start_study_matrix(path: PathLike) -> Tuple[ExperimentTask, ...]:
    """Load and verify a canonical cold-start task matrix."""

    return load_task_matrix(path, COLD_START_STUDY_MATRIX_SCHEMA)


__all__ = [
    "COLD_START_STUDY_MATRIX_SCHEMA",
    "ColdStartAudit",
    "ColdStartProtocol",
    "build_cold_start_study_tasks",
    "load_cold_start_audit",
    "load_cold_start_study_config",
    "load_cold_start_study_matrix",
    "save_cold_start_audit",
    "save_cold_start_study_matrix",
]

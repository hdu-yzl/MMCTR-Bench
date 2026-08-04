"""Seeded missing-modality transforms shared by robustness experiments."""

import hashlib
import math
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Sequence, Tuple, Union

import torch

from mmctr.config import load_yaml_mapping
from mmctr.core import Batch, ContractError
from mmctr.core.registry import RegistryError
from mmctr.experiments import ExperimentTask, load_task_matrix, save_task_matrix
from mmctr.models.common.registry import MODEL_REGISTRY


ROBUSTNESS_STUDY_MATRIX_SCHEMA = "robustness-study-matrix-v1"
_CONFIG_KEYS = frozenset(
    {
        "dataset",
        "data_fingerprint",
        "data",
        "model_configs",
        "models",
        "modalities",
        "probabilities",
        "seeds",
        "splits",
    }
)
PathLike = Union[str, Path]


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(name): _plain(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def build_robustness_study_tasks(
    dataset: str,
    data_fingerprint: str,
    data_config: Mapping[str, Any],
    model_configs: Mapping[str, Mapping[str, Any]],
    models: Sequence[str],
    modalities: Sequence[str],
    probabilities: Sequence[float],
    seeds: Sequence[int],
    splits: Sequence[str],
) -> Tuple[ExperimentTask, ...]:
    """Build production-model tasks that vary only the canonical dropout protocol."""

    if not isinstance(data_config, Mapping) or not isinstance(model_configs, Mapping):
        raise ContractError("robustness study data/model configs must be mappings")
    try:
        model_names = tuple(MODEL_REGISTRY.canonical_name(name) for name in models)
    except RegistryError as error:
        raise ContractError(str(error)) from error
    if not model_names or len(set(model_names)) != len(model_names):
        raise ContractError("robustness study models must be non-empty and unique")
    modality_names = tuple(str(name) for name in modalities)
    if (
        not modality_names
        or "id" in modality_names
        or any(not name for name in modality_names)
        or len(set(modality_names)) != len(modality_names)
    ):
        raise ContractError("robustness modalities must be unique non-ID names")
    probability_values = tuple(float(value) for value in probabilities)
    if (
        not probability_values
        or any(isinstance(value, bool) for value in probabilities)
        or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probability_values)
        or len(set(probability_values)) != len(probability_values)
    ):
        raise ContractError("robustness probabilities must be unique finite values in [0, 1]")
    seed_values = tuple(int(seed) for seed in seeds)
    if (
        not seed_values
        or any(isinstance(seed, bool) for seed in seeds)
        or len(set(seed_values)) != len(seed_values)
    ):
        raise ContractError("robustness seeds must be non-empty unique integers")
    split_names = tuple(
        "val" if str(split).lower() == "validation" else str(split).lower() for split in splits
    )
    if (
        not split_names
        or any(split not in {"train", "val", "test"} for split in split_names)
        or len(set(split_names)) != len(split_names)
    ):
        raise ContractError("robustness splits must be unique train, val, or test names")
    if not isinstance(data_fingerprint, str) or not data_fingerprint:
        raise ContractError("robustness data_fingerprint must be non-empty")

    plain_data = _plain(data_config)
    tasks = []
    for model_name in model_names:
        try:
            model_config = model_configs[model_name]
        except KeyError as error:
            raise ContractError(
                "robustness study is missing config for model {!r}".format(model_name)
            ) from error
        if not isinstance(model_config, Mapping):
            raise ContractError("robustness study model config must be a mapping")
        for probability in probability_values:
            probability_name = ("{:.6f}".format(probability)).rstrip("0").rstrip(".") or "0"
            for seed in seed_values:
                tasks.append(
                    ExperimentTask(
                        task_id="{}-drop-{}-seed-{}".format(
                            model_name, probability_name.replace(".", "p"), seed
                        ),
                        dataset=str(dataset),
                        model=model_name,
                        seed=seed,
                        resolved_config={
                            "data_fingerprint": data_fingerprint,
                            "data": plain_data,
                            "model": _plain(model_config),
                            "analysis": {
                                "protocol": "modality-dropout-v1",
                                "modalities": modality_names,
                                "probability": probability,
                                "splits": split_names,
                                "mask_seed": seed,
                            },
                        },
                    )
                )
    return tuple(tasks)


def load_robustness_study_config(path: PathLike) -> Tuple[ExperimentTask, ...]:
    """Load one strict YAML robustness definition into canonical tasks."""

    values = load_yaml_mapping(path)
    keys = set(values)
    missing = sorted(_CONFIG_KEYS - keys)
    unknown = sorted(keys - _CONFIG_KEYS)
    if missing or unknown:
        problems = []
        if missing:
            problems.append("missing keys: {}".format(", ".join(missing)))
        if unknown:
            problems.append("unknown keys: {}".format(", ".join(unknown)))
        raise ContractError("invalid robustness study config ({})".format("; ".join(problems)))
    return build_robustness_study_tasks(
        dataset=values["dataset"],
        data_fingerprint=values["data_fingerprint"],
        data_config=values["data"],
        model_configs=values["model_configs"],
        models=values["models"],
        modalities=values["modalities"],
        probabilities=values["probabilities"],
        seeds=values["seeds"],
        splits=values["splits"],
    )


def save_robustness_study_matrix(tasks: Sequence[ExperimentTask], path: PathLike) -> Path:
    """Atomically save a canonical robustness task matrix."""

    return save_task_matrix(tasks, path, ROBUSTNESS_STUDY_MATRIX_SCHEMA)


def load_robustness_study_matrix(path: PathLike) -> Tuple[ExperimentTask, ...]:
    """Load and verify a canonical robustness task matrix."""

    return load_task_matrix(path, ROBUSTNESS_STUDY_MATRIX_SCHEMA)


class ModalityDropout:
    """Zero complete target/history modalities for the same sampled examples."""

    def __init__(self, modalities: Sequence[str], probability: float, seed: int) -> None:
        names = tuple(str(name) for name in modalities)
        if not names or any(not name for name in names) or len(set(names)) != len(names):
            raise ContractError("robustness modalities must be unique non-empty names")
        if "id" in names:
            raise ContractError("ID is not a droppable modality")
        if isinstance(probability, bool) or not 0.0 <= float(probability) <= 1.0:
            raise ContractError("modality dropout probability must be in [0, 1]")
        if isinstance(seed, bool):
            raise ContractError("modality dropout seed must be an integer")
        self.modalities: Tuple[str, ...] = names
        self.probability = float(probability)
        self.seed = int(seed)

    def _generator(self, batch: Batch, modality: str) -> torch.Generator:
        split = str(batch.metadata.get("split", ""))
        batch_index = batch.metadata.get("batch_index")
        if not split or isinstance(batch_index, bool) or not isinstance(batch_index, int):
            raise ContractError("robustness batches require split and integer batch_index metadata")
        payload = "{}:{}:{}:{}".format(self.seed, split, batch_index, modality).encode("utf-8")
        derived_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(derived_seed)
        return generator

    @staticmethod
    def _zero_examples(values: torch.Tensor, missing: torch.Tensor) -> torch.Tensor:
        shape = (values.shape[0],) + (1,) * (values.ndim - 1)
        broadcast = missing.to(device=values.device).reshape(shape)
        return torch.where(broadcast, torch.zeros_like(values), values)

    def __call__(self, batch: Batch) -> Batch:
        if not isinstance(batch, Batch):
            raise ContractError("modality dropout requires a canonical Batch")
        items = dict(batch.item_features)
        histories = dict(batch.history_features)
        missing_counts: Dict[str, int] = {}
        for name in self.modalities:
            if name not in items or name not in histories:
                raise ContractError(
                    "modality {!r} must exist in both target and history features".format(name)
                )
            missing = torch.rand(batch.batch_size, generator=self._generator(batch, name)).lt(
                self.probability
            )
            items[name] = self._zero_examples(items[name], missing)
            histories[name] = self._zero_examples(histories[name], missing)
            missing_counts[name] = int(missing.sum().item())
        metadata = dict(batch.metadata)
        metadata["robustness"] = {
            "protocol": "per-example-target-history-consistent-v1",
            "modalities": self.modalities,
            "probability": self.probability,
            "seed": self.seed,
            "missing_examples": missing_counts,
        }
        return Batch(
            user_features=batch.user_features,
            item_features=items,
            history_features=histories,
            history_mask=batch.history_mask,
            labels=batch.labels,
            context_features=batch.context_features,
            metadata=metadata,
        )


class TransformedDataLoader:
    """Apply one canonical batch transform to explicitly selected splits."""

    def __init__(
        self,
        source: Any,
        transform: ModalityDropout,
        splits: Sequence[str],
    ) -> None:
        if not hasattr(source, "iter_batches") or not callable(source.iter_batches):
            raise ContractError("transformed loader requires a canonical source loader")
        if not isinstance(transform, ModalityDropout):
            raise ContractError("transformed loader requires a ModalityDropout")
        names = tuple(
            "val" if str(split).lower() == "validation" else str(split).lower() for split in splits
        )
        if not names or any(name not in {"train", "val", "test"} for name in names):
            raise ContractError("transformed loader splits must be train, val, or test")
        if len(set(names)) != len(names):
            raise ContractError("transformed loader splits must be unique")
        self.source = source
        self.transform = transform
        self.splits = frozenset(names)
        self.dataset_name = source.dataset_name
        self.manifest = source.manifest

    def iter_batches(self, split: str) -> Iterator[Batch]:
        canonical_split = "val" if str(split).lower() == "validation" else str(split).lower()
        for batch in self.source.iter_batches(canonical_split):
            yield self.transform(batch) if canonical_split in self.splits else batch


__all__ = [
    "ModalityDropout",
    "ROBUSTNESS_STUDY_MATRIX_SCHEMA",
    "TransformedDataLoader",
    "build_robustness_study_tasks",
    "load_robustness_study_config",
    "load_robustness_study_matrix",
    "save_robustness_study_matrix",
]

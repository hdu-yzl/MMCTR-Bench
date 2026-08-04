"""Create isolated, traceable directories for experiment runs."""

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import yaml


PathLike = Union[str, os.PathLike]
SCHEMA_VERSION = 1
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
_UNSAFE_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")


def _normalise(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalise(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def config_fingerprint(config: Mapping[str, Any], length: int = 10) -> str:
    """Return a stable short SHA-256 digest for a resolved configuration."""

    if length < 6 or length > 64:
        raise ValueError("fingerprint length must be between 6 and 64")
    payload = json.dumps(
        _normalise(config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def _safe_component(value: str, field: str) -> str:
    raw = str(value).strip()
    if not raw or raw in {".", ".."} or "/" in raw or "\\" in raw:
        raise ValueError("{} is not a safe path component: {!r}".format(field, value))
    safe = _UNSAFE_COMPONENT.sub("-", raw).strip("-.")
    if not safe:
        raise ValueError("{} is not a safe path component: {!r}".format(field, value))
    return safe


def _as_utc(now: Optional[datetime]) -> datetime:
    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=".{}-".format(path.name),
            suffix=".tmp",
            dir=str(path.parent),
            delete=False,
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _package_versions() -> Dict[str, str]:
    versions = {}
    for distribution in ("numpy", "torch", "tensorflow"):
        try:
            versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def _git_commit(repository_root: Optional[PathLike]) -> str:
    root = Path(repository_root or Path.cwd()).expanduser().resolve()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


@dataclass(frozen=True)
class RunContext:
    """Filesystem paths and lifecycle operations for one isolated run."""

    run_id: str
    config_hash: str
    root_dir: Path
    checkpoints_dir: Path

    @property
    def metadata_path(self) -> Path:
        return self.root_dir / "run_metadata.json"

    @property
    def resolved_config_path(self) -> Path:
        return self.root_dir / "resolved_config.yaml"

    @property
    def metrics_path(self) -> Path:
        return self.root_dir / "metrics.jsonl"

    @property
    def summary_path(self) -> Path:
        return self.root_dir / "summary.json"

    @property
    def log_path(self) -> Path:
        return self.root_dir / "run.log"

    def runtime_config(self) -> Dict[str, str]:
        return {
            "run_id": self.run_id,
            "run_dir": str(self.root_dir),
            "checkpoint_dir": str(self.checkpoints_dir),
            "log_file": str(self.log_path),
        }

    def write_resolved_config(self, config: Mapping[str, Any]) -> None:
        content = yaml.safe_dump(
            _normalise(config),
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=True,
        )
        _atomic_write_text(self.resolved_config_path, content)

    def write_summary(self, summary: Mapping[str, Any]) -> None:
        content = (
            json.dumps(
                _normalise(summary),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        _atomic_write_text(self.summary_path, content)

    def append_metrics(self, metrics: Mapping[str, Any]) -> None:
        """Append one JSON record to this run's versioned metrics stream."""

        content = json.dumps(_normalise(metrics), ensure_ascii=False, sort_keys=True) + "\n"
        with self.metrics_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())

    def update_metadata(self, updates: Mapping[str, Any]) -> None:
        with self.metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        metadata.update(_normalise(updates))
        content = json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        _atomic_write_text(self.metadata_path, content)

    def finalize(
        self,
        status: str,
        summary: Optional[Mapping[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        if status not in TERMINAL_STATUSES:
            raise ValueError("invalid terminal run status: {}".format(status))
        if summary is not None:
            self.write_summary(summary)
        updates = {
            "status": status,
            "ended_at": datetime.now(timezone.utc).isoformat(),
        }
        if error is not None:
            updates["error"] = error
        self.update_metadata(updates)


def create_run_context(
    output_root: PathLike,
    experiment_name: str,
    dataset: str,
    model: str,
    resolved_config: Mapping[str, Any],
    now: Optional[datetime] = None,
    entropy: Optional[str] = None,
    repository_root: Optional[PathLike] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> RunContext:
    """Claim a unique run directory and atomically write each provenance file.

    Directory creation uses ``exist_ok=False`` to reject collisions. Each initial file is
    independently replaced from a same-directory temporary file; callers own lifecycle
    finalization if later runtime setup fails.
    """

    experiment_component = _safe_component(experiment_name, "experiment_name")
    dataset_component = _safe_component(dataset, "dataset")
    model_component = _safe_component(model, "model")
    config_hash = config_fingerprint(resolved_config)
    timestamp = _as_utc(now)
    entropy_component = _safe_component(entropy or uuid.uuid4().hex[:8], "entropy")
    run_id = "{}-{}-{}".format(
        timestamp.strftime("%Y%m%dT%H%M%S%fZ"),
        config_hash,
        entropy_component,
    )

    root_dir = (
        Path(output_root).expanduser().resolve()
        / experiment_component
        / dataset_component
        / model_component
        / run_id
    )
    root_dir.mkdir(parents=True, exist_ok=False)
    checkpoints_dir = root_dir / "checkpoints"
    checkpoints_dir.mkdir()

    context = RunContext(
        run_id=run_id,
        config_hash=config_hash,
        root_dir=root_dir,
        checkpoints_dir=checkpoints_dir,
    )
    initial_metadata = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "experiment_name": experiment_component,
        "dataset": dataset_component,
        "model": model_component,
        "config_hash": config_hash,
        "code_version": _git_commit(repository_root),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": _package_versions(),
        "command": list(sys.argv),
        "working_directory": str(Path.cwd().resolve()),
        "started_at": timestamp.isoformat(),
        "ended_at": None,
        "status": "running",
    }
    if metadata:
        initial_metadata.update(_normalise(metadata))

    _atomic_write_text(
        context.metadata_path,
        json.dumps(initial_metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    context.write_resolved_config(resolved_config)
    _atomic_write_text(context.metrics_path, "")
    return context

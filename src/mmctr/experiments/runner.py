"""Failure-isolated, resumable experiment matrix orchestration."""

import hashlib
import json
import math
import os
import queue
import re
import tempfile
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from mmctr.core import ContractError, RunResult
from mmctr.utils.run_context import RunContext, create_run_context


RESULT_SCHEMA_VERSION = 1
MATRIX_SCHEMA_VERSION = 1
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(name): _freeze(item) for name, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(name): _plain(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True)
class ExperimentTask:
    """One independently resumable run in an experiment matrix."""

    task_id: str
    dataset: str
    model: str
    seed: int
    resolved_config: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("task_id", "dataset", "model"):
            value = str(getattr(self, name))
            if not _SAFE_NAME.fullmatch(value):
                raise ContractError("{} must be a safe non-empty name".format(name))
            object.__setattr__(self, name, value)
        if isinstance(self.seed, bool):
            raise ContractError("experiment seed must be an integer")
        if not isinstance(self.resolved_config, Mapping):
            raise ContractError("resolved_config must be a mapping")
        config = _freeze(self.resolved_config)
        fingerprint = config.get("data_fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ContractError("resolved_config must contain a data_fingerprint")
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "resolved_config", config)

    @property
    def data_fingerprint(self) -> str:
        return str(self.resolved_config["data_fingerprint"])

    @property
    def key(self) -> str:
        payload = {
            "task_id": self.task_id,
            "dataset": self.dataset,
            "model": self.model,
            "seed": self.seed,
            "resolved_config": _plain(self.resolved_config),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


ExperimentExecutor = Callable[[ExperimentTask, RunContext, str], Mapping[str, float]]


def _atomic_json(path: Path, values: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=".{}-".format(path.name),
            suffix=".tmp",
            dir=str(path.parent),
            delete=False,
        ) as stream:
            json.dump(values, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            temporary_path = Path(stream.name)
        os.replace(str(temporary_path), str(path))
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _result_payload(task: ExperimentTask, result: RunResult, device: str) -> Dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "task_id": task.task_id,
        "task_key": task.key,
        "run_id": result.run_id,
        "status": result.status,
        "dataset": task.dataset,
        "model": task.model,
        "seed": task.seed,
        "device": device,
        "data_fingerprint": task.data_fingerprint,
        "metrics": dict(result.metrics),
        "artifact_dir": str(result.artifact_dir) if result.artifact_dir is not None else None,
        "error": result.error,
        "metadata": dict(result.metadata),
    }


def _load_result(path: Path) -> RunResult:
    values = json.loads(path.read_text(encoding="utf-8"))
    if values.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ContractError("unsupported experiment result schema")
    return RunResult(
        run_id=str(values["run_id"]),
        status=str(values["status"]),
        metrics=values.get("metrics", {}),
        artifact_dir=values.get("artifact_dir"),
        error=values.get("error"),
        metadata=values.get("metadata", {}),
    )


class ExperimentRunner:
    """Run a task matrix with one active worker per explicitly named device."""

    def __init__(
        self,
        output_root: Path,
        executor: ExperimentExecutor,
        devices: Sequence[str] = ("cpu",),
        max_workers: Optional[int] = None,
        repository_root: Optional[Path] = None,
    ) -> None:
        if not callable(executor):
            raise ContractError("experiment executor must be callable")
        normalised_devices = tuple(str(device).strip() for device in devices)
        if not normalised_devices or any(not device for device in normalised_devices):
            raise ContractError("devices must contain at least one non-empty device")
        if len(set(normalised_devices)) != len(normalised_devices):
            raise ContractError("experiment devices must be unique")
        workers = len(normalised_devices) if max_workers is None else int(max_workers)
        if workers <= 0 or workers > len(normalised_devices):
            raise ContractError("max_workers must be between one and the device count")
        self.output_root = Path(output_root).expanduser().resolve()
        self.executor = executor
        self.devices = normalised_devices
        self.max_workers = workers
        self.repository_root = (
            Path(repository_root).expanduser().resolve() if repository_root is not None else None
        )
        self._lock = threading.Lock()

    def _matrix_path(self, experiment_name: str) -> Path:
        if not _SAFE_NAME.fullmatch(str(experiment_name)):
            raise ContractError("experiment_name must be a safe non-empty name")
        return self.output_root / ".matrices" / (str(experiment_name) + ".json")

    def _load_matrix(self, experiment_name: str) -> Dict[str, Any]:
        path = self._matrix_path(experiment_name)
        if not path.is_file():
            return {
                "schema_version": MATRIX_SCHEMA_VERSION,
                "experiment_name": experiment_name,
                "tasks": {},
            }
        values = json.loads(path.read_text(encoding="utf-8"))
        if (
            values.get("schema_version") != MATRIX_SCHEMA_VERSION
            or values.get("experiment_name") != experiment_name
            or not isinstance(values.get("tasks"), dict)
        ):
            raise ContractError("experiment matrix state is incompatible")
        return values

    def _save_matrix_result(
        self,
        experiment_name: str,
        matrix: Dict[str, Any],
        task: ExperimentTask,
        result_path: Path,
        result: RunResult,
    ) -> None:
        with self._lock:
            matrix["tasks"][task.key] = {
                "task_id": task.task_id,
                "status": result.status,
                "run_id": result.run_id,
                "result_path": str(result_path),
            }
            _atomic_json(self._matrix_path(experiment_name), matrix)

    def _execute_one(
        self,
        experiment_name: str,
        task: ExperimentTask,
        matrix: Dict[str, Any],
        available_devices: "queue.Queue[str]",
    ) -> RunResult:
        device = available_devices.get()
        try:
            context = create_run_context(
                output_root=self.output_root,
                experiment_name=experiment_name,
                dataset=task.dataset,
                model=task.model,
                resolved_config=task.resolved_config,
                repository_root=self.repository_root,
                metadata={
                    "task_id": task.task_id,
                    "task_key": task.key,
                    "seed": task.seed,
                    "device": device,
                    "data_fingerprint": task.data_fingerprint,
                },
            )
        except BaseException:
            available_devices.put(device)
            raise
        result_path = context.root_dir / "result.json"
        try:
            raw_metrics = self.executor(task, context, device)
            if not isinstance(raw_metrics, Mapping):
                raise ContractError("experiment executor must return a metric mapping")
            metrics = {str(name): float(value) for name, value in raw_metrics.items()}
            if any(not math.isfinite(value) for value in metrics.values()):
                raise ContractError("experiment executor returned a non-finite metric")
            result = RunResult(
                run_id=context.run_id,
                status="completed",
                metrics=metrics,
                artifact_dir=context.root_dir,
                metadata={"task_id": task.task_id, "device": device, "seed": task.seed},
            )
            _atomic_json(result_path, _result_payload(task, result, device))
            context.finalize("completed", summary={"metrics": metrics})
        except Exception as error:  # one failed task must not abort sibling runs
            result = RunResult(
                run_id=context.run_id,
                status="failed",
                artifact_dir=context.root_dir,
                error="{}: {}".format(type(error).__name__, error),
                metadata={"task_id": task.task_id, "device": device, "seed": task.seed},
            )
            _atomic_json(result_path, _result_payload(task, result, device))
            context.finalize("failed", summary={"metrics": {}}, error=result.error)
        finally:
            available_devices.put(device)
        self._save_matrix_result(experiment_name, matrix, task, result_path, result)
        return result

    def run(
        self,
        experiment_name: str,
        tasks: Sequence[ExperimentTask],
        resume: bool = True,
    ) -> Tuple[RunResult, ...]:
        """Execute pending tasks and return results in the caller's task order."""

        task_tuple = tuple(tasks)
        if not task_tuple or any(not isinstance(task, ExperimentTask) for task in task_tuple):
            raise ContractError("tasks must contain at least one ExperimentTask")
        task_ids = [task.task_id for task in task_tuple]
        task_keys = [task.key for task in task_tuple]
        if len(set(task_ids)) != len(task_ids) or len(set(task_keys)) != len(task_keys):
            raise ContractError("experiment task IDs and identities must be unique")
        matrix = self._load_matrix(experiment_name)
        results: Dict[str, RunResult] = {}
        pending = []
        for task in task_tuple:
            previous = matrix["tasks"].get(task.key) if resume else None
            if previous is not None and previous.get("status") == "completed":
                result_path = Path(previous["result_path"])
                result = _load_result(result_path)
                if result.status != "completed" or result.metadata.get("task_id") != task.task_id:
                    raise ContractError("completed experiment result does not match its task")
                results[task.key] = result
            else:
                pending.append(task)

        available_devices: "queue.Queue[str]" = queue.Queue()
        for device in self.devices:
            available_devices.put(device)
        futures: Dict[str, Future] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for task in pending:
                futures[task.key] = pool.submit(
                    self._execute_one,
                    experiment_name,
                    task,
                    matrix,
                    available_devices,
                )
            for task in pending:
                results[task.key] = futures[task.key].result()
        return tuple(results[task.key] for task in task_tuple)


__all__ = [
    "ExperimentExecutor",
    "ExperimentRunner",
    "ExperimentTask",
    "MATRIX_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
]

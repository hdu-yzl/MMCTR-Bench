"""Versioned, integrity-checked persistence for immutable experiment tasks."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

from mmctr.core import ContractError

from .runner import ExperimentTask


PathLike = Union[str, Path]


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(name): _plain(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _task_payload(task: ExperimentTask) -> Dict[str, Any]:
    return {
        "task_id": task.task_id,
        "dataset": task.dataset,
        "model": task.model,
        "seed": task.seed,
        "resolved_config": _plain(task.resolved_config),
    }


def _fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save_task_matrix(
    tasks: Sequence[ExperimentTask],
    path: PathLike,
    schema: str,
) -> Path:
    """Atomically persist tasks under an analysis-specific schema name."""

    if not isinstance(schema, str) or not schema:
        raise ContractError("experiment task matrix schema must be non-empty")
    task_values = tuple(tasks)
    if not task_values or any(not isinstance(task, ExperimentTask) for task in task_values):
        raise ContractError("experiment task matrix requires at least one ExperimentTask")
    task_keys = [task.key for task in task_values]
    task_ids = [task.task_id for task in task_values]
    if len(set(task_keys)) != len(task_keys) or len(set(task_ids)) != len(task_ids):
        raise ContractError("experiment task matrix identities must be unique")
    body = {
        "schema": schema,
        "task_count": len(task_values),
        "tasks": [_task_payload(task) for task in task_values],
    }
    payload = dict(body)
    payload["fingerprint"] = _fingerprint(body)
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=".{}-".format(target.name),
            suffix=".tmp",
            dir=str(target.parent),
            delete=False,
        ) as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            temporary_path = Path(stream.name)
        os.replace(str(temporary_path), str(target))
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return target


def load_task_matrix(path: PathLike, schema: str) -> Tuple[ExperimentTask, ...]:
    """Load tasks only when their schema, count, and content fingerprint agree."""

    target = Path(path).expanduser().resolve()
    values = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(values, dict) or values.get("schema") != schema:
        raise ContractError("unsupported experiment task matrix schema")
    fingerprint = values.pop("fingerprint", None)
    if not isinstance(fingerprint, str) or fingerprint != _fingerprint(values):
        raise ContractError("experiment task matrix fingerprint mismatch")
    raw_tasks = values.get("tasks")
    if not isinstance(raw_tasks, list) or values.get("task_count") != len(raw_tasks):
        raise ContractError("experiment task matrix count mismatch")
    try:
        tasks = tuple(
            ExperimentTask(
                task_id=item["task_id"],
                dataset=item["dataset"],
                model=item["model"],
                seed=item["seed"],
                resolved_config=item["resolved_config"],
            )
            for item in raw_tasks
        )
    except (KeyError, TypeError) as error:
        raise ContractError("experiment task matrix contains an invalid task") from error
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ContractError("experiment task matrix identities must be unique")
    return tasks


__all__ = ["load_task_matrix", "save_task_matrix"]

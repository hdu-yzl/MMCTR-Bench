"""Isolated atomic model and optimizer checkpoints."""

import os
from copy import deepcopy
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import torch


PathLike = Union[str, os.PathLike]
CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CheckpointState:
    epoch: int
    metric: Optional[float]
    metadata: Mapping[str, Any]


class CheckpointManager:
    """Own `best.pt` and `last.pt` inside one run-specific directory."""

    def __init__(self, directory: PathLike, persist: bool = True) -> None:
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.persist = persist
        self._best_payload: Optional[Dict[str, Any]] = None

    @property
    def best_path(self) -> Path:
        return self.directory / "best.pt"

    @property
    def last_path(self) -> Path:
        return self.directory / "last.pt"

    def save(
        self,
        name: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        epoch: int,
        metric: Optional[float] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        path = self._path(name)
        if not self.persist:
            # Last/optimizer state is unnecessary when interrupted runs restart.
            if name == "best":
                state = model.state_dict()
                cpu_state = deepcopy(state)
                for key, value in state.items():
                    cpu_state[key] = value.detach().cpu().clone()
                self._best_payload = {
                    "epoch": int(epoch),
                    "metric": float(metric) if metric is not None else None,
                    "model_state": cpu_state,
                    "metadata": deepcopy(dict(metadata or {})),
                }
            return path
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "epoch": int(epoch),
            "metric": float(metric) if metric is not None else None,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "metadata": dict(metadata or {}),
        }
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=".{}-".format(path.name),
                suffix=".tmp",
                dir=str(path.parent),
                delete=False,
            ) as handle:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
                temporary_path = Path(handle.name)
            temporary_path.replace(path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
        return path

    def restore(
        self,
        name: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        map_location: Optional[torch.device] = None,
    ) -> CheckpointState:
        """Restore a supported checkpoint and optionally its optimizer state."""

        path = self._path(name)
        if not self.persist:
            if name != "best" or optimizer is not None:
                raise ValueError("memory-only checkpoints cannot resume optimizer/last state")
            if self._best_payload is None:
                raise FileNotFoundError("no validation-selected weights in memory")
            payload = self._best_payload
            model.load_state_dict(payload["model_state"])
            return CheckpointState(payload["epoch"], payload["metric"], payload["metadata"])
        if not path.is_file():
            raise FileNotFoundError("checkpoint not found: {}".format(path))
        payload = torch.load(path, map_location=map_location)
        if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported checkpoint schema")
        model.load_state_dict(payload["model_state"])
        optimizer_state = payload.get("optimizer_state")
        if optimizer is not None and optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        return CheckpointState(
            epoch=int(payload["epoch"]),
            metric=payload.get("metric"),
            metadata=dict(payload.get("metadata", {})),
        )

    def _path(self, name: str) -> Path:
        if name == "best":
            return self.best_path
        if name == "last":
            return self.last_path
        raise ValueError("checkpoint name must be 'best' or 'last'")


__all__ = ["CheckpointManager", "CheckpointState"]

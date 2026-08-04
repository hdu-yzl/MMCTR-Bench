"""Isolated atomic model and optimizer checkpoints."""

import os
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

    def __init__(self, directory: PathLike) -> None:
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

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
        if not path.is_file():
            raise FileNotFoundError("checkpoint not found: {}".format(path))
        payload: Dict[str, Any] = torch.load(path, map_location=map_location)
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

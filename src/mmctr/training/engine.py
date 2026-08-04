"""Canonical training engine independent of model architecture and dataset format."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import torch

from mmctr.core import Batch, ContractError, RunResult, ensure_model_output
from mmctr.data import DataLoaderProtocol
from mmctr.evaluation import BinaryClassificationEvaluator, BinaryMetrics

from .callbacks import EarlyStopping
from .checkpointing import CheckpointManager
from .optimizers import PhasedAdam


MetricWriter = Callable[[Mapping[str, object]], None]


@dataclass(frozen=True)
class AlternatingPhase:
    """One named optimizer phase activated from an inclusive epoch index."""

    name: str
    start_epoch: int
    objective: Callable[[Batch], torch.Tensor]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or self.name == "main":
            raise ValueError("alternating phase name must be non-empty and not 'main'")
        if int(self.start_epoch) < 0:
            raise ValueError("alternating phase start_epoch must be non-negative")
        if not callable(self.objective):
            raise ValueError("alternating phase objective must be callable")
        object.__setattr__(self, "start_epoch", int(self.start_epoch))


@dataclass(frozen=True)
class EpochResult:
    epoch: int
    split: str
    loss: float
    batches: int
    samples: int
    duration_seconds: float
    metrics: Optional[BinaryMetrics] = None

    def to_dict(self) -> Dict[str, object]:
        result: Dict[str, object] = {
            "epoch": self.epoch,
            "split": self.split,
            "loss": self.loss,
            "batches": self.batches,
            "samples": self.samples,
            "duration_seconds": self.duration_seconds,
        }
        if self.metrics is not None:
            result.update(self.metrics.to_dict())
        return result


class TrainingEngine:
    """Train models that implement ``forward(Batch) -> ModelOutput``."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        checkpoint_manager: CheckpointManager,
        auxiliary_loss_weights: Optional[Mapping[str, float]] = None,
        gradient_clip_norm: Optional[float] = None,
        logger: Optional[logging.Logger] = None,
        metric_writer: Optional[MetricWriter] = None,
        alternating_phases: Sequence[AlternatingPhase] = (),
    ) -> None:
        if gradient_clip_norm is not None and gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.checkpoints = checkpoint_manager
        self.auxiliary_loss_weights = dict(auxiliary_loss_weights or {})
        self.gradient_clip_norm = gradient_clip_norm
        self.logger = logger or logging.getLogger(__name__)
        self.metric_writer = metric_writer
        self.alternating_phases = tuple(alternating_phases)
        if any(not isinstance(phase, AlternatingPhase) for phase in self.alternating_phases):
            raise ValueError("alternating_phases must contain AlternatingPhase objects")
        phase_names = tuple(phase.name for phase in self.alternating_phases)
        if len(set(phase_names)) != len(phase_names):
            raise ValueError("alternating phase names must be unique")
        if self.alternating_phases:
            if not isinstance(self.optimizer, PhasedAdam):
                raise ValueError("alternating phases require a PhasedAdam optimizer")
            missing_phases = set(phase_names).difference(self.optimizer.phases)
            if missing_phases:
                raise ValueError(
                    "alternating phases are missing optimizer groups: {}".format(
                        sorted(missing_phases)
                    )
                )
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.evaluator = BinaryClassificationEvaluator(device)
        self._resume_metadata: Dict[str, Any] = {}

    def train_epoch(self, batches, epoch: int) -> EpochResult:
        self.model.train()
        started_at = time.perf_counter()
        total_loss = 0.0
        total_samples = 0
        batch_count = 0
        for batch in batches:
            if not isinstance(batch, Batch):
                raise ContractError("training engine requires canonical Batch instances")
            device_batch = batch.to(self.device)
            self.optimizer.zero_grad()
            output = ensure_model_output(self.model(device_batch))
            if output.batch_size != device_batch.batch_size:
                raise ContractError("model output and batch sizes do not match")
            loss = self.criterion(output.logits, device_batch.labels)
            loss = loss + output.auxiliary_loss(self.auxiliary_loss_weights)
            loss.backward()
            if self.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
            self.optimizer.step()
            batch_loss = loss.detach()
            for phase in self.alternating_phases:
                if epoch < phase.start_epoch:
                    continue
                self.optimizer.zero_grad()
                phase_loss = phase.objective(device_batch)
                if not isinstance(phase_loss, torch.Tensor) or phase_loss.ndim != 0:
                    raise ContractError(
                        "alternating phase {!r} must return a scalar Tensor".format(phase.name)
                    )
                if not torch.isfinite(phase_loss):
                    raise ContractError(
                        "alternating phase {!r} returned a non-finite loss".format(phase.name)
                    )
                phase_loss.backward()
                self.optimizer.step_phase(phase.name)
                batch_loss = batch_loss + phase_loss.detach()
            samples = device_batch.batch_size
            total_loss += float(batch_loss.item()) * samples
            total_samples += samples
            batch_count += 1
        if batch_count == 0:
            raise ContractError("training split is empty")
        result = EpochResult(
            epoch=epoch,
            split="train",
            loss=total_loss / total_samples,
            batches=batch_count,
            samples=total_samples,
            duration_seconds=time.perf_counter() - started_at,
        )
        self._record(result)
        return result

    def evaluate(self, loader: DataLoaderProtocol, split: str, epoch: int) -> EpochResult:
        if split == "train":
            raise ContractError("use train_epoch for the training split")
        started_at = time.perf_counter()
        metrics = self.evaluator.evaluate(self.model, loader.iter_batches(split))
        result = EpochResult(
            epoch=epoch,
            split=split,
            loss=metrics.log_loss,
            batches=0,
            samples=metrics.samples,
            duration_seconds=time.perf_counter() - started_at,
            metrics=metrics,
        )
        self._record(result)
        return result

    def fit(
        self,
        loader: DataLoaderProtocol,
        max_epochs: int,
        early_stop_patience: int,
        run_id: str,
        artifact_dir: Optional[Path] = None,
        start_epoch: int = 0,
    ) -> RunResult:
        """Fit using train/validation only and restore the best checkpoint."""

        if max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        stopper = EarlyStopping(early_stop_patience)
        if start_epoch > 0 and self._resume_metadata:
            resume_run_id = self._resume_metadata.get("run_id")
            if resume_run_id is not None and resume_run_id != run_id:
                raise ContractError("resume checkpoint belongs to a different run")
            resume_patience = self._resume_metadata.get("early_stop_patience")
            if resume_patience is not None and int(resume_patience) != early_stop_patience:
                raise ContractError("resume early-stop patience does not match")
            if self._resume_metadata.get("best") is not None:
                stopper.best = float(self._resume_metadata["best"])
            if self._resume_metadata.get("best_epoch") is not None:
                stopper.best_epoch = int(self._resume_metadata["best_epoch"])
            stopper.bad_epochs = int(self._resume_metadata.get("bad_epochs", 0))
        last_epoch = start_epoch - 1
        epoch_range = () if stopper.should_stop else range(start_epoch, max_epochs)
        for epoch in epoch_range:
            self.train_epoch(loader.iter_batches("train"), epoch)
            validation = self.evaluate(loader, "val", epoch)
            if validation.metrics is None:
                raise RuntimeError("validation metrics were not produced")
            last_epoch = epoch
            improved = stopper.step(validation.metrics.auc, epoch)
            checkpoint_metadata = {
                "split": "val",
                "run_id": run_id,
                "best": stopper.best,
                "best_epoch": stopper.best_epoch,
                "bad_epochs": stopper.bad_epochs,
                "early_stop_patience": early_stop_patience,
            }
            self.checkpoints.save(
                "last",
                self.model,
                self.optimizer,
                epoch,
                validation.metrics.auc,
                checkpoint_metadata,
            )
            if improved:
                self.checkpoints.save(
                    "best",
                    self.model,
                    self.optimizer,
                    epoch,
                    validation.metrics.auc,
                    checkpoint_metadata,
                )
            if stopper.should_stop:
                self.logger.info("early_stop epoch=%d best_epoch=%d", epoch, stopper.best_epoch)
                break
        if stopper.best is None or stopper.best_epoch is None:
            raise RuntimeError("training completed without a validation result")
        self.checkpoints.restore("best", self.model, map_location=self.device)
        return RunResult(
            run_id=run_id,
            status="completed",
            metrics={
                "best_val_auc": stopper.best,
                "best_epoch": float(stopper.best_epoch),
                "last_epoch": float(last_epoch),
            },
            artifact_dir=artifact_dir or self.checkpoints.directory.parent,
        )

    def resume(self) -> int:
        """Restore `last.pt` and return the next epoch index."""

        state = self.checkpoints.restore(
            "last", self.model, optimizer=self.optimizer, map_location=self.device
        )
        self._resume_metadata = dict(state.metadata)
        return state.epoch + 1

    def _record(self, result: EpochResult) -> None:
        values = result.to_dict()
        self.logger.info(
            "epoch=%d split=%s loss=%.6f samples=%d duration=%.3fs",
            result.epoch,
            result.split,
            result.loss,
            result.samples,
            result.duration_seconds,
        )
        if self.metric_writer is not None:
            self.metric_writer(values)


__all__ = ["AlternatingPhase", "EpochResult", "TrainingEngine"]

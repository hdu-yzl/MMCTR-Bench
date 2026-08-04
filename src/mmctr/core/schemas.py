"""Typed contracts shared by data, models, training, and experiments."""

import math
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import torch


TensorMap = Mapping[str, torch.Tensor]
MetadataMap = Mapping[str, Any]
DeviceLike = Union[str, torch.device]
RUN_STATUSES = frozenset({"running", "completed", "failed", "cancelled"})


class ContractError(ValueError):
    """Raised when an object does not satisfy a public MMCTR contract."""


def _readonly_copy(values: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    if not isinstance(values, Mapping):
        raise ContractError("{} must be a mapping".format(field_name))
    copied = dict(values)
    for key in copied:
        if not isinstance(key, str) or not key:
            raise ContractError("{} keys must be non-empty strings".format(field_name))
    return MappingProxyType(copied)


def _tensor_map(values: TensorMap, field_name: str) -> TensorMap:
    copied = _readonly_copy(values, field_name)
    for name, value in copied.items():
        if not isinstance(value, torch.Tensor):
            raise ContractError("{}.{} must be a torch.Tensor".format(field_name, name))
        if value.ndim == 0:
            raise ContractError("{}.{} must include a batch dimension".format(field_name, name))
        is_id = name == "id" or name.endswith("_id") or name.endswith("_ids")
        if is_id and value.dtype != torch.long:
            raise ContractError("{}.{} must use torch.long".format(field_name, name))
        if not is_id and not (value.is_floating_point() or value.is_complex()):
            raise ContractError("{}.{} must use a floating dtype".format(field_name, name))
    return copied


def _require_batch_size(values: TensorMap, batch_size: int, field_name: str) -> None:
    for name, value in values.items():
        if value.shape[0] != batch_size:
            raise ContractError(
                "{}.{} batch size {} does not match labels batch size {}".format(
                    field_name, name, value.shape[0], batch_size
                )
            )


def _normalise_labels(labels: torch.Tensor) -> torch.Tensor:
    if not isinstance(labels, torch.Tensor):
        raise ContractError("labels must be a torch.Tensor")
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels.squeeze(1)
    if labels.ndim != 1:
        raise ContractError("labels must have shape [B]")
    if not labels.is_floating_point():
        raise ContractError("labels must use a floating dtype")
    return labels


def _history_mask_from_ids(history_ids: torch.Tensor, padding_id: int) -> torch.Tensor:
    if history_ids.ndim < 2:
        raise ContractError("history id tensor must have shape [B, L, ...]")
    mask = history_ids.ne(padding_id)
    while mask.ndim > 2:
        mask = mask.any(dim=-1)
    return mask


@dataclass(frozen=True)
class Batch:
    """Canonical input batch.

    The dataclass and its mappings are immutable at the top level. Tensor storage is
    intentionally not copied. Call :meth:`to` to create a device-specific batch.
    """

    user_features: TensorMap
    item_features: TensorMap
    history_features: TensorMap
    history_mask: torch.Tensor
    labels: torch.Tensor
    metadata: MetadataMap = field(default_factory=dict)
    context_features: TensorMap = field(default_factory=dict)

    def __post_init__(self) -> None:
        user_features = _tensor_map(self.user_features, "user_features")
        item_features = _tensor_map(self.item_features, "item_features")
        history_features = _tensor_map(self.history_features, "history_features")
        context_features = _tensor_map(self.context_features, "context_features")
        labels = _normalise_labels(self.labels)
        metadata = _readonly_copy(self.metadata, "metadata")

        if not item_features:
            raise ContractError("item_features must contain at least one feature")
        if not history_features:
            raise ContractError("history_features must contain at least one feature")
        if not isinstance(self.history_mask, torch.Tensor):
            raise ContractError("history_mask must be a torch.Tensor")
        if self.history_mask.dtype != torch.bool:
            raise ContractError("history_mask must use torch.bool")
        if self.history_mask.ndim != 2:
            raise ContractError("history_mask must have shape [B, L]")

        batch_size = labels.shape[0]
        sequence_length = self.history_mask.shape[1]
        if self.history_mask.shape[0] != batch_size:
            raise ContractError("history_mask and labels batch sizes do not match")
        _require_batch_size(user_features, batch_size, "user_features")
        _require_batch_size(item_features, batch_size, "item_features")
        _require_batch_size(history_features, batch_size, "history_features")
        _require_batch_size(context_features, batch_size, "context_features")
        for name, value in history_features.items():
            if value.ndim < 2 or value.shape[1] != sequence_length:
                raise ContractError("history_features.{} must start with shape [B, L]".format(name))

        object.__setattr__(self, "user_features", user_features)
        object.__setattr__(self, "item_features", item_features)
        object.__setattr__(self, "history_features", history_features)
        object.__setattr__(self, "context_features", context_features)
        object.__setattr__(self, "history_mask", self.history_mask)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "metadata", metadata)

    @property
    def batch_size(self) -> int:
        return int(self.labels.shape[0])

    @property
    def sequence_length(self) -> int:
        return int(self.history_mask.shape[1])

    def to(self, device: DeviceLike, non_blocking: bool = False) -> "Batch":
        """Return a new batch with all tensors moved to ``device``."""

        def move(values: TensorMap) -> Dict[str, torch.Tensor]:
            return {
                name: value.to(device=device, non_blocking=non_blocking)
                for name, value in values.items()
            }

        return Batch(
            user_features=move(self.user_features),
            item_features=move(self.item_features),
            history_features=move(self.history_features),
            history_mask=self.history_mask.to(device=device, non_blocking=non_blocking),
            labels=self.labels.to(device=device, non_blocking=non_blocking),
            context_features=move(self.context_features),
            metadata=self.metadata,
        )

    @classmethod
    def from_legacy(
        cls,
        values: Sequence[Any],
        history_mask: Optional[torch.Tensor] = None,
        padding_id: int = 0,
        metadata: Optional[MetadataMap] = None,
    ) -> "Batch":
        """Adapt the legacy 3-tuple or 4-tuple loader result explicitly."""

        if len(values) == 3:
            user_features: TensorMap = {}
            item_features, history_features, labels = values
        elif len(values) == 4:
            user_features, item_features, history_features, labels = values
        else:
            raise ContractError("legacy batch must contain 3 or 4 elements")
        if not isinstance(history_features, Mapping):
            raise ContractError("legacy history features must be a mapping")
        if history_mask is None:
            try:
                history_ids = history_features["id"]
            except KeyError as error:
                raise ContractError(
                    "history_mask is required when legacy history has no 'id' feature"
                ) from error
            history_mask = _history_mask_from_ids(history_ids, padding_id)
        return cls(
            user_features=user_features,
            item_features=item_features,
            history_features=history_features,
            history_mask=history_mask,
            labels=labels,
            metadata=metadata or {},
        )


@dataclass(frozen=True)
class ModelOutput:
    """Canonical model result with logits and named optional training outputs."""

    logits: torch.Tensor
    auxiliary_losses: Mapping[str, torch.Tensor] = field(default_factory=dict)
    representations: Optional[TensorMap] = None

    def __post_init__(self) -> None:
        logits = self.logits
        if not isinstance(logits, torch.Tensor):
            raise ContractError("logits must be a torch.Tensor")
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        if logits.ndim != 1:
            raise ContractError("logits must have shape [B]")
        if not logits.is_floating_point():
            raise ContractError("logits must use a floating dtype")

        losses = _tensor_map_allow_scalars(self.auxiliary_losses, "auxiliary_losses")
        for name, loss in losses.items():
            if loss.ndim != 0:
                raise ContractError("auxiliary loss {!r} must be scalar".format(name))

        representations = None
        if self.representations is not None:
            representations = _tensor_map(self.representations, "representations")
            _require_batch_size(representations, logits.shape[0], "representations")

        object.__setattr__(self, "logits", logits)
        object.__setattr__(self, "auxiliary_losses", losses)
        object.__setattr__(self, "representations", representations)

    @property
    def batch_size(self) -> int:
        return int(self.logits.shape[0])

    def auxiliary_loss(self, weights: Optional[Mapping[str, float]] = None) -> torch.Tensor:
        """Return the weighted sum of all named auxiliary losses."""

        if not self.auxiliary_losses:
            return self.logits.new_zeros(())
        configured = dict(weights or {})
        unknown = set(configured).difference(self.auxiliary_losses)
        if unknown:
            raise ContractError("unknown auxiliary loss weights: {}".format(sorted(unknown)))
        total = self.logits.new_zeros(())
        for name, loss in self.auxiliary_losses.items():
            total = total + loss * float(configured.get(name, 1.0))
        return total

    def as_legacy(self) -> Dict[str, torch.Tensor]:
        """Return the temporary legacy ``pred``/``au_loss`` mapping."""

        result = {"pred": self.logits}
        if self.auxiliary_losses:
            result["au_loss"] = self.auxiliary_loss()
        return result

    @classmethod
    def from_legacy(cls, value: Any) -> "ModelOutput":
        """Adapt a legacy output tensor or result mapping."""

        if isinstance(value, torch.Tensor):
            return cls(logits=value)
        if not isinstance(value, Mapping):
            raise ContractError("model output must be a tensor or mapping")
        logits = value.get("logits", value.get("pred"))
        if logits is None:
            raise ContractError("legacy model output must contain 'pred' or 'logits'")
        losses: Dict[str, torch.Tensor] = {}
        legacy_loss = value.get("au_loss")
        if legacy_loss is not None:
            if not isinstance(legacy_loss, torch.Tensor):
                legacy_loss = logits.new_tensor(float(legacy_loss))
            losses["legacy_auxiliary"] = legacy_loss
        configured_losses = value.get("auxiliary_losses")
        if configured_losses is not None:
            if not isinstance(configured_losses, Mapping):
                raise ContractError("auxiliary_losses must be a mapping")
            losses.update(configured_losses)
        representations = value.get("representations")
        return cls(logits=logits, auxiliary_losses=losses, representations=representations)


def _tensor_map_allow_scalars(values: TensorMap, field_name: str) -> TensorMap:
    copied = _readonly_copy(values, field_name)
    for name, value in copied.items():
        if not isinstance(value, torch.Tensor):
            raise ContractError("{}.{} must be a torch.Tensor".format(field_name, name))
        if not value.is_floating_point():
            raise ContractError("{}.{} must use a floating dtype".format(field_name, name))
    return copied


@dataclass(frozen=True)
class RunResult:
    """Frozen outcome shared by training, experiment, and analysis boundaries.

    Status and finite metrics are normalized at construction, while metadata is exposed
    through a read-only mapping so downstream consumers see a stable result contract.
    """

    run_id: str
    status: str
    metrics: Mapping[str, float] = field(default_factory=dict)
    artifact_dir: Optional[Path] = None
    error: Optional[str] = None
    metadata: MetadataMap = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id or not isinstance(self.run_id, str):
            raise ContractError("run_id must be a non-empty string")
        status = str(self.status).lower()
        if status not in RUN_STATUSES:
            raise ContractError("invalid run status: {!r}".format(self.status))
        metrics = _readonly_copy(self.metrics, "metrics")
        normalised_metrics: Dict[str, float] = {}
        for name, value in metrics.items():
            if isinstance(value, bool):
                raise ContractError("metric {!r} must be numeric".format(name))
            try:
                metric_value = float(value)
            except (TypeError, ValueError) as error:
                raise ContractError("metric {!r} must be numeric".format(name)) from error
            if not math.isfinite(metric_value):
                raise ContractError("metric {!r} must be finite".format(name))
            normalised_metrics[name] = metric_value
        artifact_dir = None
        if self.artifact_dir is not None:
            artifact_dir = Path(self.artifact_dir).expanduser().resolve()
        if status == "completed" and self.error is not None:
            raise ContractError("completed runs cannot contain an error")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "metrics", MappingProxyType(normalised_metrics))
        object.__setattr__(self, "artifact_dir", artifact_dir)
        object.__setattr__(self, "metadata", _readonly_copy(self.metadata, "metadata"))

    @property
    def succeeded(self) -> bool:
        return self.status == "completed"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "metrics": dict(self.metrics),
            "artifact_dir": str(self.artifact_dir) if self.artifact_dir else None,
            "error": self.error,
            "metadata": dict(self.metadata),
        }


def ensure_model_output(value: Any) -> ModelOutput:
    """Return ``value`` as a validated :class:`ModelOutput`."""

    if isinstance(value, ModelOutput):
        return value
    return ModelOutput.from_legacy(value)


__all__ = [
    "Batch",
    "ContractError",
    "ModelOutput",
    "RUN_STATUSES",
    "RunResult",
    "ensure_model_output",
]

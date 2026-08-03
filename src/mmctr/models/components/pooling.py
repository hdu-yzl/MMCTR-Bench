"""Registered sequence pooling components with one strict public signature."""

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from mmctr.core import ContractError
from mmctr.models.components.masking import (
    apply_feature_mask,
    apply_sequence_mask,
    masked_softmax,
    validate_sequence_mask,
)


@dataclass(frozen=True)
class PoolingCapability:
    """Declarative shape and dependency contract for one pooling component."""

    input_rank: int = 3
    output_rank: int = 2
    mask_required: bool = True
    target_required: bool = False
    output_dim_rule: str = "preserves"


class SequencePooling(torch.nn.Module):
    """Base contract for ``[B,L,D] -> [B,D]`` sequence pooling."""

    capability = PoolingCapability()

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.input_dim = int(dimension)
        self.output_dim = int(dimension)
        if self.input_dim <= 0:
            raise ContractError("pooling dimension must be positive")

    def _validate_inputs(
        self,
        sequence: torch.Tensor,
        mask: torch.Tensor,
        target: Optional[torch.Tensor],
    ) -> None:
        if not isinstance(sequence, torch.Tensor) or not sequence.is_floating_point():
            raise ContractError("pooling sequence must be a floating torch.Tensor")
        if sequence.ndim != self.capability.input_rank:
            raise ContractError("pooling sequence must have shape [B, L, D]")
        if sequence.shape[-1] != self.input_dim:
            raise ContractError(
                "pooling expected final dimension {}, got {}".format(
                    self.input_dim, sequence.shape[-1]
                )
            )
        validate_sequence_mask(mask, sequence)
        if self.capability.target_required and target is None:
            raise ContractError("this pooling component requires a target tensor")
        if target is not None:
            if not isinstance(target, torch.Tensor) or not target.is_floating_point():
                raise ContractError("pooling target must be a floating torch.Tensor")
            if target.ndim != 2 or tuple(target.shape) != (
                sequence.shape[0],
                self.input_dim,
            ):
                raise ContractError("pooling target must have shape [B, D]")
            if target.device != sequence.device or target.dtype != sequence.dtype:
                raise ContractError("pooling sequence and target must share dtype/device")


def masked_reduce(
    sequence: torch.Tensor,
    mask: torch.Tensor,
    reduction: str,
) -> torch.Tensor:
    """Implement sum/mean/max with explicit all-padding behavior."""

    if not isinstance(sequence, torch.Tensor) or not sequence.is_floating_point():
        raise ContractError("pooling sequence must be a floating torch.Tensor")
    if sequence.ndim != 3:
        raise ContractError("pooling sequence must have shape [B, L, D]")
    validate_sequence_mask(mask, sequence)
    masked = apply_sequence_mask(sequence, mask)
    if reduction == "sum":
        return masked.sum(dim=1)
    if reduction == "mean":
        counts = mask.sum(dim=1, keepdim=True).clamp_min(1)
        return masked.sum(dim=1) / counts.to(dtype=sequence.dtype)
    if reduction == "max":
        expanded = mask.unsqueeze(-1)
        minimum = torch.finfo(sequence.dtype).min
        result = sequence.masked_fill(~expanded, minimum).max(dim=1).values
        return apply_feature_mask(result, mask.any(dim=1))
    raise ContractError("unsupported pooling reduction: {!r}".format(reduction))


class MeanPooling(SequencePooling):
    def forward(
        self,
        sequence: torch.Tensor,
        mask: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._validate_inputs(sequence, mask, target)
        return masked_reduce(sequence, mask, "mean")


class SumPooling(SequencePooling):
    def forward(
        self,
        sequence: torch.Tensor,
        mask: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._validate_inputs(sequence, mask, target)
        return masked_reduce(sequence, mask, "sum")


class MaxPooling(SequencePooling):
    def forward(
        self,
        sequence: torch.Tensor,
        mask: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._validate_inputs(sequence, mask, target)
        return masked_reduce(sequence, mask, "max")


class AttentionPooling(SequencePooling):
    """Learned target-independent additive attention."""

    def __init__(self, dimension: int) -> None:
        super().__init__(dimension)
        self.projection = torch.nn.Linear(dimension, dimension, bias=True)
        self.bias = torch.nn.Parameter(torch.randn(dimension))
        self.query = torch.nn.Parameter(torch.randn(dimension))

    def forward(
        self,
        sequence: torch.Tensor,
        mask: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._validate_inputs(sequence, mask, target)
        transformed = torch.tanh(self.projection(sequence) + self.bias)
        scores = torch.einsum("d,bld->bl", self.query, transformed)
        weights = masked_softmax(scores, mask)
        return torch.sum(weights.unsqueeze(-1) * sequence, dim=1)


class _Dice(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.zeros(1))
        self.epsilon = 1e-9

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        mean = values.mean(dim=0)
        variance = values.var(dim=0, unbiased=False)
        probability = torch.sigmoid((values - mean) / torch.sqrt(variance + self.epsilon))
        return self.alpha * values * (1 - probability) + values * probability


class DinPooling(SequencePooling):
    """DIN local-activation pooling conditioned on one target vector."""

    capability = PoolingCapability(target_required=True)

    def __init__(
        self,
        dimension: int,
        hidden_dims: Sequence[int] = (64, 32),
        dropout: float = 0.0,
    ) -> None:
        super().__init__(dimension)
        dimensions = tuple(int(value) for value in hidden_dims)
        if not dimensions or any(value <= 0 for value in dimensions):
            raise ContractError("DIN hidden dimensions must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ContractError("DIN dropout must be in [0, 1)")
        layers = []
        input_dim = dimension * 4
        for hidden_dim in dimensions:
            layers.extend(
                [torch.nn.Linear(input_dim, hidden_dim), _Dice(), torch.nn.Dropout(dropout)]
            )
            input_dim = hidden_dim
        layers.append(torch.nn.Linear(input_dim, 1))
        self.network = torch.nn.Sequential(*layers)

    def forward(
        self,
        sequence: torch.Tensor,
        mask: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._validate_inputs(sequence, mask, target)
        if target is None:
            raise ContractError("DIN pooling requires target")
        targets = target.unsqueeze(1).expand(-1, sequence.shape[1], -1)
        inputs = torch.cat([targets, sequence, targets - sequence, targets * sequence], dim=-1)
        scores = self.network(inputs).squeeze(-1)
        weights = masked_softmax(scores, mask)
        return torch.sum(weights.unsqueeze(-1) * sequence, dim=1)


class CrossAttentionPooling(SequencePooling):
    """Multi-head scaled dot-product pooling from target to history."""

    capability = PoolingCapability(target_required=True)

    def __init__(self, dimension: int, heads: int = 1, dropout: float = 0.0) -> None:
        super().__init__(dimension)
        self.heads = int(heads)
        if self.heads <= 0 or dimension % self.heads != 0:
            raise ContractError("cross-attention dimension must divide positive heads")
        if not 0.0 <= float(dropout) < 1.0:
            raise ContractError("cross-attention dropout must be in [0, 1)")
        self.head_dim = dimension // self.heads
        self.scale = self.head_dim**-0.5
        self.query = torch.nn.Linear(dimension, dimension, bias=False)
        self.key = torch.nn.Linear(dimension, dimension, bias=False)
        self.value = torch.nn.Linear(dimension, dimension, bias=False)
        self.dropout = torch.nn.Dropout(dropout)
        self.output = torch.nn.Linear(dimension, dimension)

    def forward(
        self,
        sequence: torch.Tensor,
        mask: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._validate_inputs(sequence, mask, target)
        if target is None:
            raise ContractError("cross-attention pooling requires target")
        batch_size, sequence_length, _ = sequence.shape
        query = self.query(target).reshape(batch_size, self.heads, self.head_dim)
        keys = (
            self.key(sequence)
            .reshape(batch_size, sequence_length, self.heads, self.head_dim)
            .transpose(1, 2)
        )
        values = (
            self.value(sequence)
            .reshape(batch_size, sequence_length, self.heads, self.head_dim)
            .transpose(1, 2)
        )
        scores = torch.einsum("bhd,bhld->bhl", query, keys) * self.scale
        expanded_mask = mask.unsqueeze(1).expand_as(scores)
        weights = self.dropout(masked_softmax(scores, expanded_mask))
        pooled = torch.einsum("bhl,bhld->bhd", weights, values)
        pooled = pooled.reshape(batch_size, self.output_dim)
        return apply_feature_mask(self.output(pooled), mask.any(dim=1))


__all__ = [
    "AttentionPooling",
    "CrossAttentionPooling",
    "DinPooling",
    "MaxPooling",
    "MeanPooling",
    "PoolingCapability",
    "SequencePooling",
    "SumPooling",
    "masked_reduce",
]

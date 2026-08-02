"""Single pure model base for pooled and token-level history architectures."""

from abc import ABC, abstractmethod
from enum import Enum

import torch

from mmctr.core import Batch, ContractError, ModelOutput, ensure_model_output


class HistoryCapability(str, Enum):
    """How a model consumes history after the shared data boundary."""

    POOLED_HISTORY = "pooled_history"
    SEQUENCE_TOKENS = "sequence_tokens"


class BaseSeqModel(torch.nn.Module, ABC):
    """Canonical model base with no training, device, logging, or file responsibilities."""

    def __init__(self, history_capability: HistoryCapability) -> None:
        super().__init__()
        self.history_capability = HistoryCapability(history_capability)

    def forward(self, batch: Batch) -> ModelOutput:
        if not isinstance(batch, Batch):
            raise ContractError("models require a canonical Batch")
        return ensure_model_output(self.forward_batch(batch))

    @abstractmethod
    def forward_batch(self, batch: Batch) -> ModelOutput:
        """Implement a pure forward pass without mutating ``batch``."""

    @staticmethod
    def masked_pool(
        sequence: torch.Tensor,
        mask: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Pool `[B, L, D]` using an explicit boolean `[B, L]` mask."""

        if sequence.ndim != 3:
            raise ContractError("sequence must have shape [B, L, D]")
        if mask.dtype != torch.bool or mask.ndim != 2:
            raise ContractError("mask must be a bool tensor with shape [B, L]")
        if sequence.shape[:2] != mask.shape:
            raise ContractError("sequence and mask dimensions do not match")
        expanded_mask = mask.unsqueeze(-1)
        if reduction == "sum":
            return (sequence * expanded_mask).sum(dim=1)
        if reduction == "mean":
            counts = expanded_mask.sum(dim=1).clamp_min(1)
            return (sequence * expanded_mask).sum(dim=1) / counts
        if reduction == "max":
            minimum = torch.finfo(sequence.dtype).min
            pooled = sequence.masked_fill(~expanded_mask, minimum).max(dim=1).values
            empty_rows = ~mask.any(dim=1)
            return pooled.masked_fill(empty_rows.unsqueeze(-1), 0.0)
        raise ContractError("unsupported history pooling: {!r}".format(reduction))

    @staticmethod
    def masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Softmax over `[B, L]` scores with all-padding rows mapped to zero."""

        if scores.ndim != 2 or mask.ndim != 2 or scores.shape != mask.shape:
            raise ContractError("scores and mask must have matching shape [B, L]")
        if mask.dtype != torch.bool:
            raise ContractError("mask must use torch.bool")
        masked_scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(masked_scores, dim=-1)
        weights = weights * mask.to(dtype=weights.dtype)
        denominator = weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)
        return weights / denominator

    def parameter_counts(self):
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


__all__ = ["BaseSeqModel", "HistoryCapability"]

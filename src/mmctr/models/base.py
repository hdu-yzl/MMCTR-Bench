"""Single pure model base for pooled and token-level history architectures."""

from abc import ABC, abstractmethod
from enum import Enum

import torch

from mmctr.core import Batch, ContractError, ModelOutput, ensure_model_output
from mmctr.models.components.masking import (
    masked_softmax as component_masked_softmax,
    validate_sequence_mask,
)
from mmctr.models.components.pooling import masked_reduce


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
        if not sequence.is_floating_point():
            raise ContractError("sequence must use a floating dtype")
        validate_sequence_mask(mask, sequence)
        return masked_reduce(sequence, mask, reduction)

    @staticmethod
    def masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Softmax over `[B, L]` scores with all-padding rows mapped to zero."""

        if scores.ndim != 2 or mask.ndim != 2:
            raise ContractError("scores and mask must have matching shape [B, L]")
        return component_masked_softmax(scores, mask, dim=-1)

    def parameter_counts(self):
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


__all__ = ["BaseSeqModel", "HistoryCapability"]

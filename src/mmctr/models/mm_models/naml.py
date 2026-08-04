"""Neural attentive multimodal learning recommendation model."""

from typing import Mapping

import torch

from mmctr.core import Batch, ModelOutput
from mmctr.models.common.multimodal import _MAFFusion
from mmctr.models.common.sequence import _SequenceMultimodalModel
from mmctr.models.common.components import AttentionPooling, apply_sequence_mask


class _MaskedUserEncoder(AttentionPooling):
    """State-compatible name retained for the NAML default preset."""


class NAML(_SequenceMultimodalModel):
    """Encode fused history tokens into an interest vector scored against target."""

    def __init__(self, model_config: Mapping, data_config: Mapping) -> None:
        super().__init__(model_config, data_config)
        self.modal_fusion = _MAFFusion(self.feature_names, self.projection_dim)
        self.user_linear = torch.nn.Linear(
            self.projection_dim * len(self.user_feature_names), self.projection_dim
        )
        self.user_encoder = _MaskedUserEncoder(self.projection_dim)

    def forward_batch(self, batch: Batch) -> ModelOutput:
        target = self.modal_fusion(self.project_target(batch)).fused
        history = self.modal_fusion(self.project_history(batch)).fused
        history = apply_sequence_mask(history, batch.history_mask)
        interest = self.user_encoder(history, batch.history_mask)
        user = self.user_linear(self.project_user(batch)) + interest
        logits = torch.einsum("bd,bd->b", target, user)
        return ModelOutput(
            logits,
            representations={"target": target, "user_interest": interest},
        )


__all__ = ["NAML"]

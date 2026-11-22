"""Neural attentive multimodal learning recommendation model."""

from typing import Mapping

import torch

from mmctr.core import Batch, ModelOutput
from mmctr.models.common.sequence import _SequenceMultimodalModel
from mmctr.models.common.components import (
    AttentionPooling,
    ViewAttentionFusion,
    apply_sequence_mask,
    feature_presence,
)


class _MaskedUserEncoder(AttentionPooling):
    """State-compatible name retained for the NAML default preset."""


class NAML(_SequenceMultimodalModel):
    """Benchmark-adapted NAML with view-level and history-level attention."""

    def __init__(self, model_config: Mapping, data_config: Mapping) -> None:
        super().__init__(model_config, data_config)
        # Benchmark adaptation: projected frozen text/image/audio features and an
        # ID view replace the original NAML title/abstract/category encoders.
        # Restore learned view-level attention, shared by targets and history,
        # while retaining the benchmark's CTR inputs and history-level attention.
        # This is an adaptation, not an exact reproduction of the news encoders.
        self.modal_fusion = ViewAttentionFusion(self.feature_names, self.projection_dim)
        self.user_linear = torch.nn.Linear(
            self.projection_dim * len(self.user_feature_names), self.projection_dim
        )
        self.user_encoder = _MaskedUserEncoder(self.projection_dim)

    def forward_batch(self, batch: Batch) -> ModelOutput:
        target_views = self.project_target(batch)
        history_views = self.project_history(batch)
        # Derive presence from raw inputs: a present view can project to zero.
        target_presence = {
            name: (
                torch.ones(batch.batch_size, dtype=torch.bool, device=batch.history_mask.device)
                if name == "id"
                else feature_presence(self._target_feature(batch, name))
            )
            for name in self.feature_names
        }
        history_presence = {
            name: (
                batch.history_mask
                if name == "id"
                else feature_presence(batch.history_features[name]) & batch.history_mask
            )
            for name in self.feature_names
        }
        target_output, target_weights = self.modal_fusion.forward_with_attention(
            target_views, target_presence
        )
        history_output, history_weights = self.modal_fusion.forward_with_attention(
            history_views, history_presence
        )
        target, history = target_output.fused, history_output.fused
        history = apply_sequence_mask(history, batch.history_mask)
        interest = self.user_encoder(history, batch.history_mask)
        user = self.user_linear(self.project_user(batch)) + interest
        logits = torch.einsum("bd,bd->b", target, user)
        return ModelOutput(
            logits,
            representations={
                "target": target,
                "user_interest": interest,
                "target_view_weights": target_weights,
                "history_view_weights": history_weights,
            },
        )


__all__ = ["NAML"]

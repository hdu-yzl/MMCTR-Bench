"""MAKE sequence recommendation model."""

from typing import Mapping

import torch

from mmctr.core import Batch, ModelOutput
from mmctr.models.common.multimodal import _build_fusion
from mmctr.models.common.sequence import _SequenceMultimodalModel
from mmctr.models.common.similarity import SimilarityTiers
from mmctr.models.common.layers import DinAttention, MultiLayerPerceptron
from mmctr.models.common.components import apply_sequence_mask


class MAKE(_SequenceMultimodalModel):
    """Augment target-aware history attention with similarity-tier counts.

    Cosine scores are discretized into a per-row ``[batch, tier_count]``
    histogram; padded history positions contribute no counts.
    """

    def __init__(self, model_config: Mapping, data_config: Mapping) -> None:
        super().__init__(model_config, data_config)
        self.tier_count = int(model_config.get("tier_num", 10))
        self.similarity_tiers = SimilarityTiers(self.tier_count)
        method = model_config.get("modal_fusion_method", "cat")
        rank = int(model_config.get("rank", 5))
        fusion_dim = int(model_config.get("fusion_dim", 16))
        self.modal_fusion = _build_fusion(
            method, self.feature_names, self.projection_dim, rank, fusion_dim
        )
        self.attention = DinAttention(self.modal_fusion.output_dim, self.mlp_dims, self.dropout)
        predictor_dim = (
            self.modal_fusion.output_dim * 2
            + self.projection_dim * len(self.user_feature_names)
            + self.tier_count
        )
        self.dnn = MultiLayerPerceptron(
            predictor_dim,
            self.mlp_dims,
            self.dropout,
            batch_norm=self.batch_norm,
            activation="relu",
        )
        self.out_put = MultiLayerPerceptron(
            self.mlp_dims[-1],
            [1],
            self.dropout,
            batch_norm=self.batch_norm,
            activation=None,
        )

    def forward_batch(self, batch: Batch) -> ModelOutput:
        target = self.modal_fusion(self.project_target(batch)).fused
        history = self.modal_fusion(self.project_history(batch)).fused
        history = apply_sequence_mask(history, batch.history_mask)
        similarities = torch.nn.functional.cosine_similarity(target.unsqueeze(1), history, dim=-1)
        pooled = self.attention(history, batch.history_mask, target)
        tiers = self.similarity_tiers(similarities, batch.history_mask)
        user = self.project_user(batch)
        logits = self.out_put(self.dnn(torch.cat([user, pooled, target, tiers], dim=-1)))
        return ModelOutput(
            logits,
            representations={"similarities": similarities, "similarity_tiers": tiers},
        )


__all__ = ["MAKE"]

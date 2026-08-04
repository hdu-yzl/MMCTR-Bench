"""Decoupled multimodal fusion recommendation model."""

from typing import Mapping

import torch

from mmctr.core import Batch, ContractError, ModelOutput
from mmctr.models.common.multimodal import _build_fusion
from mmctr.models.common.sequence import _SequenceMultimodalModel
from mmctr.models.common.similarity import SimilarityTiers
from mmctr.models.common.base import BaseSeqModel
from mmctr.models.common.layers import FeatureEmbedding, MultiLayerPerceptron
from mmctr.models.common.components import apply_sequence_mask


class _SimilarityDiscretizer(torch.nn.Module):
    def __init__(self, bucket_count: int, minimum: float = -1.0, maximum: float = 1.0) -> None:
        super().__init__()
        if bucket_count <= 0 or minimum >= maximum:
            raise ContractError("similarity discretizer requires a valid bucket range")
        self.bucket_count = int(bucket_count)
        self.minimum = float(minimum)
        self.maximum = float(maximum)

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        clipped = torch.clamp(scores, self.minimum, self.maximum)
        normalised = (clipped - self.minimum) / (self.maximum - self.minimum)
        return (normalised * self.bucket_count).long().clamp(0, self.bucket_count - 1)


class _DecoupledTargetAttention(torch.nn.Module):
    """DMF's ID attention enriched by discretised multimodal similarity."""

    def __init__(
        self,
        dimension: int,
        attention_dim: int,
        bucket_count: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if attention_dim <= 0:
            raise ContractError("DMF attention_dim must be positive")
        self.query = torch.nn.Linear(dimension, attention_dim)
        self.key = torch.nn.Linear(dimension, attention_dim)
        self.value = torch.nn.Linear(dimension, attention_dim)
        self.discretizer = _SimilarityDiscretizer(bucket_count)
        self.similarity_key = FeatureEmbedding(bucket_count, attention_dim)
        self.similarity_value = FeatureEmbedding(bucket_count, attention_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.scale = attention_dim**0.5

    def forward(
        self,
        target_id: torch.Tensor,
        history_id: torch.Tensor,
        similarities: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        query = self.query(target_id)
        buckets = self.discretizer(similarities)
        keys = self.key(history_id) + self.similarity_key(buckets)
        values = self.value(history_id) + self.similarity_value(buckets)
        scores = torch.einsum("bd,bld->bl", query, keys) / self.scale
        weights = self.dropout(BaseSeqModel.masked_softmax(scores, mask))
        return torch.sum(weights.unsqueeze(-1) * values, dim=1)


class DMF(_SequenceMultimodalModel):
    """Decouple ID attention from multimodal similarity-center evidence.

    Non-ID target/history similarity is bucketized to enrich ID attention and
    counted into tiers for a parallel center branch. ``alpha`` interpolates the
    two equal-width interest representations.
    """

    def __init__(self, model_config: Mapping, data_config: Mapping) -> None:
        super().__init__(model_config, data_config)
        self.non_id_features = tuple(name for name in self.feature_names if name != "id")
        if not self.non_id_features:
            raise ContractError("DMF requires at least one non-ID modality")
        self.tier_count = int(model_config.get("tier_num", 10))
        self.attention_dim = int(model_config.get("attention_dim", 128))
        self.bucket_count = int(model_config.get("num_buckets", 35))
        self.alpha = float(model_config.get("alpha", 0.5))
        if not 0.0 <= self.alpha <= 1.0:
            raise ContractError("DMF alpha must be in [0, 1]")
        self.similarity_tiers = SimilarityTiers(self.tier_count)
        self.modal_fusion = _build_fusion(
            model_config.get("modal_fusion_method", "cat"),
            self.non_id_features,
            self.projection_dim,
        )
        self.target_attention = _DecoupledTargetAttention(
            self.projection_dim,
            self.attention_dim,
            self.bucket_count,
            self.dropout,
        )
        self.enhanced_mlp = MultiLayerPerceptron(
            self.attention_dim,
            self.mlp_dims,
            self.dropout,
            batch_norm=self.batch_norm,
        )
        self.center_mlp = MultiLayerPerceptron(
            self.tier_count,
            self.mlp_dims,
            self.dropout,
            batch_norm=self.batch_norm,
        )
        predictor_dim = self.projection_dim * len(self.user_feature_names) + self.mlp_dims[-1]
        self.dnn = MultiLayerPerceptron(
            predictor_dim,
            self.mlp_dims,
            self.dropout,
            batch_norm=self.batch_norm,
        )
        self.output = MultiLayerPerceptron(
            self.mlp_dims[-1],
            [1],
            self.dropout,
            batch_norm=self.batch_norm,
            activation=None,
        )

    def forward_batch(self, batch: Batch) -> ModelOutput:
        target_features = self.project_target(batch)
        history_features = self.project_history(batch)
        target_center = self.modal_fusion(
            {name: target_features[name] for name in self.non_id_features}
        ).fused
        history_center = self.modal_fusion(
            {name: history_features[name] for name in self.non_id_features}
        ).fused
        history_center = apply_sequence_mask(history_center, batch.history_mask)
        similarities = torch.nn.functional.cosine_similarity(
            target_center.unsqueeze(1), history_center, dim=-1
        )
        enhanced = self.target_attention(
            target_features["id"],
            history_features["id"],
            similarities,
            batch.history_mask,
        )
        tiers = self.similarity_tiers(similarities, batch.history_mask)
        enhanced_interest = self.enhanced_mlp(enhanced)
        center_interest = self.center_mlp(tiers)
        interest = self.alpha * enhanced_interest + (1.0 - self.alpha) * center_interest
        user = self.project_user(batch)
        logits = self.output(self.dnn(torch.cat([user, interest], dim=-1)))
        return ModelOutput(
            logits,
            representations={
                "modality_enhanced": enhanced,
                "modality_center": center_interest,
                "similarities": similarities,
                "similarity_tiers": tiers,
            },
        )


__all__ = ["DMF"]

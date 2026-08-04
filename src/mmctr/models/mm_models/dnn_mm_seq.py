"""Sequence-token multimodal DNN recommendation model."""

from typing import Mapping

import torch

from mmctr.core import Batch, ModelOutput
from mmctr.models.common.multimodal import _build_fusion
from mmctr.models.common.sequence import _SequenceMultimodalModel
from mmctr.models.common.layers import MultiLayerPerceptron


class DNN_mm_seq(_SequenceMultimodalModel):
    """Predict CTR after per-modality history pooling and branch-level fusion.

    History tensors remain ``[batch, length, projection_dim]`` until masked
    pooling, while target and user branches are rank-two representations.
    """

    def __init__(self, model_config: Mapping, data_config: Mapping) -> None:
        super().__init__(model_config, data_config)
        method = model_config.get("modal_fusion_method", "add")
        rank = int(model_config.get("rank", 5))
        fusion_dim = int(model_config.get("fusion_dim", 16))
        self.target_fusion = _build_fusion(
            method, self.feature_names, self.projection_dim, rank, fusion_dim
        )
        self.history_fusion = _build_fusion(
            method, self.feature_names, self.projection_dim, rank, fusion_dim
        )
        predictor_dim = (
            self.target_fusion.output_dim
            + self.history_fusion.output_dim
            + self.projection_dim * len(self.user_feature_names)
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
        target = self.target_fusion(self.project_target(batch)).fused
        pooled = {
            name: self.masked_pool(values, batch.history_mask)
            for name, values in self.project_history(batch).items()
        }
        history = self.history_fusion(pooled).fused
        user = self.project_user(batch)
        return ModelOutput(self.out_put(self.dnn(torch.cat([target, history, user], dim=-1))))


__all__ = ["DNN_mm_seq"]

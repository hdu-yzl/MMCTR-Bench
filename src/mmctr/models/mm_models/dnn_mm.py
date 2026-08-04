"""Pooled multimodal DNN recommendation model."""

from typing import Mapping

import torch

from mmctr.core import Batch, ModelOutput
from mmctr.models.common.multimodal import _PooledMultimodalModel, _build_fusion


class DNN_mm(_PooledMultimodalModel):
    """Fuse target and pooled-history modalities before scalar CTR prediction.

    Every configured modality is projected to ``projection_dim`` first; the
    selected fusion determines each branch's final width.
    """

    def __init__(self, model_config: Mapping, data_config: Mapping) -> None:
        super().__init__(model_config, data_config)
        method = model_config.get("modal_fusion_method", "cat")
        rank = int(model_config.get("rank", 5))
        output_dim = int(model_config.get("fusion_dim", 16))
        self.target_fusion = _build_fusion(
            method, self.feature_names, self.projection_dim, rank, output_dim
        )
        self.history_fusion = _build_fusion(
            method, self.history_feature_names, self.projection_dim, rank, output_dim
        )
        input_dim = self.target_fusion.output_dim + self.history_fusion.output_dim
        self.dnn, self.out_put = self.make_predictor(input_dim)

    def forward_batch(self, batch: Batch) -> ModelOutput:
        target = self.target_fusion(self.project_target(batch)).fused
        pooled = {
            name: self.masked_pool(values, batch.history_mask)
            for name, values in self.project_history(batch).items()
        }
        history = self.history_fusion(pooled).fused
        return ModelOutput(self.out_put(self.dnn(torch.cat([target, history], dim=-1))))


__all__ = ["DNN_mm"]

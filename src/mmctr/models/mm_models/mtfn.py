"""Multimodal tensor fusion network recommendation model."""

from typing import Mapping

import torch

from mmctr.core import Batch, ModelOutput
from mmctr.models.common.multimodal import _MTFNFusion, _PooledMultimodalModel


class MTFN(_PooledMultimodalModel):
    """Fuse target and history modalities through factorized tensor interactions."""

    def __init__(self, model_config: Mapping, data_config: Mapping) -> None:
        super().__init__(model_config, data_config)
        rank = int(model_config.get("rank", 20))
        self.target_fusion = _MTFNFusion(self.feature_names, self.projection_dim, rank)
        self.history_fusion = _MTFNFusion(self.history_feature_names, self.projection_dim, rank)
        self.dnn, self.out_put = self.make_predictor(self.projection_dim * 2)

    def forward_batch(self, batch: Batch) -> ModelOutput:
        target = self.target_fusion(self.project_target(batch)).fused
        history_tokens = self.history_fusion(self.project_history(batch)).fused
        history = self.masked_pool(history_tokens, batch.history_mask)
        return ModelOutput(self.out_put(self.dnn(torch.cat([target, history], dim=-1))))


__all__ = ["MTFN"]

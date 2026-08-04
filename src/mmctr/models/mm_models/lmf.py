"""Low-rank multimodal fusion recommendation model."""

from typing import Mapping

import torch

from mmctr.core import Batch, ModelOutput
from mmctr.models.common.multimodal import _LowRankFusion, _PooledMultimodalModel


class LMF(_PooledMultimodalModel):
    """Apply low-rank tensor fusion before pooling history token outputs."""

    def __init__(self, model_config: Mapping, data_config: Mapping) -> None:
        super().__init__(model_config, data_config)
        rank = int(model_config.get("rank", 5))
        output_dim = int(model_config.get("fusion_dim", 16))
        self.target_fusion = _LowRankFusion(
            self.feature_names, self.projection_dim, rank, output_dim
        )
        self.history_fusion = _LowRankFusion(
            self.history_feature_names, self.projection_dim, rank, output_dim
        )
        self.dnn, self.out_put = self.make_predictor(output_dim * 2)

    def forward_batch(self, batch: Batch) -> ModelOutput:
        target = self.target_fusion(self.project_target(batch)).fused
        history_tokens = self.history_fusion(self.project_history(batch)).fused
        history = self.masked_pool(history_tokens, batch.history_mask)
        return ModelOutput(self.out_put(self.dnn(torch.cat([target, history], dim=-1))))


__all__ = ["LMF"]

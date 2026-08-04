"""Deep neural network CTR baseline."""

from typing import Mapping

import torch

from mmctr.core import Batch, ModelOutput
from mmctr.models.common.baseline import _PooledIdBaseline


class DNN(_PooledIdBaseline):
    """Predict CTR from projected user/item IDs and mean-pooled history IDs."""

    def __init__(self, model_config: Mapping, data_config: Mapping) -> None:
        super().__init__(model_config, data_config)
        self.dnn = self.make_mlp(self.projection_dim * 2)
        self.out_put = self.make_output(self.mlp_dims[-1])

    def forward_batch(self, batch: Batch) -> ModelOutput:
        target, history = self.encode_ids(batch)
        logits = self.out_put(self.dnn(torch.cat([target, history], dim=-1)))
        return ModelOutput(logits)


__all__ = ["DNN"]

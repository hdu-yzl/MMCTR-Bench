"""DeepFM CTR baseline."""

from typing import Mapping

import torch

from mmctr.core import Batch, ModelOutput
from mmctr.models.common.baseline import _PooledIdBaseline
from mmctr.models.common.layers import FactorizationMachine


class DeepFM(_PooledIdBaseline):
    """Add second-order target/history interactions to a deep CTR score."""

    def __init__(self, model_config: Mapping, data_config: Mapping) -> None:
        super().__init__(model_config, data_config)
        self.fm = FactorizationMachine()
        self.dnn = self.make_mlp(self.projection_dim * 2)
        self.out_put = self.make_output(self.mlp_dims[-1])

    def forward_batch(self, batch: Batch) -> ModelOutput:
        target, history = self.encode_ids(batch)
        deep = self.out_put(self.dnn(torch.cat([target, history], dim=-1)))
        factorized = self.fm(torch.stack([target, history], dim=1))
        return ModelOutput(deep + factorized)


__all__ = ["DeepFM"]

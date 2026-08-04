"""Deep & Cross Network CTR baseline."""

from typing import Mapping

import torch

from mmctr.core import Batch, ModelOutput
from mmctr.models.common.baseline import _PooledIdBaseline
from mmctr.models.common.layers import CrossNetwork


class DCN(_PooledIdBaseline):
    """Combine explicit bounded-degree crosses with a deep ID representation."""

    def __init__(self, model_config: Mapping, data_config: Mapping) -> None:
        super().__init__(model_config, data_config)
        input_dim = self.projection_dim * 2
        self.cross = CrossNetwork(input_dim, int(model_config.get("cross_num", 3)))
        self.dnn = self.make_mlp(input_dim)
        self.out_put = self.make_output(self.mlp_dims[-1] + input_dim)

    def forward_batch(self, batch: Batch) -> ModelOutput:
        target, history = self.encode_ids(batch)
        values = torch.cat([target, history], dim=-1)
        logits = self.out_put(torch.cat([self.cross(values), self.dnn(values)], dim=-1))
        return ModelOutput(logits)


__all__ = ["DCN"]

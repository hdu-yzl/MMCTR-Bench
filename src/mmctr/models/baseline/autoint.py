"""AutoInt CTR baseline."""

from typing import Mapping

import torch

from mmctr.core import Batch, ModelOutput
from mmctr.models.common.baseline import _PooledIdBaseline
from mmctr.models.common.layers import MultiHeadSelfAttention


class AutoInt(_PooledIdBaseline):
    """Learn explicit target/history field interactions with self-attention."""

    def __init__(self, model_config: Mapping, data_config: Mapping) -> None:
        super().__init__(model_config, data_config)
        layer_count = int(model_config.get("attn_layers", 3))
        heads = int(model_config.get("attn_heads", 2))
        attention_dim = int(model_config.get("attn_size", 64))
        residual = bool(model_config.get("attn_use_residual", True))
        layers = []
        embedding_dim = self.projection_dim
        for _ in range(layer_count):
            layers.append(MultiHeadSelfAttention(embedding_dim, attention_dim, heads, residual))
            embedding_dim = attention_dim * heads
        self.attention = torch.nn.ModuleList(layers)
        self.dnn = self.make_mlp(self.projection_dim * 2)
        self.out_put = self.make_output(self.mlp_dims[-1] + 2 * embedding_dim)

    def forward_batch(self, batch: Batch) -> ModelOutput:
        target, history = self.encode_ids(batch)
        fields = torch.stack([target, history], dim=1)
        attention = fields
        for layer in self.attention:
            attention = layer(attention)
        attention = attention.flatten(start_dim=1)
        deep = self.dnn(torch.cat([target, history], dim=-1))
        return ModelOutput(self.out_put(torch.cat([deep, attention], dim=-1)))


__all__ = ["AutoInt"]

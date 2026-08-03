"""Canonical implementations of the core CTR baseline family."""

from typing import Mapping, Tuple

import torch

from mmctr.core import Batch, ContractError, ModelOutput
from mmctr.models.base import BaseSeqModel, HistoryCapability
from mmctr.models.components import NamedFeatureProjector

from .layers import (
    CrossNetwork,
    DinAttention,
    FactorizationMachine,
    FeatureEmbedding,
    MultiHeadSelfAttention,
    MultiLayerPerceptron,
)


class _PooledIdBaseline(BaseSeqModel):
    def __init__(self, model_config: Mapping, data_config: Mapping) -> None:
        super().__init__(HistoryCapability.POOLED_HISTORY)
        self.latent_dim = int(model_config.get("latent_dim", 128))
        self.projection_dim = int(model_config.get("projection_dim", 128))
        self.mlp_dims = tuple(
            int(value) for value in model_config.get("mlp_dims", [1024, 512, 256])
        )
        self.dropout = float(model_config.get("dropout", 0.5))
        self.batch_norm = bool(model_config.get("batch_norm", False))
        self.mm_projector = NamedFeatureProjector({"id": self.latent_dim * 2}, self.projection_dim)
        self.mm_seq_projector = NamedFeatureProjector({"id": self.latent_dim}, self.projection_dim)
        self.embedding = FeatureEmbedding(int(data_config["id_feature_num"]) + 1, self.latent_dim)

    def encode_ids(self, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            user_ids = batch.user_features["id"]
            item_ids = batch.item_features["id"]
            history_ids = batch.history_features["id"]
        except KeyError as error:
            raise ContractError("ID baselines require user/item/history 'id' features") from error
        target_ids = torch.cat([user_ids, item_ids], dim=1)
        target_embedding = self.embedding(target_ids).flatten(start_dim=1)
        history_embedding = self.embedding(history_ids)
        target = self.mm_projector["id"](target_embedding)
        history = self.mm_seq_projector["id"](history_embedding)
        pooled_history = self.masked_pool(history, batch.history_mask, "mean")
        return target, pooled_history

    def make_mlp(self, input_dim: int) -> MultiLayerPerceptron:
        return MultiLayerPerceptron(
            input_dim,
            self.mlp_dims,
            self.dropout,
            batch_norm=self.batch_norm,
            activation="relu",
        )

    def make_output(self, input_dim: int) -> MultiLayerPerceptron:
        return MultiLayerPerceptron(
            input_dim, [1], self.dropout, batch_norm=self.batch_norm, activation=None
        )


class DNN(_PooledIdBaseline):
    def __init__(self, model_config: Mapping, data_config: Mapping) -> None:
        super().__init__(model_config, data_config)
        self.dnn = self.make_mlp(self.projection_dim * 2)
        self.out_put = self.make_output(self.mlp_dims[-1])

    def forward_batch(self, batch: Batch) -> ModelOutput:
        target, history = self.encode_ids(batch)
        logits = self.out_put(self.dnn(torch.cat([target, history], dim=-1)))
        return ModelOutput(logits)


class DCN(_PooledIdBaseline):
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


class DeepFM(_PooledIdBaseline):
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


class AutoInt(_PooledIdBaseline):
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


class DIN(BaseSeqModel):
    def __init__(self, model_config: Mapping, data_config: Mapping) -> None:
        super().__init__(HistoryCapability.SEQUENCE_TOKENS)
        latent_dim = int(model_config.get("latent_dim", 128))
        projection_dim = int(model_config.get("projection_dim", 128))
        dropout = float(model_config.get("dropout", 0.5))
        batch_norm = bool(model_config.get("batch_norm", False))
        mlp_dims = tuple(int(value) for value in model_config.get("mlp_dims", [1024, 512, 256]))
        self.mm_projector = NamedFeatureProjector({"id": latent_dim}, projection_dim)
        self.user_projector = NamedFeatureProjector({"id": latent_dim}, projection_dim)
        self.embedding = FeatureEmbedding(int(data_config["id_feature_num"]) + 1, latent_dim)
        self.attention_pooling = DinAttention(
            projection_dim,
            model_config.get("attention_mlp_dims", [64, 32]),
            dropout,
        )
        self.dnn = MultiLayerPerceptron(
            projection_dim * 3, mlp_dims, dropout, batch_norm=batch_norm
        )
        self.out_put = MultiLayerPerceptron(
            mlp_dims[-1], [1], dropout, batch_norm=batch_norm, activation=None
        )

    def forward_batch(self, batch: Batch) -> ModelOutput:
        try:
            user_ids = batch.user_features["id"]
            item_ids = batch.item_features["id"]
            history_ids = batch.history_features["id"]
        except KeyError as error:
            raise ContractError("DIN requires user/item/history 'id' features") from error
        user = self.user_projector["id"](self.embedding(user_ids).squeeze(1))
        item = self.mm_projector["id"](self.embedding(item_ids).squeeze(1))
        history = self.mm_projector["id"](self.embedding(history_ids))
        interest = self.attention_pooling(history, batch.history_mask, item)
        logits = self.out_put(self.dnn(torch.cat([user, item, interest], dim=-1)))
        return ModelOutput(logits)


__all__ = ["AutoInt", "DCN", "DIN", "DNN", "DeepFM"]

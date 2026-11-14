"""Shared implementation for pooled ID-only baseline models."""

from typing import Mapping, Tuple

import torch

from mmctr.core import Batch, ContractError
from mmctr.models.common.base import BaseSeqModel, HistoryCapability
from mmctr.models.common.components import NamedFeatureProjector

from .layers import FeatureEmbedding, MultiLayerPerceptron


class _PooledIdBaseline(BaseSeqModel):
    def __init__(self, model_config: Mapping, data_config: Mapping) -> None:
        super().__init__(HistoryCapability.POOLED_HISTORY)
        self.extra_id_fields = int(data_config.get("extra_id_fields", 0))
        self.id_field_widths = tuple(data_config.get("id_field_widths", ()))
        if len(self.id_field_widths) != self.extra_id_fields:
            raise ContractError("additional ID widths/count mismatch")
        self.latent_dim = int(model_config.get("latent_dim", 128))
        self.projection_dim = int(model_config.get("projection_dim", 128))
        self.mlp_dims = tuple(
            int(value) for value in model_config.get("mlp_dims", [1024, 512, 256])
        )
        self.dropout = float(model_config.get("dropout", 0.5))
        self.batch_norm = bool(model_config.get("batch_norm", False))
        self.mm_projector = NamedFeatureProjector(
            {"id": self.latent_dim * (2 + self.extra_id_fields)}, self.projection_dim
        )
        self.mm_seq_projector = NamedFeatureProjector({"id": self.latent_dim}, self.projection_dim)
        self.embedding = FeatureEmbedding(int(data_config["id_feature_num"]) + 1, self.latent_dim)

    def encode_ids(self, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            item_ids = batch.item_features["id"]
            history_ids = batch.history_features["id"]
        except KeyError as error:
            raise ContractError("ID baselines require user/item/history 'id' features") from error
        target_embedding = torch.cat(
            [self.embed_user_fields(batch), self.embedding(item_ids).flatten(start_dim=1)], dim=1
        )
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


__all__ = ["_PooledIdBaseline"]

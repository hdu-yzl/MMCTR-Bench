"""Deep Interest Network CTR baseline."""

from typing import Mapping

import torch

from mmctr.core import Batch, ContractError, ModelOutput
from mmctr.models.common.base import BaseSeqModel, HistoryCapability
from mmctr.models.common.layers import DinAttention, FeatureEmbedding, MultiLayerPerceptron
from mmctr.models.common.components import NamedFeatureProjector


class DIN(BaseSeqModel):
    """Pool ID history with target-item-conditioned attention before prediction.

    History IDs are encoded as ``[batch, length, projection_dim]`` and padded
    positions are excluded by ``history_mask``.
    """

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


__all__ = ["DIN"]

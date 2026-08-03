"""Canonical sequence-token multimodal models and their migration backbone."""

from typing import Dict, Mapping

import torch

from mmctr.core import Batch, ContractError, ModelOutput
from mmctr.models.base import BaseSeqModel, HistoryCapability
from mmctr.models.baselines.layers import DinAttention, FeatureEmbedding, MultiLayerPerceptron
from mmctr.models.multimodal import _MAFFusion, _build_fusion


class _SequenceMultimodalModel(BaseSeqModel):
    """Shared pure encoder for user, target-item, and history-token branches."""

    def __init__(self, model_config: Mapping, data_config: Mapping) -> None:
        super().__init__(HistoryCapability.SEQUENCE_TOKENS)
        self.latent_dim = int(model_config.get("latent_dim", 128))
        self.projection_dim = int(model_config.get("projection_dim", 128))
        self.dropout = float(model_config.get("dropout", 0.5))
        self.batch_norm = bool(model_config.get("batch_norm", False))
        self.mlp_dims = tuple(
            int(value) for value in model_config.get("mlp_dims", [1024, 512, 256])
        )
        self.feature_names = tuple(data_config.get("use_mm_features", ("id",)))
        self.user_feature_names = tuple(data_config.get("user_features", ("id",)))
        if self.latent_dim <= 0 or self.projection_dim <= 0:
            raise ContractError("sequence latent/projection dimensions must be positive")
        if not self.mlp_dims or any(dimension <= 0 for dimension in self.mlp_dims):
            raise ContractError("sequence model mlp_dims must contain positive dimensions")
        if not 0.0 <= self.dropout < 1.0:
            raise ContractError("sequence model dropout must be in [0, 1)")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ContractError("sequence item feature names must be unique")
        if len(set(self.user_feature_names)) != len(self.user_feature_names):
            raise ContractError("sequence user feature names must be unique")
        if "id" not in self.feature_names or "id" not in self.user_feature_names:
            raise ContractError("sequence multimodal models require item/history/user IDs")

        dimensions = dict(data_config.get("mm_seq_dims", data_config.get("mm_dims", {})))
        user_dimensions = dict(data_config.get("user_features_dim", {}))
        dimensions["id"] = self.latent_dim
        user_dimensions["id"] = self.latent_dim
        missing = [name for name in self.feature_names if name not in dimensions]
        missing.extend(
            name for name in self.user_feature_names if name not in user_dimensions
        )
        if missing:
            raise ContractError("missing sequence multimodal dimensions: {}".format(missing))
        selected_dimensions = [
            ("item.{}".format(name), dimensions[name]) for name in self.feature_names
        ]
        selected_dimensions.extend(
            ("user.{}".format(name), user_dimensions[name])
            for name in self.user_feature_names
        )
        invalid = {
            name: int(dimension)
            for name, dimension in selected_dimensions
            if int(dimension) <= 0
        }
        if invalid:
            raise ContractError("sequence feature dimensions must be positive: {}".format(invalid))

        self.embedding = FeatureEmbedding(
            int(data_config["id_feature_num"]) + 1, self.latent_dim
        )
        self.projectors = torch.nn.ModuleDict(
            {
                name: torch.nn.Linear(int(dimensions[name]), self.projection_dim)
                for name in self.feature_names
            }
        )
        self.user_projectors = torch.nn.ModuleDict(
            {
                name: torch.nn.Linear(int(user_dimensions[name]), self.projection_dim)
                for name in self.user_feature_names
            }
        )

    @staticmethod
    def _target_feature(batch: Batch, name: str) -> torch.Tensor:
        if name in batch.item_features:
            return batch.item_features[name]
        if name in batch.context_features:
            return batch.context_features[name]
        raise ContractError("item/context feature {!r} is missing".format(name))

    def project_target(self, batch: Batch) -> Dict[str, torch.Tensor]:
        projected: Dict[str, torch.Tensor] = {}
        for name in self.feature_names:
            values = self._target_feature(batch, name)
            if name == "id":
                values = self.embedding(values).squeeze(1)
            projected[name] = self.projectors[name](values)
        return projected

    def project_history(self, batch: Batch) -> Dict[str, torch.Tensor]:
        projected: Dict[str, torch.Tensor] = {}
        mask = batch.history_mask.unsqueeze(-1)
        for name in self.feature_names:
            try:
                values = batch.history_features[name]
            except KeyError as error:
                raise ContractError("history feature {!r} is missing".format(name)) from error
            if name == "id":
                values = self.embedding(values)
            projected[name] = self.projectors[name](values) * mask
        return projected

    def project_user(self, batch: Batch) -> torch.Tensor:
        projected = []
        for name in self.user_feature_names:
            try:
                values = batch.user_features[name]
            except KeyError as error:
                raise ContractError("user feature {!r} is missing".format(name)) from error
            if name == "id":
                values = self.embedding(values).squeeze(1)
            projected.append(self.user_projectors[name](values))
        return torch.cat(projected, dim=-1)


class DNN_mm_seq(_SequenceMultimodalModel):
    def __init__(self, model_config: Mapping, data_config: Mapping) -> None:
        super().__init__(model_config, data_config)
        method = model_config.get("modal_fusion_method", "add")
        rank = int(model_config.get("rank", 5))
        fusion_dim = int(model_config.get("fusion_dim", 16))
        self.target_fusion = _build_fusion(
            method, self.feature_names, self.projection_dim, rank, fusion_dim
        )
        self.history_fusion = _build_fusion(
            method, self.feature_names, self.projection_dim, rank, fusion_dim
        )
        predictor_dim = (
            self.target_fusion.output_dim
            + self.history_fusion.output_dim
            + self.projection_dim * len(self.user_feature_names)
        )
        self.dnn = MultiLayerPerceptron(
            predictor_dim,
            self.mlp_dims,
            self.dropout,
            batch_norm=self.batch_norm,
            activation="relu",
        )
        self.out_put = MultiLayerPerceptron(
            self.mlp_dims[-1],
            [1],
            self.dropout,
            batch_norm=self.batch_norm,
            activation=None,
        )

    def forward_batch(self, batch: Batch) -> ModelOutput:
        target = self.target_fusion(self.project_target(batch))
        pooled = {
            name: self.masked_pool(values, batch.history_mask)
            for name, values in self.project_history(batch).items()
        }
        history = self.history_fusion(pooled)
        user = self.project_user(batch)
        return ModelOutput(
            self.out_put(self.dnn(torch.cat([target, history, user], dim=-1)))
        )


class _MaskedUserEncoder(torch.nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(dimension, dimension, bias=True)
        self.bias = torch.nn.Parameter(torch.randn(dimension))
        self.query = torch.nn.Parameter(torch.randn(dimension))

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        transformed = torch.tanh(self.projection(sequence) + self.bias)
        scores = torch.einsum("d,bld->bl", self.query, transformed)
        weights = BaseSeqModel.masked_softmax(scores, mask)
        return torch.sum(weights.unsqueeze(-1) * sequence, dim=1)


class NAML(_SequenceMultimodalModel):
    def __init__(self, model_config: Mapping, data_config: Mapping) -> None:
        super().__init__(model_config, data_config)
        self.modal_fusion = _MAFFusion(self.feature_names, self.projection_dim)
        self.user_linear = torch.nn.Linear(
            self.projection_dim * len(self.user_feature_names), self.projection_dim
        )
        self.user_encoder = _MaskedUserEncoder(self.projection_dim)

    def forward_batch(self, batch: Batch) -> ModelOutput:
        target = self.modal_fusion(self.project_target(batch))
        history = self.modal_fusion(self.project_history(batch))
        history = history * batch.history_mask.unsqueeze(-1)
        interest = self.user_encoder(history, batch.history_mask)
        user = self.user_linear(self.project_user(batch)) + interest
        logits = torch.einsum("bd,bd->b", target, user)
        return ModelOutput(
            logits,
            representations={"target": target, "user_interest": interest},
        )


class _SimilarityTiers(torch.nn.Module):
    def __init__(
        self, tier_count: int, minimum: float = -1.0, maximum: float = 1.0
    ) -> None:
        super().__init__()
        if tier_count <= 0 or minimum >= maximum:
            raise ContractError("similarity tiers require a positive count and valid range")
        self.tier_count = int(tier_count)
        self.register_buffer(
            "boundaries", torch.linspace(minimum, maximum, steps=self.tier_count + 1)
        )

    def forward(self, scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        indices = torch.bucketize(scores, self.boundaries, right=False)
        indices = torch.clamp(indices - 1, 0, self.tier_count - 1)
        counts = scores.new_zeros((scores.shape[0], self.tier_count))
        counts.scatter_add_(1, indices, mask.to(dtype=scores.dtype))
        return counts


class MAKE(_SequenceMultimodalModel):
    def __init__(self, model_config: Mapping, data_config: Mapping) -> None:
        super().__init__(model_config, data_config)
        self.tier_count = int(model_config.get("tier_num", 10))
        self.similarity_tiers = _SimilarityTiers(self.tier_count)
        method = model_config.get("modal_fusion_method", "cat")
        rank = int(model_config.get("rank", 5))
        fusion_dim = int(model_config.get("fusion_dim", 16))
        self.modal_fusion = _build_fusion(
            method, self.feature_names, self.projection_dim, rank, fusion_dim
        )
        self.attention = DinAttention(
            self.modal_fusion.output_dim, self.mlp_dims, self.dropout
        )
        predictor_dim = (
            self.modal_fusion.output_dim * 2
            + self.projection_dim * len(self.user_feature_names)
            + self.tier_count
        )
        self.dnn = MultiLayerPerceptron(
            predictor_dim,
            self.mlp_dims,
            self.dropout,
            batch_norm=self.batch_norm,
            activation="relu",
        )
        self.out_put = MultiLayerPerceptron(
            self.mlp_dims[-1],
            [1],
            self.dropout,
            batch_norm=self.batch_norm,
            activation=None,
        )

    def forward_batch(self, batch: Batch) -> ModelOutput:
        target = self.modal_fusion(self.project_target(batch))
        history = self.modal_fusion(self.project_history(batch))
        history = history * batch.history_mask.unsqueeze(-1)
        similarities = torch.nn.functional.cosine_similarity(
            target.unsqueeze(1), history, dim=-1
        )
        pooled = self.attention(target, history, batch.history_mask)
        tiers = self.similarity_tiers(similarities, batch.history_mask)
        user = self.project_user(batch)
        logits = self.out_put(
            self.dnn(torch.cat([user, pooled, target, tiers], dim=-1))
        )
        return ModelOutput(
            logits,
            representations={"similarities": similarities, "similarity_tiers": tiers},
        )


__all__ = ["DNN_mm_seq", "MAKE", "NAML"]

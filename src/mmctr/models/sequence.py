"""Canonical sequence-token multimodal models and their migration backbone."""

from typing import Dict, Mapping, Tuple

import torch

from mmctr.core import Batch, ContractError, ModelOutput
from mmctr.models.base import BaseSeqModel, HistoryCapability
from mmctr.models.baselines.layers import DinAttention, FeatureEmbedding, MultiLayerPerceptron
from mmctr.models.components import (
    AttentionPooling,
    NamedFeatureProjector,
    apply_sequence_mask,
    feature_presence,
)
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
        self.projectors = NamedFeatureProjector(
            {name: int(dimensions[name]) for name in self.feature_names},
            self.projection_dim,
        )
        self.user_projectors = NamedFeatureProjector(
            {name: int(user_dimensions[name]) for name in self.user_feature_names},
            self.projection_dim,
        )

    @staticmethod
    def _target_feature(batch: Batch, name: str) -> torch.Tensor:
        if name in batch.item_features:
            return batch.item_features[name]
        if name in batch.context_features:
            return batch.context_features[name]
        raise ContractError("item/context feature {!r} is missing".format(name))

    def project_target(self, batch: Batch) -> Dict[str, torch.Tensor]:
        encoded: Dict[str, torch.Tensor] = {}
        presence: Dict[str, torch.Tensor] = {}
        for name in self.feature_names:
            values = self._target_feature(batch, name)
            if name == "id":
                values = self.embedding(values).squeeze(1)
            else:
                presence[name] = feature_presence(values)
            encoded[name] = values
        return self.projectors(encoded, presence)

    def project_history(self, batch: Batch) -> Dict[str, torch.Tensor]:
        encoded: Dict[str, torch.Tensor] = {}
        presence: Dict[str, torch.Tensor] = {}
        for name in self.feature_names:
            try:
                values = batch.history_features[name]
            except KeyError as error:
                raise ContractError("history feature {!r} is missing".format(name)) from error
            if name == "id":
                values = self.embedding(values)
                presence[name] = batch.history_mask
            else:
                presence[name] = feature_presence(values) & batch.history_mask
            encoded[name] = values
        return self.projectors(encoded, presence)

    def project_user(self, batch: Batch) -> torch.Tensor:
        encoded: Dict[str, torch.Tensor] = {}
        presence: Dict[str, torch.Tensor] = {}
        for name in self.user_feature_names:
            try:
                values = batch.user_features[name]
            except KeyError as error:
                raise ContractError("user feature {!r} is missing".format(name)) from error
            if name == "id":
                values = self.embedding(values).squeeze(1)
            else:
                presence[name] = feature_presence(values)
            encoded[name] = values
        projected = self.user_projectors(encoded, presence)
        return torch.cat(
            [projected[name] for name in self.user_feature_names], dim=-1
        )


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


class _MaskedUserEncoder(AttentionPooling):
    """State-compatible name retained for the NAML default preset."""


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
        history = apply_sequence_mask(history, batch.history_mask)
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
        history = apply_sequence_mask(history, batch.history_mask)
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


class _SimilarityDiscretizer(torch.nn.Module):
    def __init__(
        self, bucket_count: int, minimum: float = -1.0, maximum: float = 1.0
    ) -> None:
        super().__init__()
        if bucket_count <= 0 or minimum >= maximum:
            raise ContractError("similarity discretizer requires a valid bucket range")
        self.bucket_count = int(bucket_count)
        self.minimum = float(minimum)
        self.maximum = float(maximum)

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        clipped = torch.clamp(scores, self.minimum, self.maximum)
        normalised = (clipped - self.minimum) / (self.maximum - self.minimum)
        return (normalised * self.bucket_count).long().clamp(0, self.bucket_count - 1)


class _DecoupledTargetAttention(torch.nn.Module):
    """DMF's ID attention enriched by discretised multimodal similarity."""

    def __init__(
        self,
        dimension: int,
        attention_dim: int,
        bucket_count: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if attention_dim <= 0:
            raise ContractError("DMF attention_dim must be positive")
        self.query = torch.nn.Linear(dimension, attention_dim)
        self.key = torch.nn.Linear(dimension, attention_dim)
        self.value = torch.nn.Linear(dimension, attention_dim)
        self.discretizer = _SimilarityDiscretizer(bucket_count)
        self.similarity_key = FeatureEmbedding(bucket_count, attention_dim)
        self.similarity_value = FeatureEmbedding(bucket_count, attention_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.scale = attention_dim ** 0.5

    def forward(
        self,
        target_id: torch.Tensor,
        history_id: torch.Tensor,
        similarities: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        query = self.query(target_id)
        buckets = self.discretizer(similarities)
        keys = self.key(history_id) + self.similarity_key(buckets)
        values = self.value(history_id) + self.similarity_value(buckets)
        scores = torch.einsum("bd,bld->bl", query, keys) / self.scale
        weights = self.dropout(BaseSeqModel.masked_softmax(scores, mask))
        return torch.sum(weights.unsqueeze(-1) * values, dim=1)


class DMF(_SequenceMultimodalModel):
    def __init__(self, model_config: Mapping, data_config: Mapping) -> None:
        super().__init__(model_config, data_config)
        self.non_id_features = tuple(
            name for name in self.feature_names if name != "id"
        )
        if not self.non_id_features:
            raise ContractError("DMF requires at least one non-ID modality")
        self.tier_count = int(model_config.get("tier_num", 10))
        self.attention_dim = int(model_config.get("attention_dim", 128))
        self.bucket_count = int(model_config.get("num_buckets", 35))
        self.alpha = float(model_config.get("alpha", 0.5))
        if not 0.0 <= self.alpha <= 1.0:
            raise ContractError("DMF alpha must be in [0, 1]")
        self.similarity_tiers = _SimilarityTiers(self.tier_count)
        self.modal_fusion = _build_fusion(
            model_config.get("modal_fusion_method", "cat"),
            self.non_id_features,
            self.projection_dim,
        )
        self.target_attention = _DecoupledTargetAttention(
            self.projection_dim,
            self.attention_dim,
            self.bucket_count,
            self.dropout,
        )
        self.enhanced_mlp = MultiLayerPerceptron(
            self.attention_dim,
            self.mlp_dims,
            self.dropout,
            batch_norm=self.batch_norm,
        )
        self.center_mlp = MultiLayerPerceptron(
            self.tier_count,
            self.mlp_dims,
            self.dropout,
            batch_norm=self.batch_norm,
        )
        predictor_dim = (
            self.projection_dim * len(self.user_feature_names) + self.mlp_dims[-1]
        )
        self.dnn = MultiLayerPerceptron(
            predictor_dim,
            self.mlp_dims,
            self.dropout,
            batch_norm=self.batch_norm,
        )
        self.output = MultiLayerPerceptron(
            self.mlp_dims[-1],
            [1],
            self.dropout,
            batch_norm=self.batch_norm,
            activation=None,
        )

    def forward_batch(self, batch: Batch) -> ModelOutput:
        target_features = self.project_target(batch)
        history_features = self.project_history(batch)
        target_center = self.modal_fusion(
            {name: target_features[name] for name in self.non_id_features}
        )
        history_center = self.modal_fusion(
            {name: history_features[name] for name in self.non_id_features}
        )
        history_center = apply_sequence_mask(history_center, batch.history_mask)
        similarities = torch.nn.functional.cosine_similarity(
            target_center.unsqueeze(1), history_center, dim=-1
        )
        enhanced = self.target_attention(
            target_features["id"],
            history_features["id"],
            similarities,
            batch.history_mask,
        )
        tiers = self.similarity_tiers(similarities, batch.history_mask)
        enhanced_interest = self.enhanced_mlp(enhanced)
        center_interest = self.center_mlp(tiers)
        interest = self.alpha * enhanced_interest + (1.0 - self.alpha) * center_interest
        user = self.project_user(batch)
        logits = self.output(self.dnn(torch.cat([user, interest], dim=-1)))
        return ModelOutput(
            logits,
            representations={
                "modality_enhanced": enhanced,
                "modality_center": center_interest,
                "similarities": similarities,
                "similarity_tiers": tiers,
            },
        )


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(context, values: torch.Tensor, weight: float) -> torch.Tensor:
        context.weight = float(weight)
        return values.view_as(values)

    @staticmethod
    def backward(context, gradient: torch.Tensor):
        return -context.weight * gradient, None


def _reverse_gradient(values: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    return _GradientReversal.apply(values, weight)


class _ModalitySplit(torch.nn.Module):
    def __init__(self, dimension: int, feature_names: Tuple[str, ...]) -> None:
        super().__init__()
        self.feature_names = feature_names
        self.specific = torch.nn.ModuleDict(
            {name: torch.nn.Linear(dimension, dimension) for name in feature_names}
        )
        self.invariant = torch.nn.Linear(dimension, dimension)

    def forward(
        self, features: Mapping[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        specific = {
            name: self.specific[name](features[name]) for name in self.feature_names
        }
        invariant = {
            name: self.invariant(features[name]) for name in self.feature_names
        }
        return specific, invariant


class _AdversarialModalityEncoder(torch.nn.Module):
    def __init__(self, dimension: int, feature_names: Tuple[str, ...]) -> None:
        super().__init__()
        self.feature_names = feature_names
        self.feature_count = len(feature_names)
        self.domain_classifier = self._classifier(dimension)
        self.adversarial_classifier = self._classifier(dimension)

    def _classifier(self, dimension: int) -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(dimension, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, self.feature_count),
        )

    def _weighted(
        self, features: Mapping[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        stacked = torch.stack(
            [features[name] for name in self.feature_names], dim=1
        )
        batch_size, feature_count, dimension = stacked.shape
        flattened = stacked.reshape(batch_size * feature_count, dimension)
        labels = torch.arange(feature_count, device=stacked.device).repeat(batch_size)
        logits = self.domain_classifier(flattened.detach())
        uncertainty = 1.0 - torch.softmax(logits, dim=-1)
        rows = torch.arange(batch_size * feature_count, device=stacked.device)
        weights = uncertainty[rows, labels].reshape(batch_size, feature_count, 1)
        return stacked, labels, logits, weights * stacked

    def representation(self, features: Mapping[str, torch.Tensor]) -> torch.Tensor:
        _, _, _, weighted = self._weighted(features)
        return weighted.max(dim=1).values

    def forward(
        self, features: Mapping[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stacked, labels, logits, weighted = self._weighted(features)
        dimension = stacked.shape[-1]
        domain_loss = torch.nn.functional.cross_entropy(logits, labels)
        reversed_values = _reverse_gradient(weighted.reshape(-1, dimension))
        adversarial_logits = self.adversarial_classifier(reversed_values)
        adversarial_loss = torch.nn.functional.cross_entropy(
            adversarial_logits, labels
        )
        return domain_loss, adversarial_loss, weighted.max(dim=1).values


class _ModalitySpecificClassifier(torch.nn.Module):
    def __init__(self, dimension: int, feature_names: Tuple[str, ...]) -> None:
        super().__init__()
        self.feature_names = feature_names
        self.feature_count = len(feature_names)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(dimension, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, self.feature_count),
        )

    def forward(self, features: Mapping[str, torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(
            [features[name].detach() for name in self.feature_names], dim=1
        )
        logits = self.classifier(stacked)
        labels = torch.arange(self.feature_count, device=stacked.device).repeat(
            stacked.shape[0]
        )
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, self.feature_count), labels
        )


class MARN(_SequenceMultimodalModel):
    def __init__(self, model_config: Mapping, data_config: Mapping) -> None:
        super().__init__(model_config, data_config)
        self.auxiliary_weight = float(model_config.get("lambda0", 0.05))
        if self.auxiliary_weight < 0.0:
            raise ContractError("MARN lambda0 must be non-negative")
        self.split = _ModalitySplit(self.projection_dim, self.feature_names)
        self.private_fusion = _MAFFusion(self.feature_names, self.projection_dim)
        self.adversarial = _AdversarialModalityEncoder(
            self.projection_dim, self.feature_names
        )
        self.specific_classifier = _ModalitySpecificClassifier(
            self.projection_dim, self.feature_names
        )
        self.attention = DinAttention(
            self.projection_dim, self.mlp_dims, self.dropout
        )
        predictor_dim = self.projection_dim * (
            2 + len(self.user_feature_names)
        )
        self.dnn = MultiLayerPerceptron(
            predictor_dim,
            self.mlp_dims,
            self.dropout,
            batch_norm=self.batch_norm,
        )
        self.output = MultiLayerPerceptron(
            self.mlp_dims[-1],
            [1],
            self.dropout,
            batch_norm=self.batch_norm,
            activation=None,
        )

    @staticmethod
    def _apply_mask(
        features: Mapping[str, torch.Tensor], mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return {
            name: apply_sequence_mask(values, mask)
            for name, values in features.items()
        }

    def _history_invariant(
        self, features: Mapping[str, torch.Tensor], batch: Batch
    ) -> torch.Tensor:
        flattened = {
            name: values.reshape(-1, self.projection_dim)
            for name, values in features.items()
        }
        invariant = self.adversarial.representation(flattened)
        invariant = invariant.reshape(
            batch.batch_size, batch.sequence_length, self.projection_dim
        )
        return apply_sequence_mask(invariant, batch.history_mask)

    def forward_batch(self, batch: Batch) -> ModelOutput:
        target = self.project_target(batch)
        history = self.project_history(batch)
        target_specific, target_invariant = self.split(target)
        history_specific, history_invariant = self.split(history)
        history_specific = self._apply_mask(history_specific, batch.history_mask)
        history_invariant = self._apply_mask(history_invariant, batch.history_mask)

        target_private = self.private_fusion(target_specific)
        history_private = self.private_fusion(history_specific)
        history_private = apply_sequence_mask(history_private, batch.history_mask)
        domain_loss, adversarial_loss, invariant_target = self.adversarial(
            target_invariant
        )
        specific_loss = self.specific_classifier(target_specific)
        invariant_history = self._history_invariant(history_invariant, batch)

        target_representation = invariant_target + target_private
        history_representation = invariant_history + history_private
        history_representation = apply_sequence_mask(
            history_representation, batch.history_mask
        )
        history_interest = self.attention(
            target_representation,
            history_representation,
            batch.history_mask,
        )
        user = self.project_user(batch)
        logits = self.output(
            self.dnn(
                torch.cat(
                    [user, history_interest, target_representation], dim=-1
                )
            )
        )
        return ModelOutput(
            logits,
            auxiliary_losses={
                "marn_domain_classifier": domain_loss,
                "marn_adversarial_invariance": (
                    adversarial_loss * self.auxiliary_weight
                ),
                "marn_specific_classifier": specific_loss * self.auxiliary_weight,
            },
            representations={
                "target": target_representation,
                "history_interest": history_interest,
            },
        )


__all__ = ["DMF", "DNN_mm_seq", "MAKE", "MARN", "NAML"]

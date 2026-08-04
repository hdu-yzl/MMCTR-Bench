"""Modality-adversarial recommendation network."""

from typing import Dict, Mapping, Tuple

import torch

from mmctr.core import Batch, ContractError, ModelOutput
from mmctr.models.common.multimodal import _MAFFusion
from mmctr.models.common.sequence import _SequenceMultimodalModel
from mmctr.models.common.layers import DinAttention, MultiLayerPerceptron
from mmctr.models.common.components import apply_sequence_mask


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
        specific = {name: self.specific[name](features[name]) for name in self.feature_names}
        invariant = {name: self.invariant(features[name]) for name in self.feature_names}
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
        stacked = torch.stack([features[name] for name in self.feature_names], dim=1)
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
        adversarial_loss = torch.nn.functional.cross_entropy(adversarial_logits, labels)
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
        stacked = torch.stack([features[name].detach() for name in self.feature_names], dim=1)
        logits = self.classifier(stacked)
        labels = torch.arange(self.feature_count, device=stacked.device).repeat(stacked.shape[0])
        return torch.nn.functional.cross_entropy(logits.reshape(-1, self.feature_count), labels)


class MARN(_SequenceMultimodalModel):
    """Split modalities into private and adversarially invariant interests.

    Gradient reversal trains invariant features to confuse modality identity,
    while the detached domain classifier supplies uncertainty weights. Named
    auxiliary losses remain scalar and are already weighted where applicable.
    """

    def __init__(self, model_config: Mapping, data_config: Mapping) -> None:
        super().__init__(model_config, data_config)
        self.auxiliary_weight = float(model_config.get("lambda0", 0.05))
        if self.auxiliary_weight < 0.0:
            raise ContractError("MARN lambda0 must be non-negative")
        self.split = _ModalitySplit(self.projection_dim, self.feature_names)
        self.private_fusion = _MAFFusion(self.feature_names, self.projection_dim)
        self.adversarial = _AdversarialModalityEncoder(self.projection_dim, self.feature_names)
        self.specific_classifier = _ModalitySpecificClassifier(
            self.projection_dim, self.feature_names
        )
        self.attention = DinAttention(self.projection_dim, self.mlp_dims, self.dropout)
        predictor_dim = self.projection_dim * (2 + len(self.user_feature_names))
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
        return {name: apply_sequence_mask(values, mask) for name, values in features.items()}

    def _history_invariant(
        self, features: Mapping[str, torch.Tensor], batch: Batch
    ) -> torch.Tensor:
        flattened = {
            name: values.reshape(-1, self.projection_dim) for name, values in features.items()
        }
        invariant = self.adversarial.representation(flattened)
        invariant = invariant.reshape(batch.batch_size, batch.sequence_length, self.projection_dim)
        return apply_sequence_mask(invariant, batch.history_mask)

    def forward_batch(self, batch: Batch) -> ModelOutput:
        target = self.project_target(batch)
        history = self.project_history(batch)
        target_specific, target_invariant = self.split(target)
        history_specific, history_invariant = self.split(history)
        history_specific = self._apply_mask(history_specific, batch.history_mask)
        history_invariant = self._apply_mask(history_invariant, batch.history_mask)

        target_private = self.private_fusion(target_specific).fused
        history_private = self.private_fusion(history_specific).fused
        history_private = apply_sequence_mask(history_private, batch.history_mask)
        domain_loss, adversarial_loss, invariant_target = self.adversarial(target_invariant)
        specific_loss = self.specific_classifier(target_specific)
        invariant_history = self._history_invariant(history_invariant, batch)

        target_representation = invariant_target + target_private
        history_representation = invariant_history + history_private
        history_representation = apply_sequence_mask(history_representation, batch.history_mask)
        history_interest = self.attention(
            history_representation,
            batch.history_mask,
            target_representation,
        )
        user = self.project_user(batch)
        logits = self.output(
            self.dnn(torch.cat([user, history_interest, target_representation], dim=-1))
        )
        return ModelOutput(
            logits,
            auxiliary_losses={
                "marn_domain_classifier": domain_loss,
                "marn_adversarial_invariance": (adversarial_loss * self.auxiliary_weight),
                "marn_specific_classifier": specific_loss * self.auxiliary_weight,
            },
            representations={
                "target": target_representation,
                "history_interest": history_interest,
            },
        )


__all__ = ["MARN"]

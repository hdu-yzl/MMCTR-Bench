"""Canonical GMMF model with externally orchestrated adversarial objectives."""

from typing import Dict, Mapping, Optional, Tuple

import torch

from mmctr.core import Batch, ContractError, ModelOutput
from mmctr.models.common.layers import MultiLayerPerceptron
from mmctr.models.common.components import apply_feature_mask, apply_sequence_mask
from mmctr.models.common.sequence import _SequenceMultimodalModel


class _AutoEncoder(torch.nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(dimension, dimension)
        self.decoder = torch.nn.Linear(dimension, dimension)

    def forward(self, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(values)
        return torch.relu(encoded), torch.relu(self.decoder(encoded))


class _Generator(torch.nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(dimension, dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(dimension, dimension),
        )

    def forward(self, identifiers: torch.Tensor) -> torch.Tensor:
        return self.network(identifiers)


class _Discriminator(torch.nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(dimension * 4, dimension * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(dimension * 2, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, identifiers: torch.Tensor, modality: torch.Tensor) -> torch.Tensor:
        values = torch.cat(
            [identifiers, identifiers - modality, identifiers * modality, modality], dim=-1
        )
        return self.network(values)


class _AutoDifference(torch.nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.weight = torch.nn.Linear(dimension * 4, dimension)

    def forward(self, observed: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
        values = torch.cat(
            [observed, generated, observed * generated, observed - generated], dim=-1
        )
        return observed - torch.sigmoid(self.weight(values)) * generated


class _InterestGate(torch.nn.Module):
    def __init__(self, user_dimension: int, modality_dimension: int) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(user_dimension, modality_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(modality_dimension, modality_dimension),
        )

    def forward(self, user: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.network(user), dim=-1)


def _cosine_weighted_pool(
    target: torch.Tensor,
    sequence: torch.Tensor,
    mask: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Pool valid tokens with non-negative, row-normalized target affinity."""

    scores = torch.sum(target.unsqueeze(1) * sequence, dim=-1)
    scores = scores.masked_fill(~mask, 0.0)
    denominator = scores.sum(dim=-1, keepdim=True)
    weights = torch.relu(scores / (denominator + epsilon)).masked_fill(~mask, 0.0)
    pooled = torch.sum(weights.unsqueeze(-1) * sequence, dim=1)
    return apply_feature_mask(pooled, mask.any(dim=1))


class GMMF(_SequenceMultimodalModel):
    """DSN/CGAN multimodal fusion with explicit external alternating phases.

    ``forward_batch`` owns the recommendation and reconstruction objectives;
    trainers must call ``discriminator_loss`` and ``generator_loss`` in their
    own optimizer phases using the disjoint parameter groups exposed here.
    """

    def __init__(self, model_config: Mapping, data_config: Mapping) -> None:
        super().__init__(model_config, data_config)
        self.modal_names = tuple(name for name in self.feature_names if name != "id")
        if not self.modal_names:
            raise ContractError("GMMF requires at least one non-ID modality")
        configured_weights = model_config.get("lambdas", {name: 0.05 for name in self.modal_names})
        if not isinstance(configured_weights, Mapping):
            raise ContractError("GMMF lambdas must be a modality mapping")
        missing_weights = set(self.modal_names).difference(configured_weights)
        if missing_weights:
            raise ContractError(
                "GMMF reconstruction weights are missing {}".format(sorted(missing_weights))
            )
        self.reconstruction_weights = {
            name: float(configured_weights[name]) for name in self.modal_names
        }
        if any(weight < 0.0 for weight in self.reconstruction_weights.values()):
            raise ContractError("GMMF reconstruction weights must be non-negative")
        self.adversarial_start_epoch = int(model_config.get("N", 1))
        self.discriminator_learning_rate = float(model_config.get("lr_disc", 1e-4))
        self.generator_learning_rate = float(model_config.get("lr_gen", 1e-4))
        self.weight_decay = float(model_config.get("l2", 1e-7))
        if self.adversarial_start_epoch < 0:
            raise ContractError("GMMF adversarial start epoch must be non-negative")
        if self.discriminator_learning_rate <= 0.0 or self.generator_learning_rate <= 0.0:
            raise ContractError("GMMF adversarial learning rates must be positive")
        if self.weight_decay < 0.0:
            raise ContractError("GMMF weight decay must be non-negative")

        self.autoencoders = torch.nn.ModuleDict(
            {name: _AutoEncoder(self.projection_dim) for name in self.modal_names}
        )
        self.generators = torch.nn.ModuleDict(
            {name: _Generator(self.projection_dim) for name in self.modal_names}
        )
        self.discriminators = torch.nn.ModuleDict(
            {name: _Discriminator(self.projection_dim) for name in self.modal_names}
        )
        self.differences = torch.nn.ModuleDict(
            {name: _AutoDifference(self.projection_dim) for name in self.modal_names}
        )
        user_dimension = self.projection_dim * len(self.user_feature_names)
        self.gates = torch.nn.ModuleDict(
            {
                name: _InterestGate(user_dimension, self.projection_dim)
                for name in self.feature_names
            }
        )
        predictor_dimension = self.projection_dim * len(self.feature_names) + user_dimension
        self.dnn = MultiLayerPerceptron(
            predictor_dimension,
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

    def optimization_parameter_groups(self) -> Mapping[str, Tuple[torch.nn.Parameter, ...]]:
        """Return complete, disjoint parameter groups for external optimizer composition."""

        discriminator_ids = {id(parameter) for parameter in self.discriminators.parameters()}
        generator_ids = {id(parameter) for parameter in self.generators.parameters()}
        main = tuple(
            parameter
            for parameter in self.parameters()
            if id(parameter) not in discriminator_ids and id(parameter) not in generator_ids
        )
        return {
            "main": main,
            "discriminator": tuple(self.discriminators.parameters()),
            "generator": tuple(self.generators.parameters()),
        }

    @staticmethod
    def _masked_reconstruction(
        reconstructed: torch.Tensor,
        expected: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        squared = (reconstructed - expected).pow(2).mean(dim=-1)
        selected = squared.masked_select(mask)
        return squared.new_zeros(()) if selected.numel() == 0 else selected.mean()

    @staticmethod
    def _binary_loss(
        prediction: torch.Tensor,
        real: bool,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        target = torch.ones_like(prediction) if real else torch.zeros_like(prediction)
        losses = torch.nn.functional.binary_cross_entropy(prediction, target, reduction="none")
        if mask is None:
            return losses.mean()
        selected = losses.squeeze(-1).masked_select(mask)
        return losses.new_zeros(()) if selected.numel() == 0 else selected.mean()

    def _dsn_features(self, batch: Batch):
        target = self.project_target(batch)
        history = self.project_history(batch)
        encoded_target: Dict[str, torch.Tensor] = {}
        encoded_history: Dict[str, torch.Tensor] = {}
        reconstructed_target: Dict[str, torch.Tensor] = {}
        reconstructed_history: Dict[str, torch.Tensor] = {}
        generated_target: Dict[str, torch.Tensor] = {}
        generated_history: Dict[str, torch.Tensor] = {}
        for name in self.modal_names:
            encoded_target[name], reconstructed_target[name] = self.autoencoders[name](target[name])
            encoded_history[name], reconstructed_history[name] = self.autoencoders[name](
                history[name]
            )
            encoded_history[name] = apply_sequence_mask(encoded_history[name], batch.history_mask)
            reconstructed_history[name] = apply_sequence_mask(
                reconstructed_history[name], batch.history_mask
            )
            generated_target[name] = self.generators[name](target["id"])
            generated_history[name] = apply_sequence_mask(
                self.generators[name](history["id"]), batch.history_mask
            )
        return (
            target,
            history,
            encoded_target,
            encoded_history,
            reconstructed_target,
            reconstructed_history,
            generated_target,
            generated_history,
        )

    def forward_batch(self, batch: Batch) -> ModelOutput:
        (
            target,
            history,
            encoded_target,
            encoded_history,
            reconstructed_target,
            reconstructed_history,
            generated_target,
            generated_history,
        ) = self._dsn_features(batch)
        differentiated_target = {
            name: self.differences[name](encoded_target[name], generated_target[name])
            for name in self.modal_names
        }
        differentiated_history = {
            name: apply_sequence_mask(
                self.differences[name](encoded_history[name], generated_history[name]),
                batch.history_mask,
            )
            for name in self.modal_names
        }
        pooled = {
            name: _cosine_weighted_pool(
                differentiated_target[name], differentiated_history[name], batch.history_mask
            )
            for name in self.modal_names
        }
        user = self.project_user(batch)
        gates = {name: self.gates[name](user) for name in self.feature_names}
        fields = [target["id"] * gates["id"]]
        fields.extend(pooled[name] * gates[name] for name in self.modal_names)
        logits = self.output(self.dnn(torch.cat(fields + [user], dim=-1)))
        reconstruction = logits.new_zeros(())
        for name in self.modal_names:
            weight = self.reconstruction_weights[name]
            reconstruction = reconstruction + weight * torch.nn.functional.mse_loss(
                reconstructed_target[name], target[name]
            )
            reconstruction = reconstruction + weight * self._masked_reconstruction(
                reconstructed_history[name],
                history[name],
                batch.history_mask,
            )
        return ModelOutput(
            logits,
            auxiliary_losses={"gmmf_reconstruction": reconstruction},
            representations={
                "history_fusion": torch.cat(
                    [pooled[name] * gates[name] for name in self.modal_names], dim=-1
                ),
                "modality_gates": torch.stack([gates[name] for name in self.feature_names], dim=1),
            },
        )

    def discriminator_loss(self, batch: Batch) -> torch.Tensor:
        """Return the real/fake objective for a discriminator-only update phase."""

        (
            target,
            history,
            encoded_target,
            encoded_history,
            _,
            _,
            generated_target,
            generated_history,
        ) = self._dsn_features(batch)
        total = target["id"].new_zeros(())
        for name in self.modal_names:
            discriminator = self.discriminators[name]
            total = total + (
                self._binary_loss(discriminator(target["id"], encoded_target[name]), True)
                + self._binary_loss(discriminator(target["id"], generated_target[name]), False)
                + self._binary_loss(
                    discriminator(history["id"], encoded_history[name]),
                    True,
                    batch.history_mask,
                )
                + self._binary_loss(
                    discriminator(history["id"], generated_history[name]),
                    False,
                    batch.history_mask,
                )
            )
        return total

    def generator_loss(self, batch: Batch) -> torch.Tensor:
        """Return the fooling objective for a generator-only update phase."""

        (
            target,
            history,
            _,
            _,
            _,
            _,
            generated_target,
            generated_history,
        ) = self._dsn_features(batch)
        total = target["id"].new_zeros(())
        for name in self.modal_names:
            discriminator = self.discriminators[name]
            total = total + (
                self._binary_loss(discriminator(target["id"], generated_target[name]), True)
                + self._binary_loss(
                    discriminator(history["id"], generated_history[name]),
                    True,
                    batch.history_mask,
                )
            )
        return total


__all__ = ["GMMF"]

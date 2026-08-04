"""Modality-balanced recommendation model."""

from typing import Dict, Mapping, Sequence, Tuple

import torch

from mmctr.core import Batch, ContractError, ModelOutput
from mmctr.models.common.multimodal import _PooledMultimodalModel


class _ModalityAttentionFusion(torch.nn.Module):
    def __init__(self, dimension: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(dimension, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Dropout(dropout),
        )
        self.scorer = torch.nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, values: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        stacked = torch.stack(tuple(values), dim=1)
        weights = torch.softmax(self.scorer(self.projection(stacked)).squeeze(-1), dim=1)
        return torch.sum(stacked * weights.unsqueeze(-1), dim=1), weights


class MB(_PooledMultimodalModel):
    """Modality-balanced recommendation with adversarial sensitive-modal samples.

    The PGD-based balance objective is emitted only in training mode and only
    when both configured modalities and both label classes are available.
    """

    def __init__(self, model_config: Mapping, data_config: Mapping) -> None:
        super().__init__(model_config, data_config)
        preferred = ("image", "text", "audio")
        self.modal_names = tuple(name for name in preferred if name in self.feature_names)
        self.history_modal_names = tuple(
            name for name in preferred if name in self.history_feature_names
        )
        self.sensitive_modal = str(model_config.get("sensitive_modal", "image"))
        self.insensitive_modal = str(model_config.get("insensitive_modal", "text"))
        self.balance_weight = float(
            model_config.get("mb_balance_weight", model_config.get("balance_weight", 0.1))
        )
        self.balance_sample_num = int(
            model_config.get("mb_sample_num", model_config.get("balance_sample_num", 30))
        )
        self.adversarial_epsilon = float(
            model_config.get("mb_adv_eps", model_config.get("adv_eps", 1.0))
        )
        self.pgd_steps = int(model_config.get("mb_pgd_steps", model_config.get("pgd_steps", 3)))
        default_step = self.adversarial_epsilon / max(self.pgd_steps, 1)
        self.pgd_step_size = float(
            model_config.get("mb_pgd_step_size", model_config.get("pgd_step_size", default_step))
        )
        if self.balance_weight < 0.0:
            raise ContractError("MB balance weight must be non-negative")
        if self.balance_sample_num <= 0:
            raise ContractError("MB balance sample count must be positive")
        if self.adversarial_epsilon < 0.0 or self.pgd_step_size < 0.0:
            raise ContractError("MB adversarial radii must be non-negative")
        if self.pgd_steps <= 0:
            raise ContractError("MB PGD steps must be positive")

        self.user_id_encoder = torch.nn.Linear(self.latent_dim, self.projection_dim)
        self.item_id_encoder = torch.nn.Linear(self.latent_dim, self.projection_dim)
        self.id_pair_encoder = torch.nn.Linear(self.latent_dim * 2, self.projection_dim)
        self.user_modal_encoders = torch.nn.ModuleDict(
            {
                name: torch.nn.Linear(self.latent_dim, self.projection_dim)
                for name in self.modal_names
            }
        )
        all_modal_names = tuple(dict.fromkeys(self.modal_names + self.history_modal_names))
        self.item_modal_encoders = torch.nn.ModuleDict(
            {
                name: torch.nn.Sequential(
                    torch.nn.Linear(self.projection_dim, self.projection_dim),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(self.dropout),
                    torch.nn.Linear(self.projection_dim, self.projection_dim),
                )
                for name in all_modal_names
            }
        )
        fusion_hidden_dim = int(model_config.get("mb_fusion_hidden_dim", self.projection_dim))
        if fusion_hidden_dim <= 0:
            raise ContractError("MB fusion hidden dimension must be positive")
        self.fusion = _ModalityAttentionFusion(self.projection_dim, fusion_hidden_dim, self.dropout)
        self.dnn, self.output = self.make_predictor(self.projection_dim * 6)

    def _raw_ids(
        self, batch: Batch
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        try:
            raw_user = self.embedding(batch.user_features["id"]).squeeze(1)
            raw_item = self.embedding(batch.item_features["id"]).squeeze(1)
        except KeyError as error:
            raise ContractError("MB requires user and item IDs") from error
        return (
            raw_user,
            self.user_id_encoder(raw_user),
            self.item_id_encoder(raw_item),
            self.id_pair_encoder(torch.cat([raw_user, raw_item], dim=-1)),
        )

    def _modal_score(
        self,
        raw_user: torch.Tensor,
        modal_values: Mapping[str, torch.Tensor],
        name: str,
    ) -> torch.Tensor:
        if name not in modal_values:
            return raw_user.new_zeros(raw_user.shape[0], 1)
        user = torch.nn.functional.normalize(self.user_modal_encoders[name](raw_user), dim=-1)
        item = torch.nn.functional.normalize(modal_values[name], dim=-1)
        return torch.sum(user * item, dim=-1, keepdim=True)

    def _can_balance(self, values: Mapping[str, torch.Tensor]) -> bool:
        return (
            self.balance_weight > 0.0
            and self.sensitive_modal != self.insensitive_modal
            and self.sensitive_modal in values
            and self.insensitive_modal in values
        )

    def _adversarial_sensitive_embedding(
        self,
        raw_features: Mapping[str, torch.Tensor],
        positive_indices: torch.Tensor,
        negative_indices: torch.Tensor,
    ) -> torch.Tensor:
        name = self.sensitive_modal
        initial = raw_features[name][positive_indices].detach()
        negative = raw_features[name][negative_indices]
        target = self.item_modal_encoders[name](self.target_projectors[name](negative)).detach()
        adversarial = initial.clone().requires_grad_(True)
        for _ in range(self.pgd_steps):
            embedding = self.item_modal_encoders[name](self.target_projectors[name](adversarial))
            gradient = torch.autograd.grad(
                torch.nn.functional.mse_loss(embedding, target), adversarial
            )[0]
            adversarial = torch.clamp(
                adversarial - self.pgd_step_size * gradient.sign(),
                initial - self.adversarial_epsilon,
                initial + self.adversarial_epsilon,
            )
            adversarial = adversarial.detach().requires_grad_(True)
        return self.item_modal_encoders[name](self.target_projectors[name](adversarial.detach()))

    def _balance_loss(
        self,
        batch: Batch,
        raw_user: torch.Tensor,
        modal_values: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        positive = torch.nonzero(batch.labels > 0.5, as_tuple=False).view(-1)
        negative = torch.nonzero(batch.labels <= 0.5, as_tuple=False).view(-1)
        sample_count = min(positive.numel(), negative.numel(), self.balance_sample_num)
        if sample_count <= 0:
            return raw_user.new_zeros(())
        positive = positive[torch.randperm(positive.numel(), device=positive.device)[:sample_count]]
        negative = negative[
            torch.randint(0, negative.numel(), (sample_count,), device=negative.device)
        ]
        sensitive = self.sensitive_modal
        insensitive = self.insensitive_modal
        sensitive_user = torch.nn.functional.normalize(
            self.user_modal_encoders[sensitive](raw_user[positive]), dim=-1
        )
        insensitive_user = torch.nn.functional.normalize(
            self.user_modal_encoders[insensitive](raw_user[positive]), dim=-1
        )
        sensitive_positive = torch.nn.functional.normalize(
            modal_values[sensitive][positive], dim=-1
        )
        raw_features = {name: self._target_feature(batch, name) for name in self.modal_names}
        sensitive_adversarial = torch.nn.functional.normalize(
            self._adversarial_sensitive_embedding(raw_features, positive, negative),
            dim=-1,
        )
        insensitive_positive = torch.nn.functional.normalize(
            modal_values[insensitive][positive], dim=-1
        )
        insensitive_negative = torch.nn.functional.normalize(
            modal_values[insensitive][negative], dim=-1
        )
        sensitive_margin = torch.sum(sensitive_user * sensitive_positive, dim=-1) - torch.sum(
            sensitive_user * sensitive_adversarial, dim=-1
        )
        insensitive_margin = torch.sum(insensitive_user * insensitive_positive, dim=-1) - torch.sum(
            insensitive_user * insensitive_negative, dim=-1
        )
        return self.balance_weight * torch.relu(sensitive_margin - insensitive_margin).mean()

    def forward_batch(self, batch: Batch) -> ModelOutput:
        projected_target = self.project_target(batch)
        projected_history = self.project_history(batch)
        raw_user, user_id, item_id, pair_id = self._raw_ids(batch)
        modal_values = {
            name: self.item_modal_encoders[name](projected_target[name])
            for name in self.modal_names
        }
        fused_modal, modal_weights = self.fusion(tuple(modal_values.values()) or (item_id,))
        pooled_history = {
            name: self.masked_pool(values, batch.history_mask)
            for name, values in projected_history.items()
        }
        history_values = [
            self.item_modal_encoders[name](pooled_history[name])
            for name in self.history_modal_names
        ]
        history_fusion = self.fusion(history_values)[0] if history_values else pooled_history["id"]
        modal_scores = {
            name: self._modal_score(raw_user, modal_values, name) for name in self.modal_names
        }
        id_score = torch.sum(
            torch.nn.functional.normalize(user_id, dim=-1)
            * torch.nn.functional.normalize(item_id, dim=-1),
            dim=-1,
            keepdim=True,
        )
        predictor_input = torch.cat(
            [
                pair_id,
                user_id,
                item_id,
                fused_modal,
                history_fusion,
                user_id * fused_modal,
            ],
            dim=-1,
        )
        logits = self.output(self.dnn(predictor_input)) + id_score
        logits = logits + sum(modal_scores.values(), id_score.new_zeros(()))
        balance_loss = logits.new_zeros(())
        if self.training and self._can_balance(modal_values):
            balance_loss = self._balance_loss(batch, raw_user, modal_values)
        representations: Dict[str, torch.Tensor] = {
            "id_score": id_score,
            "modal_weights": modal_weights,
        }
        representations.update(
            {"{}_score".format(name): score for name, score in modal_scores.items()}
        )
        return ModelOutput(
            logits,
            auxiliary_losses={"mb_modality_balance": balance_loss},
            representations=representations,
        )


__all__ = ["MB"]

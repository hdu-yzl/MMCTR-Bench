"""Canonical specialized multimodal models migrated from legacy research modules."""

import math
from itertools import combinations
from typing import Dict, Mapping, Sequence, Tuple

import torch

from mmctr.core import Batch, ContractError, ModelOutput
from mmctr.models.baselines.layers import MultiLayerPerceptron
from mmctr.models.components import apply_sequence_mask
from mmctr.models.multimodal import _PooledMultimodalModel
from mmctr.models.sequence import _SequenceMultimodalModel


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


class _PAMDDisentangleBlock(torch.nn.Module):
    def __init__(
        self,
        dimension: int,
        hidden_dim: int,
        dropout: float,
        layer_norm: bool,
    ) -> None:
        super().__init__()
        self.common_a = self._mlp(dimension, hidden_dim, dropout, layer_norm)
        self.common_b = self._mlp(dimension, hidden_dim, dropout, layer_norm)
        self.a_to_b = self._mlp(dimension, hidden_dim, dropout, layer_norm)
        self.b_to_a = self._mlp(dimension, hidden_dim, dropout, layer_norm)
        self.query = torch.nn.Linear(dimension, dimension, bias=False)
        self.key = torch.nn.Linear(dimension, dimension, bias=False)

    @staticmethod
    def _mlp(
        dimension: int, hidden_dim: int, dropout: float, layer_norm: bool
    ) -> torch.nn.Sequential:
        layers = [
            torch.nn.Linear(dimension, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, dimension),
        ]
        if layer_norm:
            layers.append(torch.nn.LayerNorm(dimension))
        return torch.nn.Sequential(*layers)

    def forward(
        self, first: torch.Tensor, second: torch.Tensor, query: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        common_first = self.common_a(first)
        common_second = self.common_b(second)
        specific_first = first - common_first
        specific_second = second - common_second
        representations = torch.stack(
            [common_first, common_second, specific_first, specific_second], dim=-2
        )
        scores = torch.sum(self.query(query).unsqueeze(-2) * self.key(representations), dim=-1)
        weights = torch.softmax(scores / math.sqrt(query.shape[-1]), dim=-1)
        fused = query + torch.sum(weights.unsqueeze(-1) * representations, dim=-2)

        alignment = torch.nn.functional.mse_loss(common_first, common_second)
        orthogonal = torch.mean(
            torch.sum(
                torch.nn.functional.normalize(specific_first, dim=-1)
                * torch.nn.functional.normalize(specific_second, dim=-1),
                dim=-1,
            ).pow(2)
        )
        common_loss = torch.nn.functional.mse_loss(
            second, self.a_to_b(common_first)
        ) + torch.nn.functional.mse_loss(first, self.b_to_a(common_second))
        complete_loss = torch.nn.functional.mse_loss(
            second, self.a_to_b(first)
        ) + torch.nn.functional.mse_loss(first, self.b_to_a(second))
        specific_loss = torch.nn.functional.mse_loss(
            second, self.a_to_b(specific_first)
        ) + torch.nn.functional.mse_loss(first, self.b_to_a(specific_second))
        ranking = -torch.nn.functional.logsigmoid(complete_loss - common_loss)
        ranking = ranking - torch.nn.functional.logsigmoid(specific_loss - complete_loss)
        return fused, alignment + orthogonal + ranking


class PAMD(_PooledMultimodalModel):
    """Pairwise adaptive modality disentanglement over target and history.

    Each modality pair is decomposed into common and residual components; the
    auxiliary loss aligns common components, discourages residual correlation,
    and ranks cross-reconstruction quality.
    """

    def __init__(self, model_config: Mapping, data_config: Mapping) -> None:
        super().__init__(model_config, data_config)
        preferred = ("image", "text", "audio")
        self.modal_names = tuple(name for name in preferred if name in self.feature_names)
        self.history_modal_names = tuple(
            name for name in preferred if name in self.history_feature_names
        )
        if len(self.modal_names) < 2 or len(self.history_modal_names) < 2:
            raise ContractError("PAMD requires at least two target and history modalities")
        hidden_dim = int(model_config.get("pamd_hidden_dim", self.projection_dim))
        self.auxiliary_weight = float(model_config.get("pamd_aux_weight", 0.1))
        if hidden_dim <= 0 or self.auxiliary_weight < 0.0:
            raise ContractError("PAMD hidden dimension must be positive and weight non-negative")
        layer_norm = bool(model_config.get("pamd_layer_norm", True))
        self.target_blocks = self._build_blocks(self.modal_names, hidden_dim, layer_norm)
        self.history_blocks = self._build_blocks(self.history_modal_names, hidden_dim, layer_norm)
        self.dnn, self.output = self.make_predictor(self.projection_dim * 2)

    @staticmethod
    def _pair_key(first: str, second: str) -> str:
        return "{}__{}".format(first, second)

    def _build_blocks(
        self, names: Sequence[str], hidden_dim: int, layer_norm: bool
    ) -> torch.nn.ModuleDict:
        return torch.nn.ModuleDict(
            {
                self._pair_key(first, second): _PAMDDisentangleBlock(
                    self.projection_dim,
                    hidden_dim,
                    self.dropout,
                    layer_norm,
                )
                for first, second in combinations(names, 2)
            }
        )

    def _run_blocks(
        self,
        blocks: torch.nn.ModuleDict,
        names: Sequence[str],
        values: Mapping[str, torch.Tensor],
        query: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fused_values = []
        losses = []
        for first, second in combinations(names, 2):
            fused, loss = blocks[self._pair_key(first, second)](
                values[first], values[second], query
            )
            fused_values.append(fused)
            losses.append(loss)
        return (
            torch.stack(fused_values, dim=1).mean(dim=1),
            torch.stack(losses).mean(),
        )

    def forward_batch(self, batch: Batch) -> ModelOutput:
        target = self.project_target(batch)
        history = {
            name: self.masked_pool(values, batch.history_mask)
            for name, values in self.project_history(batch).items()
        }
        target_fusion, target_loss = self._run_blocks(
            self.target_blocks, self.modal_names, target, target["id"]
        )
        history_fusion, history_loss = self._run_blocks(
            self.history_blocks,
            self.history_modal_names,
            history,
            history["id"],
        )
        logits = self.output(self.dnn(torch.cat([target_fusion, history_fusion], dim=-1)))
        return ModelOutput(
            logits,
            auxiliary_losses={
                "pamd_disentanglement": self.auxiliary_weight * (target_loss + history_loss)
            },
            representations={
                "history_fusion": history_fusion,
                "target_fusion": target_fusion,
            },
        )


class _MixerBlock(torch.nn.Module):
    def __init__(
        self,
        sequence_length: int,
        dimension: int,
        token_hidden_dim: int,
        channel_hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.token_norm = torch.nn.LayerNorm(dimension)
        self.token_mlp = torch.nn.Sequential(
            torch.nn.Linear(sequence_length, token_hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(token_hidden_dim, sequence_length),
            torch.nn.Dropout(dropout),
        )
        self.channel_norm = torch.nn.LayerNorm(dimension)
        self.channel_mlp = torch.nn.Sequential(
            torch.nn.Linear(dimension, channel_hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(channel_hidden_dim, dimension),
            torch.nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        values = apply_sequence_mask(values, mask)
        mixed_tokens = self.token_mlp(self.token_norm(values).transpose(1, 2)).transpose(1, 2)
        values = apply_sequence_mask(values + mixed_tokens, mask)
        values = values + self.channel_mlp(self.channel_norm(values))
        return apply_sequence_mask(values, mask)


class _MixerModule(torch.nn.Module):
    def __init__(
        self,
        sequence_length: int,
        dimension: int,
        layer_count: int,
        token_hidden_dim: int,
        channel_hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                _MixerBlock(
                    sequence_length,
                    dimension,
                    token_hidden_dim,
                    channel_hidden_dim,
                    dropout,
                )
                for _ in range(layer_count)
            ]
        )
        self.final_norm = torch.nn.LayerNorm(dimension)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            values = layer(values, mask)
        return apply_sequence_mask(self.final_norm(values), mask)


def _last_valid_token(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
    positions = positions.expand(mask.shape[0], -1).masked_fill(~mask, -1)
    indices = positions.max(dim=1).values
    gathered = values.gather(
        1,
        indices.clamp_min(0).view(-1, 1, 1).expand(-1, 1, values.shape[-1]),
    ).squeeze(1)
    return gathered.masked_fill(indices.eq(-1).unsqueeze(-1), 0.0)


class MMMLP(_SequenceMultimodalModel):
    """Mask-aware MLP-Mixer over modality-specific and fused history tokens.

    Token-mixing weights are tied to configured ``seq_len``; batches with a
    different padded length are rejected. Fully padded rows produce a zero
    history vector rather than selecting an arbitrary token.
    """

    def __init__(self, model_config: Mapping, data_config: Mapping) -> None:
        super().__init__(model_config, data_config)
        if not self.feature_names:
            raise ContractError("MMMLP requires at least one feature")
        try:
            self.sequence_length = int(data_config["seq_len"])
        except KeyError as error:
            raise ContractError("MMMLP requires data seq_len") from error
        feature_layers = int(
            model_config.get("feature_mixer_layers", model_config.get("mixer_layers", 2))
        )
        fusion_layers = int(
            model_config.get("fusion_mixer_layers", model_config.get("mixer_layers", 2))
        )
        token_hidden_dim = int(
            model_config.get("token_hidden_dim", max(self.sequence_length * 2, 1))
        )
        channel_hidden_dim = int(model_config.get("channel_hidden_dim", self.projection_dim * 2))
        self.fusion_dim = self.projection_dim * len(self.feature_names)
        fusion_channel_dim = int(model_config.get("fusion_channel_hidden_dim", self.fusion_dim * 2))
        dimensions = (
            self.sequence_length,
            feature_layers,
            fusion_layers,
            token_hidden_dim,
            channel_hidden_dim,
            fusion_channel_dim,
        )
        if any(value <= 0 for value in dimensions):
            raise ContractError("MMMLP mixer dimensions and layers must be positive")
        self.feature_mixers = torch.nn.ModuleDict(
            {
                name: _MixerModule(
                    self.sequence_length,
                    self.projection_dim,
                    feature_layers,
                    token_hidden_dim,
                    channel_hidden_dim,
                    self.dropout,
                )
                for name in self.feature_names
            }
        )
        self.fusion_mixer = _MixerModule(
            self.sequence_length,
            self.fusion_dim,
            fusion_layers,
            token_hidden_dim,
            fusion_channel_dim,
            self.dropout,
        )
        self.target_projector = torch.nn.Sequential(
            torch.nn.LayerNorm(self.fusion_dim),
            torch.nn.Linear(self.fusion_dim, self.fusion_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(self.dropout),
        )
        predictor_dim = self.fusion_dim * 3 + self.projection_dim * len(self.user_feature_names)
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
        if batch.sequence_length != self.sequence_length:
            raise ContractError("MMMLP batch sequence length does not match configured seq_len")
        history = self.project_history(batch)
        mixed_modalities = [
            self.feature_mixers[name](history[name], batch.history_mask)
            for name in self.feature_names
        ]
        mixed_history = self.fusion_mixer(torch.cat(mixed_modalities, dim=-1), batch.history_mask)
        history_vector = _last_valid_token(mixed_history, batch.history_mask)
        target = self.project_target(batch)
        target_vector = self.target_projector(
            torch.cat([target[name] for name in self.feature_names], dim=-1)
        )
        user = self.project_user(batch)
        logits = self.output(
            self.dnn(
                torch.cat(
                    [
                        history_vector,
                        target_vector,
                        history_vector * target_vector,
                        user,
                    ],
                    dim=-1,
                )
            )
        )
        return ModelOutput(
            logits,
            representations={
                "history_fusion": history_vector,
                "target_fusion": target_vector,
            },
        )


class _FeedForwardExpert(torch.nn.Module):
    def __init__(self, dimension: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(dimension, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, dimension),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class _MoELayer(torch.nn.Module):
    def __init__(self, dimension: int, hidden_dim: int, expert_count: int, dropout: float) -> None:
        super().__init__()
        self.router = torch.nn.Linear(dimension, expert_count)
        self.experts = torch.nn.ModuleList(
            [_FeedForwardExpert(dimension, hidden_dim, dropout) for _ in range(expert_count)]
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.normalisation = torch.nn.LayerNorm(dimension)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        gates = torch.softmax(self.router(values), dim=-1)
        expert_values = torch.stack([expert(values) for expert in self.experts], dim=-2)
        mixed = torch.sum(gates.unsqueeze(-1) * expert_values, dim=-2)
        return self.normalisation(values + self.dropout(mixed))


class _SelfAttentionBlock(torch.nn.Module):
    def __init__(self, dimension: int, heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        if heads <= 0 or dimension % heads != 0:
            raise ContractError("M3SRec projection dimension must be divisible by positive heads")
        self.attention = torch.nn.MultiheadAttention(
            dimension, heads, dropout=dropout, batch_first=True
        )
        self.first_norm = torch.nn.LayerNorm(dimension)
        self.second_norm = torch.nn.LayerNorm(dimension)
        self.dropout = torch.nn.Dropout(dropout)
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(dimension, ffn_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(ffn_dim, dimension),
        )

    def forward(self, values: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        safe_mask = padding_mask
        all_padding = padding_mask.all(dim=1)
        if all_padding.any():
            safe_mask = padding_mask.clone()
            safe_mask[all_padding] = False
        attended, _ = self.attention(
            values,
            values,
            values,
            key_padding_mask=safe_mask,
            need_weights=False,
        )
        if all_padding.any():
            attended = attended.masked_fill(all_padding.view(-1, 1, 1), 0.0)
        values = self.first_norm(values + self.dropout(attended))
        return self.second_norm(values + self.dropout(self.feed_forward(values)))


class _VectorAttentionFusion(torch.nn.Module):
    def __init__(self, dimension: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(dimension, hidden_dim)
        self.score = torch.nn.Linear(hidden_dim, 1, bias=False)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(torch.tanh(self.projection(values))).squeeze(-1), dim=-1)
        return self.dropout(torch.sum(weights.unsqueeze(-1) * values, dim=1))


class M3SRec(_SequenceMultimodalModel):
    """Shared-attention and mixture-of-experts multimodal sequential model.

    Modality sequences are concatenated along the token axis for shared
    attention, then split back into ``[batch, length, projection_dim]`` chunks.
    All-padding rows are made attention-safe and reduce to zero history tokens.
    """

    def __init__(self, model_config: Mapping, data_config: Mapping) -> None:
        super().__init__(model_config, data_config)
        configured_names = model_config.get("m3_modalities")
        if configured_names is None:
            self.modal_names = self.feature_names
        elif isinstance(configured_names, str):
            raise ContractError("M3SRec m3_modalities must be a sequence of names")
        else:
            self.modal_names = tuple(dict.fromkeys(configured_names))
        missing = [name for name in self.modal_names if name not in self.feature_names]
        if not self.modal_names or missing:
            raise ContractError(
                "M3SRec invalid modalities: {}, missing={}".format(self.modal_names, missing)
            )
        heads = int(model_config.get("num_heads", 4))
        expert_count = int(model_config.get("num_experts", 4))
        expert_hidden_dim = int(model_config.get("moe_hidden_dim", self.projection_dim * 4))
        attention_ffn_dim = int(model_config.get("attn_ffn_dim", self.projection_dim * 4))
        specific_layers = int(model_config.get("num_specific_layers", 1))
        cross_layers = int(model_config.get("num_cross_layers", 1))
        default_max_length = data_config.get("seq_len")
        if default_max_length is None and "max_seq_len" not in model_config:
            raise ContractError("M3SRec requires max_seq_len or data seq_len")
        self.max_sequence_length = int(model_config.get("max_seq_len", default_max_length))
        positive_values = (
            expert_count,
            expert_hidden_dim,
            attention_ffn_dim,
            specific_layers,
            cross_layers,
            self.max_sequence_length,
        )
        if any(value <= 0 for value in positive_values):
            raise ContractError("M3SRec dimensions, experts, and layers must be positive")
        self.position_embedding = torch.nn.Embedding(self.max_sequence_length, self.projection_dim)
        self.modality_embedding = torch.nn.Embedding(len(self.modal_names), self.projection_dim)
        self.input_attention = _SelfAttentionBlock(
            self.projection_dim, heads, attention_ffn_dim, self.dropout
        )
        self.specific_experts = torch.nn.ModuleDict(
            {
                name: torch.nn.ModuleList(
                    [
                        _MoELayer(
                            self.projection_dim,
                            expert_hidden_dim,
                            expert_count,
                            self.dropout,
                        )
                        for _ in range(specific_layers)
                    ]
                )
                for name in self.modal_names
            }
        )
        self.cross_attention = torch.nn.ModuleList(
            [
                _SelfAttentionBlock(
                    self.projection_dim,
                    heads,
                    attention_ffn_dim,
                    self.dropout,
                )
                for _ in range(cross_layers)
            ]
        )
        self.cross_experts = torch.nn.ModuleList(
            [
                _MoELayer(
                    self.projection_dim,
                    expert_hidden_dim,
                    expert_count,
                    self.dropout,
                )
                for _ in range(cross_layers)
            ]
        )
        fusion_hidden_dim = int(model_config.get("fusion_hidden_dim", self.projection_dim))
        if fusion_hidden_dim <= 0:
            raise ContractError("M3SRec fusion hidden dimension must be positive")
        self.history_fusion = _VectorAttentionFusion(
            self.projection_dim, fusion_hidden_dim, self.dropout
        )
        self.target_fusion = _VectorAttentionFusion(
            self.projection_dim, fusion_hidden_dim, self.dropout
        )
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

    def _add_embeddings(self, sequences: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        encoded = []
        for index, sequence in enumerate(sequences):
            batch_size, sequence_length, _ = sequence.shape
            if sequence_length > self.max_sequence_length:
                raise ContractError("M3SRec sequence length exceeds configured maximum")
            positions = torch.arange(sequence_length, device=sequence.device)
            positions = positions.unsqueeze(0).expand(batch_size, -1)
            modalities = torch.full(
                (batch_size, sequence_length),
                index,
                dtype=torch.long,
                device=sequence.device,
            )
            encoded.append(
                sequence + self.position_embedding(positions) + self.modality_embedding(modalities)
            )
        return encoded

    def _split_modalities(
        self, values: torch.Tensor, sequence_length: int
    ) -> Sequence[torch.Tensor]:
        return tuple(
            values[:, index * sequence_length : (index + 1) * sequence_length]
            for index in range(len(self.modal_names))
        )

    def _encode_history(
        self, history: Mapping[str, torch.Tensor], mask: torch.Tensor
    ) -> torch.Tensor:
        sequences = self._add_embeddings([history[name] for name in self.modal_names])
        sequence_length = mask.shape[1]
        full_padding_mask = torch.cat([~mask for _ in self.modal_names], dim=1)
        values = self.input_attention(torch.cat(tuple(sequences), dim=1), full_padding_mask)
        refined = []
        for name, chunk in zip(self.modal_names, self._split_modalities(values, sequence_length)):
            for expert in self.specific_experts[name]:
                chunk = expert(chunk)
            refined.append(chunk)
        values = torch.cat(refined, dim=1)
        for attention, expert in zip(self.cross_attention, self.cross_experts):
            values = expert(attention(values, full_padding_mask))
        modal_vectors = torch.stack(
            [
                _last_valid_token(chunk, mask)
                for chunk in self._split_modalities(values, sequence_length)
            ],
            dim=1,
        )
        return self.history_fusion(modal_vectors)

    def forward_batch(self, batch: Batch) -> ModelOutput:
        history_vector = self._encode_history(self.project_history(batch), batch.history_mask)
        target = self.project_target(batch)
        target_vector = self.target_fusion(
            torch.stack([target[name] for name in self.modal_names], dim=1)
        )
        user = self.project_user(batch)
        logits = self.output(self.dnn(torch.cat([history_vector, target_vector, user], dim=-1)))
        return ModelOutput(
            logits,
            representations={
                "history_fusion": history_vector,
                "target_fusion": target_vector,
            },
        )


__all__ = ["M3SRec", "MB", "MMMLP", "PAMD"]

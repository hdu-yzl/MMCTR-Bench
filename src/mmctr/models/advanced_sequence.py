"""Canonical advanced sequence models with label-aware auxiliary objectives."""

import math
from typing import Dict, Mapping, Sequence

import torch

from mmctr.core import Batch, ContractError, ModelOutput
from mmctr.models.baselines.layers import CrossNetwork, DinAttention, MultiLayerPerceptron
from mmctr.models.components import apply_sequence_mask
from mmctr.models.sequence import _SequenceMultimodalModel


class _FQAttentionLayer(torch.nn.Module):
    def __init__(
        self,
        dimension: int,
        heads: int,
        attention_dropout: float,
        hidden_dropout: float,
    ) -> None:
        super().__init__()
        if heads <= 0 or dimension % heads != 0:
            raise ContractError("FQ-Former dimension must be divisible by positive heads")
        self.heads = heads
        self.head_dimension = dimension // heads
        self.query = torch.nn.Linear(dimension, dimension)
        self.key = torch.nn.Linear(dimension, dimension)
        self.value = torch.nn.Linear(dimension, dimension)
        self.attention_dropout = torch.nn.Dropout(attention_dropout)
        self.output = torch.nn.Linear(dimension, dimension)
        self.output_dropout = torch.nn.Dropout(hidden_dropout)
        self.normalisation = torch.nn.LayerNorm(dimension, eps=1e-5)

    def _split_heads(self, values: torch.Tensor) -> torch.Tensor:
        batch_size, length, _ = values.shape
        return values.view(
            batch_size, length, self.heads, self.head_dimension
        ).permute(0, 2, 1, 3)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        query = self._split_heads(self.query(values))
        key = self._split_heads(self.key(values))
        value = self._split_heads(self.value(values))
        scale = math.sqrt(self.head_dimension + 1e-5)
        probabilities = self.attention_dropout(
            torch.softmax(torch.matmul(query, key.transpose(-1, -2)) / scale, dim=-1)
        )
        attended = torch.matmul(probabilities, value)
        attended = attended.permute(0, 2, 1, 3).contiguous()
        attended = attended.view(values.shape[0], values.shape[1], -1)
        attended = self.output_dropout(self.output(attended))
        return self.normalisation(attended + values)


class _FQFormer(torch.nn.Module):
    def __init__(
        self,
        dimension: int,
        query_count: int,
        layer_count: int = 2,
        heads: int = 8,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if query_count <= 0 or layer_count <= 0:
            raise ContractError("FQ-Former query and layer counts must be positive")
        self.query_count = query_count
        self.dimension = dimension
        self.queries = torch.nn.Parameter(torch.empty(1, query_count, dimension))
        torch.nn.init.xavier_uniform_(self.queries)
        self.layers = torch.nn.ModuleList(
            [
                _FQAttentionLayer(
                    dimension,
                    heads,
                    attention_dropout,
                    hidden_dropout,
                )
                for _ in range(layer_count)
            ]
        )

    @property
    def output_dim(self) -> int:
        return self.query_count * self.dimension

    def forward(self, modality_tokens: torch.Tensor) -> torch.Tensor:
        queries = self.queries.expand(modality_tokens.shape[0], -1, -1)
        values = torch.cat([queries, modality_tokens], dim=1)
        for layer in self.layers:
            values = layer(values)
        return values[:, : self.query_count].reshape(modality_tokens.shape[0], -1)


class EM3(_SequenceMultimodalModel):
    def __init__(self, model_config: Mapping, data_config: Mapping) -> None:
        super().__init__(model_config, data_config)
        self.query_count = int(model_config.get("query_num", 5))
        self.cic_temperature = float(model_config.get("cic_tau", 0.1))
        self.cic_weight = float(model_config.get("cic_weight", 0.1))
        if self.cic_temperature <= 0.0 or self.cic_weight < 0.0:
            raise ContractError("EM3 requires positive cic_tau and non-negative cic_weight")
        self.fq_former = _FQFormer(
            self.projection_dim,
            self.query_count,
            layer_count=int(model_config.get("fq_layer_num", 2)),
            heads=int(model_config.get("fq_heads", 8)),
        )
        fusion_dim = self.fq_former.output_dim
        self.attention = DinAttention(fusion_dim, self.mlp_dims, self.dropout)
        self.content_map = MultiLayerPerceptron(
            fusion_dim,
            [self.projection_dim],
            self.dropout,
            batch_norm=self.batch_norm,
        )
        predictor_dim = (
            fusion_dim
            + self.projection_dim * len(self.user_feature_names)
            + self.projection_dim
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

    def _content_item_loss(
        self, content: torch.Tensor, item_id: torch.Tensor
    ) -> torch.Tensor:
        content_to_item = torch.matmul(content, item_id.transpose(0, 1))
        content_to_item = content_to_item / self.cic_temperature
        labels = torch.arange(content.shape[0], device=content.device)
        return (
            torch.nn.functional.cross_entropy(content_to_item, labels)
            + torch.nn.functional.cross_entropy(content_to_item.transpose(0, 1), labels)
        ) / 2.0

    def forward_batch(self, batch: Batch) -> ModelOutput:
        target_features = self.project_target(batch)
        history_features = self.project_history(batch)
        target_tokens = torch.stack(
            [target_features[name] for name in self.feature_names], dim=1
        )
        history_tokens = torch.stack(
            [history_features[name] for name in self.feature_names], dim=2
        )
        flat_history = history_tokens.reshape(
            batch.batch_size * batch.sequence_length,
            len(self.feature_names),
            self.projection_dim,
        )
        target_fusion = self.fq_former(target_tokens)
        history_fusion = self.fq_former(flat_history).reshape(
            batch.batch_size, batch.sequence_length, -1
        )
        history_fusion = apply_sequence_mask(history_fusion, batch.history_mask)
        history_interest = self.attention(
            target_fusion, history_fusion, batch.history_mask
        )
        content = self.content_map(target_fusion)
        cic_loss = self._content_item_loss(content, target_features["id"])
        user = self.project_user(batch)
        logits = self.output(
            self.dnn(torch.cat([user, history_interest, content], dim=-1))
        )
        return ModelOutput(
            logits,
            auxiliary_losses={"em3_content_item_contrastive": cic_loss * self.cic_weight},
            representations={
                "content": content,
                "history_interest": history_interest,
                "target_queries": target_fusion,
            },
        )


class _CrossModalAttention(torch.nn.Module):
    def __init__(self, dimension: int, heads: int) -> None:
        super().__init__()
        if heads <= 0 or dimension % heads != 0:
            raise ContractError("cross-modal dimension must be divisible by positive heads")
        self.heads = heads
        self.head_dimension = dimension // heads
        self.scale = self.head_dimension ** -0.5
        self.query = torch.nn.Linear(dimension, dimension, bias=False)
        self.key = torch.nn.Linear(dimension, dimension, bias=False)
        self.value = torch.nn.Linear(dimension, dimension, bias=False)
        self.output = torch.nn.Linear(dimension, dimension)
        self.residual_weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        batch_size = target.shape[0]
        query = self.query(target).view(batch_size, self.heads, self.head_dimension)
        key = self.key(source).view(batch_size, self.heads, self.head_dimension)
        value = self.value(source).view(batch_size, self.heads, self.head_dimension)
        scores = torch.einsum("bhd,bhd->bh", query, key) * self.scale
        weights = torch.softmax(scores, dim=-1)
        attended = (weights.unsqueeze(-1) * value).reshape(batch_size, -1)
        return self.residual_weight * self.output(attended) + target


class _StochasticReverseFusion(torch.nn.Module):
    def __init__(
        self, dimension: int, feature_names: Sequence[str], steps: int
    ) -> None:
        super().__init__()
        if steps <= 0 or len(feature_names) < 2:
            raise ContractError("SRC fusion requires positive steps and two modalities")
        if dimension % 8 != 0:
            raise ContractError("SRC projection_dim must be divisible by 8")
        self.feature_names = tuple(feature_names)
        self.steps = steps
        self.epsilon = 1e-8
        angles = torch.linspace(0, 0.9 * (torch.pi / 2), steps + 1)
        alpha_bars = torch.cos(angles) ** 2
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer(
            "alphas", alpha_bars[1:] / (alpha_bars[:-1] + self.epsilon)
        )
        self.source_weights = torch.nn.ParameterDict(
            {
                "{}_to_{}".format(source, target): torch.nn.Parameter(
                    torch.full((1,), 0.5)
                )
                for source in self.feature_names
                for target in self.feature_names
                if source != target
            }
        )
        self.attention = torch.nn.ModuleDict(
            {
                name: _CrossModalAttention(dimension, 8)
                for name in self.feature_names
            }
        )
        self.fusion = torch.nn.Sequential(
            torch.nn.LayerNorm(len(self.feature_names) * dimension),
            torch.nn.Linear(len(self.feature_names) * dimension, 2 * dimension),
            torch.nn.GELU(),
            torch.nn.Linear(2 * dimension, dimension),
        )
        for layer in self.fusion:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)
                torch.nn.init.constant_(layer.bias, 0.1)

    def _diffuse(self, values: torch.Tensor, step: int) -> torch.Tensor:
        alpha = self.alphas[step]
        if not self.training:
            return values * torch.sqrt(alpha + self.epsilon)
        noise = torch.randn_like(values)
        return values * torch.sqrt(alpha + self.epsilon) + noise * torch.sqrt(
            torch.clamp(1.0 - alpha, min=self.epsilon)
        )

    def _reverse(
        self,
        target: torch.Tensor,
        sources: Mapping[str, torch.Tensor],
        step: int,
        target_name: str,
    ) -> torch.Tensor:
        source_names = tuple(sources)
        logits = torch.cat(
            [
                self.source_weights["{}_to_{}".format(name, target_name)]
                for name in source_names
            ]
        )
        weights = torch.softmax(logits, dim=0)
        aggregated = torch.stack(
            [weight * sources[name] for weight, name in zip(weights, source_names)]
        ).sum(dim=0)
        prediction = self.attention[target_name](target, aggregated)
        alpha = self.alphas[step]
        alpha_bar = self.alpha_bars[step + 1]
        denominator = torch.sqrt(torch.clamp(alpha, min=self.epsilon))
        factor = (1.0 - alpha) / torch.sqrt(
            torch.clamp(1.0 - alpha_bar, min=self.epsilon)
        )
        return (target - factor * prediction) / denominator

    def forward(self, features: Mapping[str, torch.Tensor]) -> torch.Tensor:
        current = {name: features[name] for name in self.feature_names}
        for step in range(self.steps):
            noisy = {
                name: self._diffuse(current[name], step)
                for name in self.feature_names
            }
            current = {
                target: self._reverse(
                    noisy[target],
                    {
                        source: noisy[source]
                        for source in self.feature_names
                        if source != target
                    },
                    step,
                    target,
                )
                for target in self.feature_names
            }
        return self.fusion(
            torch.cat([current[name] for name in self.feature_names], dim=-1)
        )


class _SigmoidMLP(torch.nn.Module):
    def __init__(self, input_dim: int, output_dims: Sequence[int]) -> None:
        super().__init__()
        layers = []
        for output_dim in output_dims:
            layers.extend([torch.nn.Linear(input_dim, output_dim), torch.nn.Sigmoid()])
            input_dim = output_dim
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


def _mean_pairwise_cosine(
    features: Mapping[str, torch.Tensor], names: Sequence[str]
) -> torch.Tensor:
    total = next(iter(features.values())).new_zeros(())
    pair_count = 0
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            total = total + torch.nn.functional.cosine_similarity(
                features[left_name], features[right_name], dim=-1
            ).mean()
            pair_count += 1
    return total / max(pair_count, 1)


def _hinge_cosine(
    first: torch.Tensor, second: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    similarity = torch.nn.functional.cosine_similarity(first, second, dim=-1)
    positive = torch.relu(1.0 - similarity)
    negative = torch.relu(similarity + 1.0)
    return ((1.0 - labels) * negative + labels * positive).mean()


class Diff_MSIN(_SequenceMultimodalModel):
    def __init__(self, model_config: Mapping, data_config: Mapping) -> None:
        super().__init__(model_config, data_config)
        self.synthesis_weight = float(model_config.get("lambda1", 0.1))
        self.contrastive_weight = float(model_config.get("lambda2", 0.1))
        if self.synthesis_weight < 0.0 or self.contrastive_weight < 0.0:
            raise ContractError("Diff-MSIN auxiliary weights must be non-negative")
        expert_dims = self.mlp_dims + (self.projection_dim,)
        self.modal_attention = torch.nn.ModuleDict(
            {
                name: DinAttention(self.projection_dim, [64, 32], 0.0)
                for name in self.feature_names
            }
        )
        self.specific_experts = torch.nn.ModuleDict(
            {
                name: MultiLayerPerceptron(
                    self.projection_dim,
                    expert_dims,
                    self.dropout,
                    batch_norm=self.batch_norm,
                )
                for name in self.feature_names
            }
        )
        self.shared_expert = MultiLayerPerceptron(
            self.projection_dim,
            expert_dims,
            self.dropout,
            batch_norm=self.batch_norm,
        )
        self.modal_gates = torch.nn.ModuleDict(
            {
                name: _SigmoidMLP(
                    self.projection_dim,
                    [self.projection_dim * 2, self.projection_dim],
                )
                for name in self.feature_names
            }
        )
        self.final_gate = _SigmoidMLP(
            self.projection_dim,
            [
                self.projection_dim * 2,
                self.projection_dim * (len(self.feature_names) + 1),
            ],
        )
        self.src_fusion = _StochasticReverseFusion(
            self.projection_dim,
            self.feature_names,
            int(model_config.get("T", 10)),
        )
        self.gated_entry_count = len(self.feature_names) + 1
        cross_layers = int(model_config.get("num_cross_layers", 3))
        if cross_layers <= 0:
            raise ContractError("Diff-MSIN num_cross_layers must be positive")
        self.cross_network = CrossNetwork(
            self.gated_entry_count * self.projection_dim,
            cross_layers,
        )
        self.id_attention = _CrossModalAttention(
            self.projection_dim, int(model_config.get("heads", 4))
        )
        predictor_dim = self.projection_dim * (
            len(self.feature_names) + 2 + len(self.user_feature_names)
        )
        self.dnn = MultiLayerPerceptron(
            predictor_dim,
            expert_dims,
            self.dropout,
            batch_norm=self.batch_norm,
        )
        self.output = MultiLayerPerceptron(
            self.projection_dim,
            [1],
            self.dropout,
            batch_norm=self.batch_norm,
            activation=None,
        )

    def _expert_outputs(
        self,
        target: Mapping[str, torch.Tensor],
        history: Mapping[str, torch.Tensor],
        mask: torch.Tensor,
    ):
        pooled = {
            name: self.modal_attention[name](target[name], history[name], mask)
            for name in self.feature_names
        }
        target_specific = {
            name: self.specific_experts[name](target[name])
            for name in self.feature_names
        }
        sequence_specific = {
            name: self.specific_experts[name](pooled[name])
            for name in self.feature_names
        }
        target_shared = {
            name: self.shared_expert(target[name]) for name in self.feature_names
        }
        sequence_shared = {
            name: self.shared_expert(pooled[name]) for name in self.feature_names
        }
        return target_specific, sequence_specific, target_shared, sequence_shared

    def forward_batch(self, batch: Batch) -> ModelOutput:
        target = self.project_target(batch)
        history = self.project_history(batch)
        target_specific, sequence_specific, target_shared, sequence_shared = (
            self._expert_outputs(target, history, batch.history_mask)
        )
        target_shared_mean = sum(target_shared.values()) / len(self.feature_names)
        sequence_shared_mean = sum(sequence_shared.values()) / len(self.feature_names)
        contrastive = (
            _mean_pairwise_cosine(target_specific, self.feature_names)
            - _mean_pairwise_cosine(target_shared, self.feature_names)
            + _mean_pairwise_cosine(sequence_specific, self.feature_names)
            - _mean_pairwise_cosine(sequence_shared, self.feature_names)
        ) / 2.0
        target_gates = {
            name: self.modal_gates[name](target_specific[name])
            for name in self.feature_names
        }
        sequence_gates = {
            name: self.modal_gates[name](sequence_specific[name])
            for name in self.feature_names
        }
        target_mixed = {
            name: target_gates[name] * target_specific[name]
            + (1.0 - target_gates[name]) * target_shared_mean
            for name in self.feature_names
        }
        sequence_mixed = {
            name: sequence_gates[name] * sequence_specific[name]
            + (1.0 - sequence_gates[name]) * sequence_shared_mean
            for name in self.feature_names
        }
        target_synthesis = self.src_fusion(target_mixed)
        sequence_synthesis = self.src_fusion(sequence_mixed)
        synthesis_loss = _hinge_cosine(
            target_synthesis, sequence_synthesis, batch.labels
        )

        gate_values = self.final_gate(target_mixed["id"]).view(
            batch.batch_size, self.gated_entry_count, self.projection_dim
        )
        gate_names = tuple(
            name for name in self.feature_names if name != "id"
        ) + ("share", "synthesis")
        gates: Dict[str, torch.Tensor] = {
            name: gate_values[:, index]
            for index, name in enumerate(gate_names)
        }
        gated_entries = [
            gates[name] * sequence_mixed[name]
            for name in self.feature_names
            if name != "id"
        ]
        gated_entries.extend(
            [
                gates["share"] * sequence_shared_mean,
                gates["synthesis"] * sequence_synthesis,
            ]
        )
        flattened = torch.stack(gated_entries, dim=1).reshape(batch.batch_size, -1)
        crossed = self.cross_network(flattened)
        cross_mean = crossed.view(
            batch.batch_size, self.gated_entry_count, self.projection_dim
        ).mean(dim=1)
        attended_id = self.id_attention(sequence_mixed["id"], cross_mean)
        user = self.project_user(batch)
        logits = self.output(
            self.dnn(torch.cat([crossed, attended_id, user], dim=-1))
        )
        return ModelOutput(
            logits,
            auxiliary_losses={
                "diff_msin_synthesis": synthesis_loss * self.synthesis_weight,
                "diff_msin_contrastive": contrastive * self.contrastive_weight,
            },
            representations={
                "sequence_synthesis": sequence_synthesis,
                "target_synthesis": target_synthesis,
            },
        )


__all__ = ["Diff_MSIN", "EM3"]

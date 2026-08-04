"""EM3 multimodal sequence recommendation model."""

import math
from typing import Mapping

import torch

from mmctr.core import Batch, ContractError, ModelOutput
from mmctr.models.common.sequence import _SequenceMultimodalModel
from mmctr.models.common.layers import DinAttention, MultiLayerPerceptron
from mmctr.models.common.components import apply_sequence_mask


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
        return values.view(batch_size, length, self.heads, self.head_dimension).permute(0, 2, 1, 3)

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
    """Compress modality tokens with learned queries and align content to IDs.

    The FQ-Former maps ``[batch, modalities, projection_dim]`` to a flattened
    query representation; its symmetric in-batch contrastive loss treats the
    matching content/ID row as the positive pair.
    """

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
            fusion_dim + self.projection_dim * len(self.user_feature_names) + self.projection_dim
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

    def _content_item_loss(self, content: torch.Tensor, item_id: torch.Tensor) -> torch.Tensor:
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
        target_tokens = torch.stack([target_features[name] for name in self.feature_names], dim=1)
        history_tokens = torch.stack([history_features[name] for name in self.feature_names], dim=2)
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
        history_interest = self.attention(history_fusion, batch.history_mask, target_fusion)
        content = self.content_map(target_fusion)
        cic_loss = self._content_item_loss(content, target_features["id"])
        user = self.project_user(batch)
        logits = self.output(self.dnn(torch.cat([user, history_interest, content], dim=-1)))
        return ModelOutput(
            logits,
            auxiliary_losses={"em3_content_item_contrastive": cic_loss * self.cic_weight},
            representations={
                "content": content,
                "history_interest": history_interest,
                "target_queries": target_fusion,
            },
        )


__all__ = ["EM3"]

"""M3SRec multimodal sequential recommendation model."""

from typing import Mapping, Sequence

import torch

from mmctr.core import Batch, ContractError, ModelOutput
from mmctr.models.common.sequence import _SequenceMultimodalModel
from mmctr.models.common.sequence_utils import last_valid_token
from mmctr.models.common.layers import MultiLayerPerceptron


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
                last_valid_token(chunk, mask)
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


__all__ = ["M3SRec"]

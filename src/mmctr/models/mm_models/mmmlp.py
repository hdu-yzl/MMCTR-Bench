"""Mask-aware multimodal MLP-Mixer recommendation model."""

from typing import Mapping

import torch

from mmctr.core import Batch, ContractError, ModelOutput
from mmctr.models.common.sequence import _SequenceMultimodalModel
from mmctr.models.common.sequence_utils import last_valid_token
from mmctr.models.common.layers import MultiLayerPerceptron
from mmctr.models.common.components import apply_sequence_mask


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
        history_vector = last_valid_token(mixed_history, batch.history_mask)
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


__all__ = ["MMMLP"]

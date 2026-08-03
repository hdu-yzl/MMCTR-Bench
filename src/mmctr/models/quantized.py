"""Canonical quantization-aware CTR models with injected pretrained artifacts."""

from typing import Dict, Mapping

import torch

from mmctr.core import Batch, ContractError, ModelOutput
from mmctr.models.base import BaseSeqModel
from mmctr.models.baselines.layers import (
    CrossNetwork,
    FeatureEmbedding,
    MultiLayerPerceptron,
)
from mmctr.models.components import (
    apply_feature_mask,
    apply_sequence_mask,
    feature_presence,
)
from mmctr.models.sequence import _SequenceMultimodalModel
from mmctr.quantization import PSRQPretrainer, ResidualQuantizer


def _dataset_config(model_config: Mapping, data_config: Mapping) -> Dict:
    selected = model_config.get(str(data_config.get("name", "")).lower(), model_config)
    if not isinstance(selected, Mapping):
        raise ContractError("dataset-specific quantized model config must be a mapping")
    return dict(selected)


class _QuantizedSequenceModel(_SequenceMultimodalModel):
    def __init__(self, model_config: Mapping, data_config: Mapping) -> None:
        super().__init__(model_config, data_config)
        self.quantized_modalities = tuple(
            name for name in self.feature_names if name != "id"
        )
        if not self.quantized_modalities:
            raise ContractError("quantized CTR models require a non-ID modality")
        self.n_levels = int(model_config.get("n_levels", 3))
        self.codebook_size = int(model_config.get("codebook_size", 256))
        if self.n_levels <= 0 or self.codebook_size <= 0:
            raise ContractError("quantized CTR levels and codebook size must be positive")
        self.code_embeddings = torch.nn.ModuleDict(
            {
                name: torch.nn.ModuleList(
                    [
                        FeatureEmbedding(self.codebook_size, self.latent_dim)
                        for _ in range(self.n_levels)
                    ]
                )
                for name in self.quantized_modalities
            }
        )
        for name in self.quantized_modalities:
            self.projectors.replace(name, self.n_levels * self.latent_dim)

    def _embed_codes(self, name: str, codes: torch.Tensor) -> torch.Tensor:
        if codes.dtype != torch.long or codes.shape[-1] != self.n_levels:
            raise ContractError("quantizer returned incompatible codes for {!r}".format(name))
        return torch.cat(
            [
                self.code_embeddings[name][level](codes[..., level])
                for level in range(self.n_levels)
            ],
            dim=-1,
        )

    @staticmethod
    def _present(values: torch.Tensor) -> torch.Tensor:
        return feature_presence(values)


class QARM(_QuantizedSequenceModel):
    """Residual-codebook recommendation model with pure injected RQ dependencies."""

    def __init__(
        self,
        model_config: Mapping,
        data_config: Mapping,
        quantizers: Mapping[str, ResidualQuantizer],
    ) -> None:
        config = _dataset_config(model_config, data_config)
        super().__init__(config, data_config)
        if set(quantizers) != set(self.quantized_modalities):
            raise ContractError("QARM quantizers must match non-ID modalities exactly")
        dimensions = dict(data_config.get("mm_seq_dims", data_config.get("mm_dims", {})))
        checked: Dict[str, ResidualQuantizer] = {}
        for name in self.quantized_modalities:
            quantizer = quantizers[name]
            if not isinstance(quantizer, ResidualQuantizer) or not quantizer.is_fitted:
                raise ContractError("QARM requires fitted ResidualQuantizer instances")
            expected = (self.n_levels, self.codebook_size, int(dimensions[name]))
            actual = (
                quantizer.n_levels,
                quantizer.codebook_size,
                quantizer.dimension,
            )
            if actual != expected:
                raise ContractError(
                    "QARM {!r} quantizer {} does not match {}".format(name, actual, expected)
                )
            checked[name] = quantizer
        self.quantizers = torch.nn.ModuleDict(checked)
        self.cross_layers = int(config.get("cross_num", 3))
        if self.cross_layers <= 0:
            raise ContractError("QARM cross_num must be positive")
        predictor_dim = self.projection_dim * (
            2 * len(self.feature_names) + len(self.user_feature_names)
        )
        self.cross = CrossNetwork(predictor_dim, self.cross_layers)
        self.dnn = MultiLayerPerceptron(
            predictor_dim,
            self.mlp_dims,
            self.dropout,
            batch_norm=self.batch_norm,
        )
        self.combination = MultiLayerPerceptron(
            predictor_dim + self.mlp_dims[-1],
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

    def _encode(self, name: str, values: torch.Tensor) -> torch.Tensor:
        denominator = values.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        codes, _ = self.quantizers[name].encode(values / denominator)
        embedded = self._embed_codes(name, codes)
        return embedded * self._present(values).unsqueeze(-1)

    def _project_quantized_target(self, batch: Batch):
        encoded = {
            "id": self.embedding(self._target_feature(batch, "id")).squeeze(1)
        }
        presence = {}
        for name in self.quantized_modalities:
            values = self._target_feature(batch, name)
            encoded[name] = self._encode(name, values)
            presence[name] = self._present(values)
        return self.projectors(encoded, presence)

    def _project_quantized_history(self, batch: Batch):
        encoded = {
            "id": self.embedding(batch.history_features["id"])
        }
        presence = {"id": batch.history_mask}
        for name in self.quantized_modalities:
            values = batch.history_features[name]
            encoded[name] = self._encode(name, values)
            presence[name] = self._present(values) & batch.history_mask
        return self.projectors(encoded, presence)

    def forward_batch(self, batch: Batch) -> ModelOutput:
        target_fields = self._project_quantized_target(batch)
        history_fields = self._project_quantized_history(batch)
        target = torch.cat([target_fields[name] for name in self.feature_names], dim=-1)
        history = torch.cat(
            [
                self.masked_pool(history_fields[name], batch.history_mask)
                for name in self.feature_names
            ],
            dim=-1,
        )
        user = self.project_user(batch)
        inputs = torch.cat([user, target, history], dim=-1)
        crossed = self.cross(inputs)
        deep = self.dnn(inputs)
        hidden = self.combination(torch.cat([crossed, deep], dim=-1))
        return ModelOutput(
            self.output(hidden),
            representations={"target_quantized": target, "history_quantized": history},
        )


class _MaskedCrossAttentionPool(torch.nn.Module):
    def __init__(self, dimension: int, dropout: float) -> None:
        super().__init__()
        self.scorer = torch.nn.Sequential(
            torch.nn.Linear(2 * dimension, dimension),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dimension, 1),
        )

    def forward(
        self,
        query: torch.Tensor,
        sequence: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        expanded = query.unsqueeze(1).expand(-1, sequence.shape[1], -1)
        scores = self.scorer(torch.cat([sequence, expanded], dim=-1)).squeeze(-1)
        weights = BaseSeqModel.masked_softmax(scores, mask)
        return torch.sum(weights.unsqueeze(-1) * sequence, dim=1)


class MCCA(_QuantizedSequenceModel):
    """PSRQ code alignment model with mask-aware cross-attention histories."""

    def __init__(
        self,
        model_config: Mapping,
        data_config: Mapping,
        quantizer: PSRQPretrainer,
    ) -> None:
        config = _dataset_config(model_config, data_config)
        super().__init__(config, data_config)
        if not isinstance(quantizer, PSRQPretrainer) or not quantizer.is_initialized:
            raise ContractError("MCCA requires an initialized PSRQPretrainer")
        dimensions = dict(data_config.get("mm_seq_dims", data_config.get("mm_dims", {})))
        expected_dimensions = {
            name: int(dimensions[name]) for name in self.quantized_modalities
        }
        expected_structure = (
            self.quantized_modalities,
            expected_dimensions,
            self.n_levels,
            self.codebook_size,
            int(config.get("projection_dim", 128)),
            tuple(int(value) for value in config.get("psrq_dims", (256, 128))),
            float(config.get("dropout", 0.0)),
            bool(config.get("batch_norm", True)),
        )
        actual_structure = (
            quantizer.modalities,
            quantizer.modality_dimensions,
            quantizer.n_levels,
            quantizer.codebook_size,
            quantizer.embedding_dimension,
            quantizer.hidden_dimensions,
            quantizer.dropout,
            quantizer.batch_norm,
        )
        if actual_structure != expected_structure:
            raise ContractError(
                "MCCA PSRQ structure {} does not match {}".format(
                    actual_structure, expected_structure
                )
            )
        self.quantizer = quantizer
        self.quantizer.requires_grad_(False)
        self.quantizer.eval()
        self.joint_embeddings = torch.nn.ModuleList(
            [
                FeatureEmbedding(self.codebook_size, self.latent_dim)
                for _ in range(self.n_levels)
            ]
        )
        self.joint_projector = torch.nn.Linear(
            self.n_levels * self.latent_dim, self.projection_dim
        )
        self.history_attention = torch.nn.ModuleDict(
            {
                name: _MaskedCrossAttentionPool(self.projection_dim, self.dropout)
                for name in self.feature_names
            }
        )
        predictor_dim = self.projection_dim * (
            len(self.feature_names) + len(self.user_feature_names) + 1
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

    def train(self, mode: bool = True):
        super().train(mode)
        self.quantizer.eval()
        return self

    def _raw_target(self, batch: Batch):
        return {
            name: self._target_feature(batch, name)
            for name in self.quantized_modalities
        }

    def _raw_history(self, batch: Batch):
        return {name: batch.history_features[name] for name in self.quantized_modalities}

    def forward_batch(self, batch: Batch) -> ModelOutput:
        raw_target = self._raw_target(batch)
        raw_history = self._raw_history(batch)
        modality_codes, joint_codes = self.quantizer.encode_items(raw_target)
        history_codes = self.quantizer.encode_history(raw_history)

        target = {
            "id": self.projectors["id"](
                self.embedding(self._target_feature(batch, "id")).squeeze(1)
            )
        }
        history = {
            "id": self.projectors["id"](self.embedding(batch.history_features["id"]))
        }
        for name in self.quantized_modalities:
            target_embedding = self._embed_codes(name, modality_codes[name])
            target_embedding = target_embedding * self._present(raw_target[name]).unsqueeze(-1)
            history_embedding = self._embed_codes(name, history_codes[name])
            history_embedding = history_embedding * self._present(raw_history[name]).unsqueeze(-1)
            target[name] = self.projectors[name](target_embedding)
            history[name] = self.projectors[name](history_embedding)
            target[name] = target[name] * self._present(raw_target[name]).unsqueeze(-1)
            history[name] = history[name] * self._present(raw_history[name]).unsqueeze(-1)
        history = {
            name: apply_sequence_mask(values, batch.history_mask)
            for name, values in history.items()
        }

        joint = torch.cat(
            [self.joint_embeddings[level](joint_codes[:, level]) for level in range(self.n_levels)],
            dim=-1,
        )
        joint_present = torch.stack(
            [self._present(raw_target[name]) for name in self.quantized_modalities], dim=-1
        ).any(dim=-1)
        joint = self.joint_projector(apply_feature_mask(joint, joint_present))
        joint = apply_feature_mask(joint, joint_present)

        pooled = {}
        for name in self.feature_names:
            query = target["id"] if name == "id" else joint
            pooled[name] = self.history_attention[name](
                query, history[name], batch.history_mask
            )
        history_vector = torch.cat(
            [pooled[name] for name in self.feature_names], dim=-1
        )
        user = self.project_user(batch)
        hidden = self.dnn(torch.cat([user, history_vector, joint], dim=-1))
        return ModelOutput(
            self.output(hidden),
            representations={
                "joint_quantized": joint,
                "history_quantized": history_vector,
            },
        )


__all__ = ["MCCA", "QARM"]

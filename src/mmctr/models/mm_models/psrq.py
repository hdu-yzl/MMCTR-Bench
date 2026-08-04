"""PSRQ benchmark consumer model implemented by MCCA."""

from typing import Mapping

import torch

from mmctr.core import Batch, ContractError, ModelOutput
from mmctr.models.common.quantized import _QuantizedSequenceModel, _dataset_config
from mmctr.models.common.base import BaseSeqModel
from mmctr.models.common.layers import FeatureEmbedding, MultiLayerPerceptron
from mmctr.models.common.components import apply_feature_mask, apply_sequence_mask
from mmctr.quantization import PSRQPretrainer


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
    """PSRQ benchmark consumer with mask-aware cross-attention histories.

    The injected PSRQ module is frozen permanently, including when the parent
    model enters training mode. Modality codes encode each branch separately;
    joint codes provide the shared target query for non-ID history modalities.
    """

    def __init__(
        self,
        model_config: Mapping,
        data_config: Mapping,
        quantizer: PSRQPretrainer,
    ) -> None:
        config = _dataset_config(model_config, data_config)
        super().__init__(config, data_config)
        if not isinstance(quantizer, PSRQPretrainer) or not quantizer.is_initialized:
            raise ContractError("PSRQ benchmark consumer requires an initialized PSRQPretrainer")
        dimensions = dict(data_config.get("mm_seq_dims", data_config.get("mm_dims", {})))
        expected_dimensions = {name: int(dimensions[name]) for name in self.quantized_modalities}
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
                "PSRQ benchmark consumer structure {} does not match {}".format(
                    actual_structure, expected_structure
                )
            )
        self.quantizer = quantizer
        self.quantizer.requires_grad_(False)
        self.quantizer.eval()
        self.joint_embeddings = torch.nn.ModuleList(
            [FeatureEmbedding(self.codebook_size, self.latent_dim) for _ in range(self.n_levels)]
        )
        self.joint_projector = torch.nn.Linear(self.n_levels * self.latent_dim, self.projection_dim)
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
        """Change model mode while keeping the pretrained quantizer frozen in eval."""

        super().train(mode)
        self.quantizer.eval()
        return self

    def _raw_target(self, batch: Batch):
        return {name: self._target_feature(batch, name) for name in self.quantized_modalities}

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
        history = {"id": self.projectors["id"](self.embedding(batch.history_features["id"]))}
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
            pooled[name] = self.history_attention[name](query, history[name], batch.history_mask)
        history_vector = torch.cat([pooled[name] for name in self.feature_names], dim=-1)
        user = self.project_user(batch)
        hidden = self.dnn(torch.cat([user, history_vector, joint], dim=-1))
        return ModelOutput(
            self.output(hidden),
            representations={
                "joint_quantized": joint,
                "history_quantized": history_vector,
            },
        )


__all__ = ["MCCA"]

"""Quantization-aware residual-codebook recommendation model."""

from typing import Dict, Mapping

import torch

from mmctr.core import Batch, ContractError, ModelOutput
from mmctr.models.common.quantized import _QuantizedSequenceModel, _dataset_config
from mmctr.models.common.layers import CrossNetwork, MultiLayerPerceptron
from mmctr.quantization import ResidualQuantizer


class QARM(_QuantizedSequenceModel):
    """Residual-codebook recommendation model with injected fitted quantizers.

    One quantizer is required per non-ID modality. Codes have shape
    ``[..., n_levels]`` and are embedded level-wise before projection; the
    injected quantizers remain registered modules so checkpoints are complete.
    """

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
        encoded = {"id": self.embedding(self._target_feature(batch, "id")).squeeze(1)}
        presence = {}
        for name in self.quantized_modalities:
            values = self._target_feature(batch, name)
            encoded[name] = self._encode(name, values)
            presence[name] = self._present(values)
        return self.projectors(encoded, presence)

    def _project_quantized_history(self, batch: Batch):
        encoded = {"id": self.embedding(batch.history_features["id"])}
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


__all__ = ["QARM"]

"""Shared implementation for quantization-aware sequence models."""

from typing import Dict, Mapping

import torch

from mmctr.core import ContractError
from mmctr.models.common.sequence import _SequenceMultimodalModel
from mmctr.models.common.layers import FeatureEmbedding
from mmctr.models.common.components import feature_presence


def _dataset_config(model_config: Mapping, data_config: Mapping) -> Dict:
    selected = model_config.get(str(data_config.get("name", "")).lower(), model_config)
    if not isinstance(selected, Mapping):
        raise ContractError("dataset-specific quantized model config must be a mapping")
    return dict(selected)


class _QuantizedSequenceModel(_SequenceMultimodalModel):
    def __init__(self, model_config: Mapping, data_config: Mapping) -> None:
        super().__init__(model_config, data_config)
        self.quantized_modalities = tuple(name for name in self.feature_names if name != "id")
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
        """Map ``[..., n_levels]`` integer codes to concatenated level embeddings."""

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


__all__ = ["_QuantizedSequenceModel", "_dataset_config"]

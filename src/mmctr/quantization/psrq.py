"""Progressive semantic residual quantization pretraining components."""

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple, cast

import numpy as np
import torch
import torch.nn.functional as functional

from mmctr.core import ContractError
from mmctr.models.baselines.layers import MultiLayerPerceptron

from .artifacts import (
    PathLike,
    QuantizationArtifactError,
    load_quantization_artifact,
    save_quantization_artifact,
)


@dataclass(frozen=True)
class PSRQOutput:
    """Named pretraining output, deliberately separate from CTR ``ModelOutput``."""

    losses: Mapping[str, torch.Tensor]
    codes: Mapping[str, torch.Tensor]
    joint_codes: torch.Tensor
    reconstructions: Mapping[str, torch.Tensor]
    joint_reconstruction: torch.Tensor

    def total_loss(self) -> torch.Tensor:
        if not self.losses:
            raise ContractError("PSRQ output contains no losses")
        return torch.stack(tuple(self.losses.values())).sum()


class _VectorQuantizer(torch.nn.Module):
    def __init__(self, codebook_size: int, dimension: int, commitment: float) -> None:
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.dimension = int(dimension)
        self.commitment = float(commitment)
        self.embedding = torch.nn.Embedding(self.codebook_size, self.dimension)
        self.register_buffer("initted", torch.tensor(False, dtype=torch.bool))
        self.embedding.weight.data.zero_()

    def _initialize(self, values: torch.Tensor) -> None:
        if values.shape[0] < self.codebook_size:
            raise ContractError("PSRQ initialization requires at least codebook_size samples")
        from sklearn.cluster import KMeans

        source = values.detach().cpu().numpy()
        estimator = KMeans(
            n_clusters=self.codebook_size,
            max_iter=10,
            n_init=10,
            verbose=0,
            random_state=2025,
        )
        estimator.fit(source)
        centers = torch.from_numpy(np.asarray(estimator.cluster_centers_)).to(
            device=values.device, dtype=values.dtype
        )
        self.embedding.weight.data.copy_(centers)
        self.initted.fill_(True)

    def forward(self, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if values.shape[-1] != self.dimension:
            raise ContractError("PSRQ vector-quantizer input dimension mismatch")
        flat = values.reshape(-1, self.dimension)
        if not bool(self.initted.item()):
            if not self.training:
                raise ContractError("PSRQ codebook is not initialized")
            self._initialize(flat)
        codebook = self.embedding.weight
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            + codebook.pow(2).sum(dim=1).unsqueeze(0)
            - 2.0 * flat.matmul(codebook.t())
        )
        indices = distances.argmin(dim=-1)
        quantized = self.embedding(indices).reshape(values.shape)
        commitment_loss = functional.mse_loss(quantized.detach(), values)
        codebook_loss = functional.mse_loss(quantized, values.detach())
        loss = codebook_loss + self.commitment * commitment_loss
        straight_through = values + (quantized - values).detach()
        return straight_through, loss, indices.reshape(values.shape[:-1])


class _ProgressiveResidualQuantizer(torch.nn.Module):
    def __init__(
        self,
        n_levels: int,
        codebook_size: int,
        dimension: int,
        commitment: float,
    ) -> None:
        super().__init__()
        self.n_levels = int(n_levels)
        self.dimension = int(dimension)
        dimensions = [self.dimension] + [2 * self.dimension] * (self.n_levels - 1)
        self.levels = torch.nn.ModuleList(
            [
                _VectorQuantizer(codebook_size, level_dimension, commitment)
                for level_dimension in dimensions
            ]
        )
        self.projectors = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(2 * self.dimension, self.dimension),
                    torch.nn.LayerNorm(self.dimension),
                )
                for _ in range(self.n_levels - 1)
            ]
        )

    @property
    def is_initialized(self) -> bool:
        return all(bool(level.initted.item()) for level in self.levels)

    def forward(self, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = values
        prefix = torch.zeros_like(values)
        reconstruction = torch.zeros_like(values)
        losses = []
        codes = []
        for level_index, quantizer in enumerate(self.levels):
            quantizer_input = residual
            if level_index:
                quantizer_input = torch.cat([residual, prefix], dim=-1)
            contribution, loss, indices = quantizer(quantizer_input)
            if level_index:
                contribution = self.projectors[level_index - 1](contribution)
            residual = residual - contribution
            prefix = values - residual
            reconstruction = reconstruction + contribution
            losses.append(loss)
            codes.append(indices)
        return reconstruction, torch.stack(losses).mean(), torch.stack(codes, dim=-1)


class _PSRQAutoEncoder(torch.nn.Module):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimensions: Tuple[int, ...],
        embedding_dimension: int,
        n_levels: int,
        codebook_size: int,
        dropout: float,
        batch_norm: bool,
        commitment: float,
        quantization_weight: float,
    ) -> None:
        super().__init__()
        encoder_dimensions = hidden_dimensions + (embedding_dimension,)
        self.encoder = MultiLayerPerceptron(
            input_dimension,
            encoder_dimensions,
            dropout,
            batch_norm=batch_norm,
        )
        self.quantizer = _ProgressiveResidualQuantizer(
            n_levels, codebook_size, embedding_dimension, commitment
        )
        decoder_dimensions = tuple(reversed(encoder_dimensions[:-1])) + (input_dimension,)
        self.decoder = MultiLayerPerceptron(
            embedding_dimension,
            decoder_dimensions,
            dropout,
            batch_norm=batch_norm,
        )
        self.quantization_weight = float(quantization_weight)

    def forward(
        self, values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encoder(values)
        quantized, quantization_loss, codes = self.quantizer(encoded)
        reconstruction = self.decoder(quantized)
        reconstruction_loss = functional.mse_loss(reconstruction, values)
        loss = reconstruction_loss + self.quantization_weight * quantization_loss
        return reconstruction, loss, codes

    @torch.no_grad()
    def encode(self, values: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(values)
        _, _, codes = self.quantizer(encoded)
        return codes


class PSRQPretrainer(torch.nn.Module):
    """Multimodal PSRQ autoencoder and code provider without optimizer or path state."""

    artifact_kind = "progressive-semantic-residual-quantizer"

    def __init__(self, model_config: Mapping, data_config: Mapping) -> None:
        super().__init__()
        self.dataset_name = str(data_config.get("name", "unknown")).lower()
        self.modalities = tuple(
            name for name in data_config.get("use_mm_features", ()) if name != "id"
        )
        dimensions = dict(data_config.get("mm_dims", {}))
        if not self.modalities:
            raise ContractError("PSRQ requires at least one non-ID modality")
        missing = [name for name in self.modalities if name not in dimensions]
        if missing:
            raise ContractError("PSRQ modality dimensions are missing: {}".format(missing))
        self.modality_dimensions = {
            name: int(dimensions[name]) for name in self.modalities
        }
        if any(value <= 0 for value in self.modality_dimensions.values()):
            raise ContractError("PSRQ modality dimensions must be positive")

        self.embedding_dimension = int(model_config.get("projection_dim", 128))
        self.hidden_dimensions = tuple(
            int(value) for value in model_config.get("psrq_dims", (256, 128))
        )
        self.n_levels = int(model_config.get("n_levels", 3))
        self.codebook_size = int(model_config.get("codebook_size", 256))
        self.dropout = float(model_config.get("dropout", 0.0))
        self.batch_norm = bool(model_config.get("batch_norm", True))
        self.commitment = float(model_config.get("mu", 0.25))
        self.quantization_weight = float(model_config.get("quant_loss_weight", 1.0))
        if self.embedding_dimension <= 0 or self.n_levels <= 0 or self.codebook_size <= 0:
            raise ContractError("PSRQ dimensions, levels, and codebook size must be positive")
        if not self.hidden_dimensions or any(value <= 0 for value in self.hidden_dimensions):
            raise ContractError("PSRQ hidden dimensions must contain positive values")
        if not 0.0 <= self.dropout < 1.0 or self.commitment < 0:
            raise ContractError("PSRQ dropout or commitment configuration is invalid")
        if self.quantization_weight < 0:
            raise ContractError("PSRQ quantization weight cannot be negative")

        def build_autoencoder(input_dimension: int) -> _PSRQAutoEncoder:
            return _PSRQAutoEncoder(
                input_dimension=input_dimension,
                hidden_dimensions=self.hidden_dimensions,
                embedding_dimension=self.embedding_dimension,
                n_levels=self.n_levels,
                codebook_size=self.codebook_size,
                dropout=self.dropout,
                batch_norm=self.batch_norm,
                commitment=self.commitment,
                quantization_weight=self.quantization_weight,
            )

        self.modality_models = torch.nn.ModuleDict(
            {
                name: build_autoencoder(self.modality_dimensions[name])
                for name in self.modalities
            }
        )
        self.joint_model = build_autoencoder(sum(self.modality_dimensions.values()))

    def _modality_model(self, name: str) -> _PSRQAutoEncoder:
        return cast(_PSRQAutoEncoder, self.modality_models[name])

    @property
    def is_initialized(self) -> bool:
        models = [self._modality_model(name) for name in self.modalities]
        models.append(self.joint_model)
        return all(model.quantizer.is_initialized for model in models)

    def _validate_features(self, features: Mapping[str, torch.Tensor], rank: int) -> None:
        for name in self.modalities:
            if name not in features:
                raise ContractError("PSRQ feature {!r} is missing".format(name))
            values = features[name]
            if values.ndim != rank or values.shape[-1] != self.modality_dimensions[name]:
                raise ContractError("PSRQ feature {!r} has an incompatible shape".format(name))
            if not torch.is_floating_point(values):
                raise ContractError("PSRQ features must use floating dtypes")

    def forward(self, features: Mapping[str, torch.Tensor]) -> PSRQOutput:
        self._validate_features(features, 2)
        losses: Dict[str, torch.Tensor] = {}
        codes: Dict[str, torch.Tensor] = {}
        reconstructions: Dict[str, torch.Tensor] = {}
        for name in self.modalities:
            reconstruction, loss, modality_codes = self._modality_model(name)(features[name])
            losses[name] = loss
            codes[name] = modality_codes
            reconstructions[name] = reconstruction
        joint = torch.cat([features[name] for name in self.modalities], dim=-1)
        joint_reconstruction, joint_loss, joint_codes = self.joint_model(joint)
        losses["joint"] = joint_loss
        return PSRQOutput(
            losses=losses,
            codes=codes,
            joint_codes=joint_codes,
            reconstructions=reconstructions,
            joint_reconstruction=joint_reconstruction,
        )

    @torch.no_grad()
    def encode_items(
        self, features: Mapping[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        self._validate_features(features, 2)
        codes = {
            name: self._modality_model(name).encode(features[name])
            for name in self.modalities
        }
        joint = torch.cat([features[name] for name in self.modalities], dim=-1)
        return codes, self.joint_model.encode(joint)

    @torch.no_grad()
    def encode_history(
        self, features: Mapping[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        self._validate_features(features, 3)
        first = features[self.modalities[0]]
        batch_size, sequence_length = first.shape[:2]
        result = {}
        for name in self.modalities:
            values = features[name]
            if values.shape[:2] != (batch_size, sequence_length):
                raise ContractError("PSRQ history modalities must share [B, L]")
            flattened = values.reshape(-1, values.shape[-1])
            result[name] = self._modality_model(name).encode(flattened).reshape(
                batch_size, sequence_length, self.n_levels
            )
        return result

    def save(self, path: PathLike, metadata: Optional[Mapping[str, object]] = None):
        if not self.is_initialized:
            raise ContractError("PSRQ codebooks are not initialized")
        artifact_metadata: Dict[str, object] = {
            "dataset": self.dataset_name,
            "modalities": list(self.modalities),
            "modality_dimensions": dict(self.modality_dimensions),
            "embedding_dimension": self.embedding_dimension,
            "hidden_dimensions": list(self.hidden_dimensions),
            "n_levels": self.n_levels,
            "codebook_size": self.codebook_size,
            "dropout": self.dropout,
            "batch_norm": self.batch_norm,
            "commitment": self.commitment,
            "quantization_weight": self.quantization_weight,
        }
        for name, value in dict(metadata or {}).items():
            if name in artifact_metadata and artifact_metadata[name] != value:
                raise ContractError("PSRQ artifact metadata cannot override {!r}".format(name))
            artifact_metadata[name] = value
        arrays = {
            "state." + name: tensor.detach().cpu().numpy()
            for name, tensor in self.state_dict().items()
        }
        return save_quantization_artifact(path, self.artifact_kind, artifact_metadata, arrays)

    @classmethod
    def from_artifact(cls, path: PathLike) -> "PSRQPretrainer":
        metadata, arrays = load_quantization_artifact(path, cls.artifact_kind)
        required = {
            "modalities",
            "modality_dimensions",
            "embedding_dimension",
            "hidden_dimensions",
            "n_levels",
            "codebook_size",
        }
        if not required.issubset(metadata):
            raise QuantizationArtifactError("PSRQ artifact metadata is incomplete")
        model_config = {
            "projection_dim": metadata["embedding_dimension"],
            "psrq_dims": metadata["hidden_dimensions"],
            "n_levels": metadata["n_levels"],
            "codebook_size": metadata["codebook_size"],
            "dropout": metadata.get("dropout", 0.0),
            "batch_norm": metadata.get("batch_norm", True),
            "mu": metadata.get("commitment", 0.25),
            "quant_loss_weight": metadata.get("quantization_weight", 1.0),
        }
        data_config = {
            "name": metadata.get("dataset", "unknown"),
            "use_mm_features": ["id"] + list(metadata["modalities"]),
            "mm_dims": dict(metadata["modality_dimensions"]),
        }
        model = cls(model_config, data_config)
        if not arrays or any(not name.startswith("state.") for name in arrays):
            raise QuantizationArtifactError("PSRQ artifact contains invalid state arrays")
        state = {
            name[len("state.") :]: torch.from_numpy(value)
            for name, value in arrays.items()
            if name.startswith("state.")
        }
        incompatible = model.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise QuantizationArtifactError("PSRQ artifact state is incompatible")
        if not model.is_initialized:
            raise QuantizationArtifactError("PSRQ artifact contains uninitialized codebooks")
        return model


__all__ = ["PSRQOutput", "PSRQPretrainer"]

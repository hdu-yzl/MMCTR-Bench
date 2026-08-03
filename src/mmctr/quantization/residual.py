"""Residual vector quantization independent from CTR model lifecycle."""

from typing import Dict, Mapping, Optional, Tuple, Union

import numpy as np
import torch

from mmctr.core import ContractError

from .artifacts import (
    PathLike,
    QuantizationArtifactError,
    load_quantization_artifact,
    save_quantization_artifact,
)


def _config_int(
    explicit: Optional[int],
    values: Mapping[str, object],
    name: str,
    default: int,
) -> int:
    value: object = explicit if explicit is not None else values.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ContractError("RQ {!r} must be an integer".format(name))
    try:
        return int(value)
    except ValueError as error:
        raise ContractError("RQ {!r} must be an integer".format(name)) from error


def _config_float(
    explicit: Optional[float],
    values: Mapping[str, object],
    name: str,
    default: float,
) -> float:
    value: object = explicit if explicit is not None else values.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ContractError("RQ {!r} must be numeric".format(name))
    try:
        return float(value)
    except ValueError as error:
        raise ContractError("RQ {!r} must be numeric".format(name)) from error


class ResidualQuantizer(torch.nn.Module):
    """K-means residual quantizer with device-aware inference buffers."""

    artifact_kind = "residual-quantizer"

    def __init__(
        self,
        config: Optional[Mapping[str, object]] = None,
        *,
        n_levels: Optional[int] = None,
        codebook_size: Optional[int] = None,
        dimension: Optional[int] = None,
        random_state: Optional[int] = None,
        n_init: Optional[int] = None,
        max_iter: Optional[int] = None,
        tolerance: Optional[float] = None,
    ) -> None:
        super().__init__()
        values = dict(config or {})
        self.n_levels = _config_int(n_levels, values, "n_levels", 3)
        self.codebook_size = _config_int(codebook_size, values, "codebook_size", 1024)
        self.dimension = _config_int(dimension, values, "dimension", 128)
        self.random_state = _config_int(random_state, values, "random_state", 42)
        self.n_init = _config_int(n_init, values, "n_init", 5)
        self.max_iter = _config_int(max_iter, values, "max_iter", 20)
        self.tolerance = _config_float(tolerance, values, "tol", 1e-4)
        if self.n_levels <= 0 or self.codebook_size <= 0 or self.dimension <= 0:
            raise ContractError("RQ levels, codebook size, and dimension must be positive")
        if self.n_init <= 0 or self.max_iter <= 0 or self.tolerance < 0:
            raise ContractError("RQ fitting parameters are invalid")
        self.artifact_metadata: Dict[str, object] = {}
        self.register_buffer("codebooks", torch.empty(0, dtype=torch.float32))

    @property
    def is_fitted(self) -> bool:
        return self.codebooks.ndim == 3 and self.codebooks.shape[0] == self.n_levels

    def _validate_vectors(self, vectors: torch.Tensor) -> None:
        if vectors.ndim not in (2, 3):
            raise ContractError("RQ vectors must have shape [N, D] or [B, L, D]")
        if vectors.shape[-1] != self.dimension:
            raise ContractError(
                "RQ vector dimension {} does not match {}".format(vectors.shape[-1], self.dimension)
            )
        if not torch.is_floating_point(vectors):
            raise ContractError("RQ vectors must use a floating dtype")

    def set_codebooks(self, codebooks: Union[np.ndarray, torch.Tensor]) -> None:
        tensor = torch.as_tensor(codebooks, dtype=torch.float32)
        expected = (self.n_levels, self.codebook_size, self.dimension)
        if tuple(tensor.shape) != expected:
            raise ContractError(
                "RQ codebooks must have shape {}, got {}".format(expected, tuple(tensor.shape))
            )
        if not torch.isfinite(tensor).all():
            raise ContractError("RQ codebooks must contain only finite values")
        self.codebooks = tensor.detach().clone()

    def fit(self, data: np.ndarray) -> "ResidualQuantizer":
        """Fit every residual level with deterministic scikit-learn K-means."""

        values = np.asarray(data, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ContractError("RQ fitting data must have shape [N, {}]".format(self.dimension))
        if values.shape[0] < self.codebook_size:
            raise ContractError("RQ fitting requires at least codebook_size samples")
        if not np.isfinite(values).all():
            raise ContractError("RQ fitting data must contain only finite values")
        from sklearn.cluster import KMeans  # type: ignore[import-untyped]

        residual = values.copy()
        fitted = []
        for level in range(self.n_levels):
            kmeans = KMeans(
                n_clusters=self.codebook_size,
                init="k-means++",
                n_init=self.n_init,
                max_iter=self.max_iter,
                tol=self.tolerance,
                random_state=self.random_state + level,
                verbose=0,
            )
            indices = kmeans.fit_predict(residual)
            centers = np.asarray(kmeans.cluster_centers_, dtype=np.float32)
            residual = residual - centers[indices]
            fitted.append(centers)
        self.set_codebooks(np.stack(fitted, axis=0))
        return self

    def encode(self, vectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return level codes and their additive reconstruction."""

        self._validate_vectors(vectors)
        if not self.is_fitted:
            raise ContractError("RQ codebooks are not fitted or loaded")
        original_shape = tuple(vectors.shape)
        residual = vectors.reshape(-1, self.dimension)
        reconstruction = torch.zeros_like(residual)
        codes = []
        codebooks = self.codebooks.to(device=vectors.device, dtype=vectors.dtype)
        for level in range(self.n_levels):
            codebook = codebooks[level]
            distances = (
                residual.pow(2).sum(dim=1, keepdim=True)
                + codebook.pow(2).sum(dim=1).unsqueeze(0)
                - 2.0 * residual.matmul(codebook.t())
            )
            indices = distances.argmin(dim=1)
            selected = codebook.index_select(0, indices)
            residual = residual - selected
            reconstruction = reconstruction + selected
            codes.append(indices)
        code_shape = original_shape[:-1] + (self.n_levels,)
        return (
            torch.stack(codes, dim=-1).reshape(code_shape),
            reconstruction.reshape(original_shape),
        )

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted:
            raise ContractError("RQ codebooks are not fitted or loaded")
        if codes.ndim < 2 or codes.shape[-1] != self.n_levels or codes.dtype != torch.long:
            raise ContractError("RQ codes must be long tensors ending in n_levels")
        if codes.numel() and (codes.min() < 0 or codes.max() >= self.codebook_size):
            raise ContractError("RQ code index is outside the codebook")
        flat = codes.reshape(-1, self.n_levels)
        books = self.codebooks.to(codes.device)
        decoded = torch.zeros(flat.shape[0], self.dimension, device=codes.device)
        for level in range(self.n_levels):
            decoded = decoded + books[level].index_select(0, flat[:, level])
        return decoded.reshape(tuple(codes.shape[:-1]) + (self.dimension,))

    def forward(self, vectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encode(vectors)

    def save(self, path: PathLike, metadata: Optional[Mapping[str, object]] = None):
        if not self.is_fitted:
            raise ContractError("RQ codebooks are not fitted or loaded")
        artifact_metadata: Dict[str, object] = {
            "n_levels": self.n_levels,
            "codebook_size": self.codebook_size,
            "dimension": self.dimension,
            "random_state": self.random_state,
            "n_init": self.n_init,
            "max_iter": self.max_iter,
            "tolerance": self.tolerance,
        }
        for name, value in dict(metadata or {}).items():
            if name in artifact_metadata and artifact_metadata[name] != value:
                raise ContractError("RQ artifact metadata cannot override {!r}".format(name))
            artifact_metadata[name] = value
        return save_quantization_artifact(
            path,
            self.artifact_kind,
            artifact_metadata,
            {"codebooks": self.codebooks.detach().cpu().numpy()},
        )

    @classmethod
    def from_artifact(cls, path: PathLike) -> "ResidualQuantizer":
        metadata, arrays = load_quantization_artifact(path, cls.artifact_kind)
        required = {"n_levels", "codebook_size", "dimension"}
        if not required.issubset(metadata):
            raise QuantizationArtifactError("RQ artifact metadata is incomplete")
        quantizer = cls(
            n_levels=int(metadata["n_levels"]),
            codebook_size=int(metadata["codebook_size"]),
            dimension=int(metadata["dimension"]),
            random_state=int(metadata.get("random_state", 42)),
            n_init=int(metadata.get("n_init", 5)),
            max_iter=int(metadata.get("max_iter", 20)),
            tolerance=float(metadata.get("tolerance", 1e-4)),
        )
        if set(arrays) != {"codebooks"}:
            raise QuantizationArtifactError("RQ artifact must contain only codebooks")
        quantizer.set_codebooks(arrays["codebooks"])
        quantizer.artifact_metadata = dict(metadata)
        return quantizer


__all__ = ["ResidualQuantizer"]

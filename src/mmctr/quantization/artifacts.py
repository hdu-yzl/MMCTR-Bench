"""Versioned, pickle-free persistence for quantization artifacts."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple, Union

import numpy as np


PathLike = Union[str, Path]
ARTIFACT_FORMAT = "mmctr-quantization-npz"
ARTIFACT_VERSION = 1
MANIFEST_KEY = "__manifest__"


class QuantizationArtifactError(ValueError):
    """Raised when a quantization artifact is missing, corrupt, or incompatible."""


def _artifact_path(path: PathLike) -> Path:
    artifact = Path(path).expanduser()
    if artifact.suffix.lower() != ".npz":
        artifact = artifact.with_suffix(".npz")
    return artifact.resolve()


def rq_artifact_path(root: PathLike, dataset: str, modality: str) -> Path:
    """Return the stable location for one dataset/modality RQ artifact."""

    return _artifact_path(Path(root) / "rq" / dataset.lower() / modality.lower())


def psrq_artifact_path(root: PathLike, dataset: str) -> Path:
    """Return the stable location for one dataset PSRQ artifact."""

    return _artifact_path(Path(root) / "psrq" / dataset.lower() / "model")


def _array_digest(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def save_quantization_artifact(
    path: PathLike,
    kind: str,
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> Path:
    """Atomically save a validated NPZ artifact and return its absolute path."""

    if not kind or not isinstance(kind, str):
        raise QuantizationArtifactError("artifact kind must be a non-empty string")
    if not arrays:
        raise QuantizationArtifactError("artifact must contain at least one array")
    serializable: Dict[str, np.ndarray] = {}
    array_manifest: Dict[str, Dict[str, Any]] = {}
    for name, value in arrays.items():
        if not isinstance(name, str) or not name or name == MANIFEST_KEY:
            raise QuantizationArtifactError(
                "artifact array names must be non-empty and reserved-safe"
            )
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise QuantizationArtifactError("artifact arrays cannot use object dtype")
        if array.dtype.kind not in "biufc":
            raise QuantizationArtifactError("artifact arrays must use numeric dtypes")
        if not np.isfinite(array).all():
            raise QuantizationArtifactError("artifact arrays must contain only finite values")
        serializable[name] = array
        array_manifest[name] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": _array_digest(array),
        }

    manifest = {
        "format": ARTIFACT_FORMAT,
        "version": ARTIFACT_VERSION,
        "kind": kind,
        "metadata": dict(metadata),
        "arrays": array_manifest,
    }
    try:
        encoded_manifest = json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise QuantizationArtifactError("artifact metadata must be JSON serializable") from error
    serializable[MANIFEST_KEY] = np.asarray(encoded_manifest)

    destination = _artifact_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".npz",
            prefix=".quantization-",
            dir=str(destination.parent),
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            np.savez(temporary, **serializable)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return destination


def load_quantization_artifact(
    path: PathLike,
    expected_kind: str,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Load and integrity-check a versioned NPZ quantization artifact."""

    artifact = _artifact_path(path)
    if not artifact.is_file():
        raise QuantizationArtifactError("quantization artifact not found: {}".format(artifact))
    try:
        with np.load(str(artifact), allow_pickle=False) as payload:
            if MANIFEST_KEY not in payload.files:
                raise QuantizationArtifactError("artifact manifest is missing")
            manifest = json.loads(str(payload[MANIFEST_KEY].item()))
            if manifest.get("format") != ARTIFACT_FORMAT:
                raise QuantizationArtifactError("unsupported quantization artifact format")
            if manifest.get("version") != ARTIFACT_VERSION:
                raise QuantizationArtifactError(
                    "unsupported quantization artifact version: {}".format(
                        manifest.get("version")
                    )
                )
            if manifest.get("kind") != expected_kind:
                raise QuantizationArtifactError(
                    "expected {!r} artifact, found {!r}".format(
                        expected_kind, manifest.get("kind")
                    )
                )
            declared = manifest.get("arrays")
            if not isinstance(declared, dict):
                raise QuantizationArtifactError("artifact array manifest is invalid")
            actual_names = set(payload.files) - {MANIFEST_KEY}
            if actual_names != set(declared):
                raise QuantizationArtifactError("artifact array set does not match manifest")
            arrays: Dict[str, np.ndarray] = {}
            for name, specification in declared.items():
                array = np.array(payload[name], copy=True)
                if list(array.shape) != specification.get("shape"):
                    raise QuantizationArtifactError("array {!r} shape mismatch".format(name))
                if str(array.dtype) != specification.get("dtype"):
                    raise QuantizationArtifactError("array {!r} dtype mismatch".format(name))
                if not np.isfinite(array).all():
                    raise QuantizationArtifactError(
                        "array {!r} contains non-finite values".format(name)
                    )
                if _array_digest(array) != specification.get("sha256"):
                    raise QuantizationArtifactError("array {!r} checksum mismatch".format(name))
                arrays[name] = array
    except QuantizationArtifactError:
        raise
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise QuantizationArtifactError(
            "cannot read quantization artifact {}: {}".format(artifact, error)
        ) from error
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        raise QuantizationArtifactError("artifact metadata must be a mapping")
    return dict(metadata), arrays


__all__ = [
    "ARTIFACT_FORMAT",
    "ARTIFACT_VERSION",
    "QuantizationArtifactError",
    "load_quantization_artifact",
    "psrq_artifact_path",
    "rq_artifact_path",
    "save_quantization_artifact",
]

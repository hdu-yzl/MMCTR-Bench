"""Restartable batch feature extraction without model lifecycle duplication."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Protocol, Sequence, Tuple

import numpy as np

from mmctr.core import ContractError


EXTRACTION_SCHEMA_VERSION = 1


class BatchEncoder(Protocol):
    """A loaded text or image encoder used for the complete extraction job."""

    def encode(self, values: Sequence[Any]) -> np.ndarray: ...


@dataclass(frozen=True)
class ExtractionInput:
    key: str
    value: Any

    def __post_init__(self) -> None:
        if not self.key:
            raise ContractError("extraction keys cannot be empty")


@dataclass(frozen=True)
class ExtractionShard:
    start: int
    stop: int
    feature_file: str
    key_file: str
    sha256: str
    key_sha256: str

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop <= self.start:
            raise ContractError("invalid extraction shard range")
        if len(self.sha256) != 64 or len(self.key_sha256) != 64:
            raise ContractError("extraction shard requires a SHA-256 digest")


@dataclass(frozen=True)
class ExtractionManifest:
    """Bind contiguous committed shards to one source and encoder-output contract."""

    field: str
    dimension: int
    source_fingerprint: str
    shards: Tuple[ExtractionShard, ...]
    missing_keys: Tuple[str, ...]
    schema_version: int = EXTRACTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EXTRACTION_SCHEMA_VERSION:
            raise ContractError("unsupported extraction manifest schema")
        if not self.field or self.dimension <= 0 or not self.source_fingerprint:
            raise ContractError("field, dimension, and source fingerprint are required")
        expected_start = 0
        for shard in self.shards:
            if shard.start != expected_start:
                raise ContractError("extraction shards must be contiguous")
            expected_start = shard.stop

    @property
    def completed(self) -> int:
        return self.shards[-1].stop if self.shards else 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "field": self.field,
            "dimension": self.dimension,
            "source_fingerprint": self.source_fingerprint,
            "completed": self.completed,
            "missing_keys": list(self.missing_keys),
            "shards": [
                {
                    "start": shard.start,
                    "stop": shard.stop,
                    "feature_file": shard.feature_file,
                    "key_file": shard.key_file,
                    "sha256": shard.sha256,
                    "key_sha256": shard.key_sha256,
                }
                for shard in self.shards
            ],
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ExtractionManifest":
        return cls(
            field=str(values["field"]),
            dimension=int(values["dimension"]),
            source_fingerprint=str(values["source_fingerprint"]),
            shards=tuple(ExtractionShard(**value) for value in values.get("shards", ())),
            missing_keys=tuple(str(value) for value in values.get("missing_keys", ())),
            schema_version=int(values.get("schema_version", EXTRACTION_SCHEMA_VERSION)),
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, values: Mapping[str, Any]) -> str:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(values, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = _sha256_file(temporary)
    temporary.replace(path)
    return digest


def _atomic_array(path: Path, values: np.ndarray) -> str:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
    digest = _sha256_file(temporary)
    temporary.replace(path)
    return digest


def _load_manifest(path: Path) -> ExtractionManifest:
    return ExtractionManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def run_batch_extraction(
    records: Iterable[ExtractionInput],
    output_dir: Path,
    field: str,
    dimension: int,
    source_fingerprint: str,
    encoder_factory: Callable[[], BatchEncoder],
    batch_size: int = 256,
) -> ExtractionManifest:
    """Encode records once per batch and atomically checkpoint every completed shard.

    The encoder factory is invoked exactly once and only when unfinished, non-missing
    values exist. Missing values receive a zero vector and are listed in the manifest.
    A compatible manifest resumes at its last contiguous shard boundary.
    """

    if batch_size <= 0 or dimension <= 0:
        raise ContractError("batch_size and dimension must be positive")
    materialized = tuple(records)
    if len({record.key for record in materialized}) != len(materialized):
        raise ContractError("extraction keys must be unique")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest = ExtractionManifest(field, dimension, source_fingerprint, (), ())
    if manifest_path.exists():
        manifest = _load_manifest(manifest_path)
        if (
            manifest.field != field
            or manifest.dimension != dimension
            or manifest.source_fingerprint != source_fingerprint
        ):
            raise ContractError("existing extraction manifest does not match this job")
        for shard in manifest.shards:
            feature_path = output_dir / shard.feature_file
            key_path = output_dir / shard.key_file
            if not feature_path.is_file() or not key_path.is_file():
                raise ContractError("completed extraction shard is missing")
            if _sha256_file(feature_path) != shard.sha256:
                raise ContractError("completed extraction shard checksum changed")
            if _sha256_file(key_path) != shard.key_sha256:
                raise ContractError("completed extraction key shard checksum changed")
            saved_keys = json.loads(key_path.read_text(encoding="utf-8"))["keys"]
            current_keys = [record.key for record in materialized[shard.start : shard.stop]]
            if saved_keys != current_keys:
                raise ContractError("resume input keys do not match the completed shard")
    if manifest.completed > len(materialized):
        raise ContractError("resume manifest is longer than the current input")

    encoder = None
    shards = list(manifest.shards)
    missing_keys = list(manifest.missing_keys)
    for start in range(manifest.completed, len(materialized), batch_size):
        stop = min(start + batch_size, len(materialized))
        batch = materialized[start:stop]
        present = [
            (index, record) for index, record in enumerate(batch) if not _is_missing(record.value)
        ]
        features = np.zeros((len(batch), dimension), dtype=np.float32)
        if present:
            if encoder is None:
                encoder = encoder_factory()
            encoded = np.asarray(
                encoder.encode([record.value for _, record in present]), dtype=np.float32
            )
            if encoded.shape != (len(present), dimension):
                raise ContractError(
                    "encoder output shape {} does not match ({}, {})".format(
                        encoded.shape, len(present), dimension
                    )
                )
            if not np.isfinite(encoded).all():
                raise ContractError("encoder output contains NaN or Inf")
            for encoded_row, (batch_index, _) in zip(encoded, present):
                features[batch_index] = encoded_row
        missing_keys.extend(record.key for record in batch if _is_missing(record.value))

        stem = "{:012d}-{:012d}".format(start, stop)
        feature_file = stem + ".npy"
        key_file = stem + ".keys.json"
        digest = _atomic_array(output_dir / feature_file, features)
        key_digest = _atomic_json(output_dir / key_file, {"keys": [record.key for record in batch]})
        shards.append(ExtractionShard(start, stop, feature_file, key_file, digest, key_digest))
        manifest = ExtractionManifest(
            field,
            dimension,
            source_fingerprint,
            tuple(shards),
            tuple(missing_keys),
        )
        _atomic_json(manifest_path, manifest.to_dict())
    if not manifest_path.exists():
        _atomic_json(manifest_path, manifest.to_dict())
    return manifest


def iter_extracted_features(
    output_dir: Path,
) -> Iterable[Tuple[Tuple[str, ...], np.ndarray]]:
    """Yield verified extraction shards in deterministic input order."""

    output_dir = Path(output_dir)
    manifest = _load_manifest(output_dir / "manifest.json")
    for shard in manifest.shards:
        feature_path = output_dir / shard.feature_file
        if _sha256_file(feature_path) != shard.sha256:
            raise ContractError("extraction shard checksum changed")
        key_path = output_dir / shard.key_file
        if _sha256_file(key_path) != shard.key_sha256:
            raise ContractError("extraction key shard checksum changed")
        keys = json.loads(key_path.read_text(encoding="utf-8"))["keys"]
        values = np.load(feature_path, allow_pickle=False)
        if values.shape != (len(keys), manifest.dimension):
            raise ContractError("extraction shard keys and values disagree")
        yield tuple(str(key) for key in keys), values


__all__ = [
    "BatchEncoder",
    "ExtractionInput",
    "ExtractionManifest",
    "ExtractionShard",
    "iter_extracted_features",
    "run_batch_extraction",
]

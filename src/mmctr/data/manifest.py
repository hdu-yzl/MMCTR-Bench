"""Versioned dataset metadata used by loaders and experiment provenance."""

import hashlib
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional

from mmctr.core import ContractError


MANIFEST_SCHEMA_VERSION = 1
KNOWN_SPLITS = frozenset({"train", "val", "test"})


def _copy_mapping(values: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    if not isinstance(values, Mapping):
        raise ContractError("{} must be a mapping".format(field_name))
    copied = dict(values)
    for name in copied:
        if not isinstance(name, str) or not name:
            raise ContractError("{} keys must be non-empty strings".format(field_name))
    return MappingProxyType(copied)


@dataclass(frozen=True)
class SplitStatistics:
    """Auditable summary for one dataset split."""

    samples: int
    positives: Optional[int] = None
    users: Optional[int] = None
    items: Optional[int] = None
    sha256: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("samples", "positives", "users", "items"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or int(value) < 0):
                raise ContractError("{} must be a non-negative integer".format(name))
        if self.positives is not None and self.positives > self.samples:
            raise ContractError("positives cannot exceed samples")
        if self.sha256 is not None:
            digest = self.sha256.lower()
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ContractError("sha256 must be a 64-character hexadecimal digest")
            object.__setattr__(self, "sha256", digest)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "samples": int(self.samples),
            "positives": self.positives,
            "users": self.users,
            "items": self.items,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class DatasetManifest:
    """Stable description of a processed dataset contract."""

    name: str
    version: str
    storage_format: str
    sequence_length: int
    padding_id: int
    feature_dimensions: Mapping[str, int]
    splits: Mapping[str, SplitStatistics] = field(default_factory=dict)
    id_offsets: Mapping[str, int] = field(default_factory=dict)
    oov_id: Optional[int] = None
    source_fingerprint: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ContractError(
                "unsupported dataset manifest schema version: {}".format(self.schema_version)
            )
        if not self.name or not self.version or not self.storage_format:
            raise ContractError("manifest name, version, and storage_format are required")
        if isinstance(self.sequence_length, bool) or int(self.sequence_length) <= 0:
            raise ContractError("sequence_length must be a positive integer")

        dimensions = _copy_mapping(self.feature_dimensions, "feature_dimensions")
        normalised_dimensions: Dict[str, int] = {}
        for name, value in dimensions.items():
            if isinstance(value, bool) or int(value) <= 0:
                raise ContractError("feature dimension {!r} must be positive".format(name))
            normalised_dimensions[name] = int(value)

        offsets = _copy_mapping(self.id_offsets, "id_offsets")
        normalised_offsets: Dict[str, int] = {}
        for name, value in offsets.items():
            if isinstance(value, bool):
                raise ContractError("ID offset {!r} must be an integer".format(name))
            normalised_offsets[name] = int(value)

        splits = _copy_mapping(self.splits, "splits")
        normalised_splits: Dict[str, SplitStatistics] = {}
        for name, value in splits.items():
            if name not in KNOWN_SPLITS:
                raise ContractError("unknown dataset split: {!r}".format(name))
            if isinstance(value, SplitStatistics):
                normalised_splits[name] = value
            elif isinstance(value, Mapping):
                normalised_splits[name] = SplitStatistics(**dict(value))
            else:
                raise ContractError("split {!r} must be SplitStatistics or a mapping".format(name))

        object.__setattr__(self, "feature_dimensions", MappingProxyType(normalised_dimensions))
        object.__setattr__(self, "id_offsets", MappingProxyType(normalised_offsets))
        object.__setattr__(self, "splits", MappingProxyType(normalised_splits))
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata, "metadata"))

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(include_fingerprint=False),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self, include_fingerprint: bool = True) -> Dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "name": self.name,
            "version": self.version,
            "storage_format": self.storage_format,
            "sequence_length": self.sequence_length,
            "padding_id": self.padding_id,
            "oov_id": self.oov_id,
            "id_offsets": dict(self.id_offsets),
            "feature_dimensions": dict(self.feature_dimensions),
            "splits": {name: value.to_dict() for name, value in self.splits.items()},
            "source_fingerprint": self.source_fingerprint,
            "metadata": dict(self.metadata),
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    @classmethod
    def from_config(cls, name: str, config: Mapping[str, Any]) -> "DatasetManifest":
        """Build the minimum manifest available from a legacy data config."""

        multimodal_dimensions = dict(config.get("mm_seq_dims", config.get("mm_dims", {})))
        feature_dimensions = {
            feature_name: int(dimension)
            for feature_name, dimension in multimodal_dimensions.items()
            if int(dimension) > 0
        }
        feature_dimensions.setdefault("id", 1)
        return cls(
            name=name.lower(),
            version=str(config.get("version", "legacy-unversioned")),
            storage_format=str(config.get("storage_format", "tfrecord")),
            sequence_length=int(config["seq_len"]),
            padding_id=int(config.get("padding_id", 0)),
            oov_id=config.get("oov_id"),
            id_offsets=dict(config.get("id_offsets", {})),
            feature_dimensions=feature_dimensions,
            source_fingerprint=config.get("fingerprint"),
            metadata={"manifest_completeness": "legacy-config-only"},
        )


__all__ = ["DatasetManifest", "KNOWN_SPLITS", "SplitStatistics"]

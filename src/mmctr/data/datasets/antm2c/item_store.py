"""Deterministic item indexing and deduplicated AntM2C feature storage."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from mmctr.core import ContractError


@dataclass(frozen=True)
class ItemIndex:
    """Bidirectional original-item to contiguous-index mapping with padding row zero."""

    by_original_id: Mapping[Any, int]
    by_index: Tuple[Optional[Any], ...]
    padding_id: int = 0
    oov_id: Optional[int] = None

    def __post_init__(self) -> None:
        mapping = dict(self.by_original_id)
        if self.padding_id != 0:
            raise ContractError("AntM2C item index currently requires padding ID 0")
        if not self.by_index or self.by_index[0] is not None:
            raise ContractError("item index row zero must be reserved for padding")
        expected = set(range(1, len(self.by_index)))
        if set(mapping.values()) != expected:
            raise ContractError("item indices must be contiguous and start at 1")
        for original_id, index in mapping.items():
            if self.by_index[index] != original_id:
                raise ContractError("item index forward and reverse mappings disagree")
        object.__setattr__(self, "by_original_id", MappingProxyType(mapping))

    @property
    def item_count(self) -> int:
        return len(self.by_index) - 1

    def encode(self, original_id: Any) -> int:
        try:
            return self.by_original_id[original_id]
        except KeyError as error:
            if self.oov_id is not None:
                return self.oov_id
            raise ContractError("unknown original item ID: {!r}".format(original_id)) from error

    def encode_many(self, original_ids: Iterable[Any]) -> Tuple[int, ...]:
        return tuple(self.encode(original_id) for original_id in original_ids)

    def decode(self, index: int) -> Optional[Any]:
        if index < 0 or index >= len(self.by_index):
            raise ContractError("item index is out of range: {}".format(index))
        return self.by_index[index]


def build_item_index(original_ids: Iterable[Any]) -> ItemIndex:
    """Assign indices by stable first appearance in the versioned event stream."""

    mapping: Dict[Any, int] = {}
    reverse = [None]
    for original_id in original_ids:
        if original_id is None:
            raise ContractError("original item IDs cannot be null")
        if original_id not in mapping:
            mapping[original_id] = len(reverse)
            reverse.append(original_id)
    return ItemIndex(mapping, tuple(reverse))


@dataclass(frozen=True)
class FeatureStoreAudit:
    missing_by_feature: Mapping[str, Tuple[Any, ...]]

    def __post_init__(self) -> None:
        values = {name: tuple(missing) for name, missing in self.missing_by_feature.items()}
        object.__setattr__(self, "missing_by_feature", MappingProxyType(values))


@dataclass(frozen=True)
class ItemFeatureStore:
    index: ItemIndex
    features: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        copied: Dict[str, np.ndarray] = {}
        expected_rows = self.index.item_count + 1
        for name, values in self.features.items():
            array = np.asarray(values)
            if array.ndim != 2:
                raise ContractError("item feature {!r} must have shape [N, D]".format(name))
            if array.shape[0] != expected_rows:
                raise ContractError("item feature {!r} row count does not match index".format(name))
            if not np.issubdtype(array.dtype, np.floating):
                raise ContractError("item feature {!r} must use a floating dtype".format(name))
            if not np.isfinite(array).all():
                raise ContractError("item feature {!r} contains NaN or Inf".format(name))
            if np.any(array[self.index.padding_id] != 0):
                raise ContractError("item feature padding row must be all zero")
            copied[name] = array
        object.__setattr__(self, "features", MappingProxyType(copied))

    def gather(self, feature: str, item_indices: Union[Sequence[int], np.ndarray]) -> np.ndarray:
        try:
            values = self.features[feature]
        except KeyError as error:
            raise ContractError("unknown item feature: {!r}".format(feature)) from error
        indices = np.asarray(item_indices, dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= values.shape[0]):
            raise ContractError("item feature gather index is out of range")
        return values[indices]


def build_feature_store(
    index: ItemIndex,
    features_by_name: Mapping[str, Mapping[Any, np.ndarray]],
    dimensions: Mapping[str, int],
) -> Tuple[ItemFeatureStore, FeatureStoreAudit]:
    """Build one row per item and report zero-filled missing features."""

    arrays: Dict[str, np.ndarray] = {}
    missing_by_feature: Dict[str, Tuple[Any, ...]] = {}
    for name, dimension in dimensions.items():
        if dimension <= 0:
            raise ContractError("feature dimensions must be positive")
        source = features_by_name.get(name, {})
        values = np.zeros((index.item_count + 1, dimension), dtype=np.float32)
        missing = []
        for item_index in range(1, index.item_count + 1):
            original_id = index.decode(item_index)
            if original_id not in source:
                missing.append(original_id)
                continue
            feature = np.asarray(source[original_id], dtype=np.float32).reshape(-1)
            if feature.shape != (dimension,):
                raise ContractError(
                    "feature {!r} for item {!r} has dimension {}, expected {}".format(
                        name, original_id, feature.shape[0], dimension
                    )
                )
            values[item_index] = feature
        arrays[name] = values
        missing_by_feature[name] = tuple(missing)
    return ItemFeatureStore(index, arrays), FeatureStoreAudit(missing_by_feature)


__all__ = [
    "FeatureStoreAudit",
    "ItemFeatureStore",
    "ItemIndex",
    "build_feature_store",
    "build_item_index",
]

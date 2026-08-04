"""Named, slice-free AntM2C interaction storage and canonical batch loader."""

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Tuple

import numpy as np
import torch

from mmctr.core import Batch, ContractError
from mmctr.data.manifest import DatasetManifest

from .item_store import ItemFeatureStore, ItemIndex


ARRAY_STORE_SCHEMA_VERSION = 1
LEGACY_USER_TEXT_CONTEXT = (
    "service_text",
    "query_text",
    "bill_text",
    "entity_text",
    "time_context",
)
MMapMode = Optional[Literal["r+", "r", "w+", "c"]]


@dataclass(frozen=True)
class InteractionTable:
    """One split of interactions with independently named context features."""

    event_ids: Tuple[str, ...]
    user_indices: np.ndarray
    item_indices: np.ndarray
    history_item_indices: np.ndarray
    labels: np.ndarray
    context_features: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        users = np.asarray(self.user_indices, dtype=np.int64)
        items = np.asarray(self.item_indices, dtype=np.int64)
        histories = np.asarray(self.history_item_indices, dtype=np.int64)
        labels = np.asarray(self.labels, dtype=np.float32)
        sample_count = len(self.event_ids)
        if users.shape != (sample_count,) or items.shape != (sample_count,):
            raise ContractError("user/item indices must have shape [N]")
        if histories.ndim != 2 or histories.shape[0] != sample_count:
            raise ContractError("history item indices must have shape [N, L]")
        if labels.shape != (sample_count,):
            raise ContractError("labels must have shape [N]")
        if len(set(self.event_ids)) != sample_count:
            raise ContractError("event IDs must be unique within a split")
        if np.any(users <= 0) or np.any(items <= 0) or np.any(histories < 0):
            raise ContractError("interaction indices violate padding/item rules")
        if not np.isfinite(labels).all():
            raise ContractError("labels contain NaN or Inf")
        contexts: Dict[str, np.ndarray] = {}
        for name, values in self.context_features.items():
            array = np.asarray(values, dtype=np.float32)
            if not name or array.ndim != 2 or array.shape[0] != sample_count:
                raise ContractError("context features must be named arrays with shape [N, D]")
            if not np.isfinite(array).all():
                raise ContractError("context feature {!r} contains NaN or Inf".format(name))
            contexts[name] = array
        object.__setattr__(self, "event_ids", tuple(str(value) for value in self.event_ids))
        object.__setattr__(self, "user_indices", users)
        object.__setattr__(self, "item_indices", items)
        object.__setattr__(self, "history_item_indices", histories)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "context_features", MappingProxyType(contexts))

    @property
    def sample_count(self) -> int:
        return len(self.event_ids)

    @property
    def sequence_length(self) -> int:
        return int(self.history_item_indices.shape[1])


def _write_array(path: Path, values: np.ndarray) -> None:
    with path.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)


def _write_json(path: Path, values: Mapping) -> None:
    path.write_text(
        json.dumps(values, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_array_store(
    output_dir: Path,
    manifest: DatasetManifest,
    splits: Mapping[str, InteractionTable],
    item_store: ItemFeatureStore,
) -> Path:
    """Write a candidate named-array layout without committing the final benchmark format."""

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ContractError("array store output already exists: {}".format(output_dir))
    staging = output_dir.with_name(output_dir.name + ".incomplete")
    if staging.exists():
        raise ContractError("incomplete array store already exists: {}".format(staging))
    if manifest.name != "antm2c":
        raise ContractError("AntM2C array store requires an antm2c manifest")
    if manifest.sequence_length <= 0:
        raise ContractError("manifest sequence length must be positive")

    all_event_ids: List[str] = []
    for split_name, table in sorted(splits.items()):
        if split_name not in {"train", "val", "test"}:
            raise ContractError("unknown split: {!r}".format(split_name))
        if table.sequence_length != manifest.sequence_length:
            raise ContractError("split sequence length does not match manifest")
        if np.any(table.item_indices > item_store.index.item_count):
            raise ContractError("target item index is outside the item store")
        if np.any(table.history_item_indices > item_store.index.item_count):
            raise ContractError("history item index is outside the item store")
        if split_name in manifest.splits:
            expected_samples = manifest.splits[split_name].samples
            if table.sample_count != expected_samples:
                raise ContractError("split sample count does not match manifest")
        for name, values in table.context_features.items():
            expected_dimension = manifest.feature_dimensions.get(name)
            if expected_dimension is not None and values.shape[1] != expected_dimension:
                raise ContractError(
                    "context feature {!r} dimension does not match manifest".format(name)
                )
        all_event_ids.extend(table.event_ids)

    if len(set(all_event_ids)) != len(all_event_ids):
        raise ContractError("event IDs must be unique across all splits")
    for name, values in item_store.features.items():
        expected_dimension = manifest.feature_dimensions.get(name)
        if expected_dimension is not None and values.shape[1] != expected_dimension:
            raise ContractError("item feature {!r} dimension does not match manifest".format(name))

    staging.mkdir(parents=True)
    layout: Dict[str, Any] = {
        "schema_version": ARRAY_STORE_SCHEMA_VERSION,
        "format": "named-npy-candidate-v1",
        "splits": {},
        "item_features": sorted(item_store.features),
    }
    for split_name, table in sorted(splits.items()):
        split_dir = staging / "splits" / split_name
        context_dir = split_dir / "context"
        context_dir.mkdir(parents=True)
        _write_array(split_dir / "event_id.npy", np.asarray(table.event_ids, dtype=np.str_))
        _write_array(split_dir / "user_index.npy", table.user_indices)
        _write_array(split_dir / "item_index.npy", table.item_indices)
        _write_array(split_dir / "history_item_index.npy", table.history_item_indices)
        _write_array(split_dir / "labels.npy", table.labels)
        for name, values in table.context_features.items():
            _write_array(context_dir / (name + ".npy"), values)
        layout["splits"][split_name] = {
            "samples": table.sample_count,
            "context_features": sorted(table.context_features),
        }

    item_dir = staging / "items"
    feature_dir = item_dir / "features"
    feature_dir.mkdir(parents=True)
    _write_json(
        item_dir / "index.json",
        {
            "padding_id": item_store.index.padding_id,
            "oov_id": item_store.index.oov_id,
            "by_index": list(item_store.index.by_index),
        },
    )
    for name, values in item_store.features.items():
        _write_array(feature_dir / (name + ".npy"), values)
    _write_json(staging / "dataset_manifest.json", manifest.to_dict())
    _write_json(staging / "layout.json", layout)
    staging.rename(output_dir)
    return output_dir


def _load_manifest(path: Path) -> DatasetManifest:
    values = json.loads(path.read_text(encoding="utf-8"))
    values.pop("fingerprint", None)
    return DatasetManifest(**values)


def load_array_store(
    input_dir: Path,
    mmap_mode: MMapMode = "r",
) -> Tuple[DatasetManifest, Mapping[str, InteractionTable], ItemFeatureStore]:
    """Read the candidate format using its layout metadata rather than fixed slices."""

    input_dir = Path(input_dir)
    layout = json.loads((input_dir / "layout.json").read_text(encoding="utf-8"))
    if layout.get("schema_version") != ARRAY_STORE_SCHEMA_VERSION:
        raise ContractError("unsupported AntM2C array store schema")
    manifest = _load_manifest(input_dir / "dataset_manifest.json")
    splits: Dict[str, InteractionTable] = {}
    for split_name, split_layout in layout["splits"].items():
        split_dir = input_dir / "splits" / split_name
        contexts = {
            name: np.load(split_dir / "context" / (name + ".npy"), mmap_mode=mmap_mode)
            for name in split_layout["context_features"]
        }
        splits[split_name] = InteractionTable(
            event_ids=tuple(
                str(value) for value in np.load(split_dir / "event_id.npy", allow_pickle=False)
            ),
            user_indices=np.load(split_dir / "user_index.npy", mmap_mode=mmap_mode),
            item_indices=np.load(split_dir / "item_index.npy", mmap_mode=mmap_mode),
            history_item_indices=np.load(split_dir / "history_item_index.npy", mmap_mode=mmap_mode),
            labels=np.load(split_dir / "labels.npy", mmap_mode=mmap_mode),
            context_features=contexts,
        )

    item_layout = json.loads((input_dir / "items" / "index.json").read_text(encoding="utf-8"))
    by_index = tuple(item_layout["by_index"])
    item_index = ItemIndex(
        {original_id: index for index, original_id in enumerate(by_index) if index},
        by_index,
        padding_id=int(item_layout["padding_id"]),
        oov_id=item_layout.get("oov_id"),
    )
    item_features = {
        name: np.load(input_dir / "items" / "features" / (name + ".npy"), mmap_mode=mmap_mode)
        for name in layout["item_features"]
    }
    return manifest, MappingProxyType(splits), ItemFeatureStore(item_index, item_features)


class AntM2CArrayLoader:
    """Canonical loader over named interaction and item arrays."""

    dataset_name = "antm2c"

    def __init__(
        self,
        manifest: DatasetManifest,
        splits: Mapping[str, InteractionTable],
        item_store: ItemFeatureStore,
        batch_size: int,
    ) -> None:
        if batch_size <= 0:
            raise ContractError("batch_size must be positive")
        self.manifest = manifest
        self.splits = MappingProxyType(dict(splits))
        self.item_store = item_store
        self.batch_size = int(batch_size)

    @classmethod
    def from_directory(
        cls, input_dir: Path, batch_size: int, mmap_mode: MMapMode = "r"
    ) -> "AntM2CArrayLoader":
        manifest, splits, item_store = load_array_store(input_dir, mmap_mode=mmap_mode)
        return cls(manifest, splits, item_store, batch_size)

    @staticmethod
    def _offset_ids(values: np.ndarray, offset: int, padding_id: int = 0) -> np.ndarray:
        result = np.asarray(values, dtype=np.int64).copy()
        result[result != padding_id] += int(offset)
        return result

    def iter_batches(self, split: str) -> Iterable[Batch]:
        split = "val" if split == "validation" else split
        try:
            table = self.splits[split]
        except KeyError as error:
            raise ContractError("array store does not contain split {!r}".format(split)) from error
        user_offset = int(self.manifest.id_offsets.get("user", 0))
        item_offset = int(self.manifest.id_offsets.get("item", 0))
        for start in range(0, table.sample_count, self.batch_size):
            stop = min(start + self.batch_size, table.sample_count)
            item_indices = table.item_indices[start:stop]
            history_indices = table.history_item_indices[start:stop]
            item_features = {
                "id": torch.as_tensor(
                    self._offset_ids(item_indices, item_offset)[:, None], dtype=torch.long
                )
            }
            history_features = {
                "id": torch.as_tensor(
                    self._offset_ids(history_indices, item_offset), dtype=torch.long
                )
            }
            for name in self.item_store.features:
                item_features[name] = torch.as_tensor(
                    self.item_store.gather(name, item_indices), dtype=torch.float32
                )
                history_features[name] = torch.as_tensor(
                    self.item_store.gather(name, history_indices), dtype=torch.float32
                )
            context_features = {
                name: torch.as_tensor(np.array(values[start:stop], copy=True), dtype=torch.float32)
                for name, values in table.context_features.items()
            }
            if "text" not in context_features and all(
                name in context_features for name in LEGACY_USER_TEXT_CONTEXT
            ):
                context_features["text"] = torch.cat(
                    [context_features[name] for name in LEGACY_USER_TEXT_CONTEXT], dim=-1
                )
            yield Batch(
                user_features={
                    "id": torch.as_tensor(
                        self._offset_ids(table.user_indices[start:stop], user_offset)[:, None],
                        dtype=torch.long,
                    )
                },
                item_features=item_features,
                history_features=history_features,
                history_mask=torch.as_tensor(
                    history_indices != self.manifest.padding_id, dtype=torch.bool
                ),
                labels=torch.as_tensor(
                    np.array(table.labels[start:stop], copy=True), dtype=torch.float32
                ),
                context_features=context_features,
                metadata={
                    "dataset": self.dataset_name,
                    "dataset_version": self.manifest.version,
                    "dataset_fingerprint": self.manifest.fingerprint,
                    "split": split,
                    "event_ids": table.event_ids[start:stop],
                },
            )


__all__ = [
    "ARRAY_STORE_SCHEMA_VERSION",
    "AntM2CArrayLoader",
    "InteractionTable",
    "LEGACY_USER_TEXT_CONTEXT",
    "load_array_store",
    "write_array_store",
]

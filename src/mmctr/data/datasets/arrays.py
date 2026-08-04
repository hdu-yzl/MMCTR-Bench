"""Memory-mapped named-array dataset shared by canonical dataset implementations."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Tuple

import numpy as np
import torch

from mmctr.core import Batch, ContractError
from mmctr.data.manifest import DatasetManifest


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without loading a dataset file into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint(files: Mapping[str, Path]) -> Tuple[str, Mapping[str, str]]:
    """Hash named source files and their logical names into one stable fingerprint."""

    digests = {name: sha256_file(Path(path)) for name, path in sorted(files.items())}
    payload = json.dumps(digests, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), digests


def aggregate_file_digest(files: Mapping[str, Path]) -> str:
    """Hash already-written split files into one order-independent digest."""

    _, digests = source_fingerprint(files)
    payload = json.dumps(digests, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save_manifest(manifest: DatasetManifest, path: Path) -> None:
    """Serialize a manifest together with its self-verifying contract fingerprint."""

    Path(path).write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_manifest(path: Path) -> DatasetManifest:
    """Load and verify a serialized :class:`DatasetManifest`."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected_fingerprint = payload.pop("fingerprint", None)
    manifest = DatasetManifest(**payload)
    if expected_fingerprint != manifest.fingerprint:
        raise ContractError("dataset manifest fingerprint does not match its contents")
    return manifest


class NamedArrayDatasetLoader:
    """Read canonical interaction splits and one dense item-feature store."""

    dataset_name = ""

    def __init__(self, data_config: Mapping[str, Any], batch_size: int) -> None:
        if not self.dataset_name:
            raise TypeError("NamedArrayDatasetLoader subclasses must declare dataset_name")
        if isinstance(batch_size, bool) or int(batch_size) <= 0:
            raise ContractError("batch_size must be a positive integer")
        self.batch_size = int(batch_size)
        try:
            self.root = Path(data_config["data_dir"])
        except KeyError as error:
            raise ContractError("canonical dataset config requires data_dir") from error
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise ContractError("canonical dataset manifest is missing: {}".format(manifest_path))
        self.manifest = load_manifest(manifest_path)
        if self.manifest.name != self.dataset_name:
            raise ContractError("dataset manifest name does not match loader")
        if self.manifest.storage_format != "named-npy-v1":
            raise ContractError("unsupported canonical dataset storage format")
        self._item_ids = np.load(self.root / "item_features" / "item_ids.npy", mmap_mode="r")
        self._modalities = {
            name: np.load(self.root / "item_features" / "{}.npy".format(name), mmap_mode="r")
            for name in self.manifest.feature_dimensions
            if name != "id"
        }
        self._validate_item_store()

    def _validate_item_store(self) -> None:
        if self._item_ids.ndim != 1 or self._item_ids.shape[0] < 2:
            raise ContractError("item feature store must contain padding and at least one item")
        if int(self._item_ids[0]) != self.manifest.padding_id:
            raise ContractError("item feature store padding row does not match manifest")
        item_ids = np.asarray(self._item_ids[1:])
        if not np.array_equal(item_ids, np.arange(item_ids[0], item_ids[0] + len(item_ids))):
            raise ContractError("item feature store IDs must be contiguous")
        for name, values in self._modalities.items():
            expected_dimension = self.manifest.feature_dimensions[name]
            if values.ndim != 2 or values.shape != (len(self._item_ids), expected_dimension):
                raise ContractError("item feature array {!r} has an invalid shape".format(name))
            if not np.issubdtype(values.dtype, np.floating):
                raise ContractError("item feature array {!r} must be floating point".format(name))

    @staticmethod
    def _split_name(split: str) -> str:
        value = str(split).lower()
        if value == "validation":
            value = "val"
        if value not in {"train", "val", "test"}:
            raise ContractError("split must be one of train, val, or test")
        return value

    def _feature_rows(self, item_ids: torch.Tensor) -> torch.Tensor:
        first_item_id = int(self._item_ids[1])
        rows = torch.where(
            item_ids.eq(self.manifest.padding_id),
            torch.zeros_like(item_ids),
            item_ids - first_item_id + 1,
        )
        if bool((rows < 0).any()) or bool((rows >= len(self._item_ids)).any()):
            raise ContractError("interaction references an item outside the feature store")
        return rows

    @staticmethod
    def _tensor(values: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(np.array(values, copy=True), dtype=dtype)

    def iter_batches(self, split: str) -> Iterator[Batch]:
        canonical_split = self._split_name(split)
        split_root = self.root / "splits" / canonical_split
        arrays = {
            name: np.load(split_root / "{}.npy".format(name), mmap_mode="r")
            for name in ("user_ids", "item_ids", "history_item_ids", "labels")
        }
        sample_count = len(arrays["labels"])
        if any(len(values) != sample_count for values in arrays.values()):
            raise ContractError("canonical split arrays have inconsistent sample counts")
        for batch_index, start in enumerate(range(0, sample_count, self.batch_size)):
            stop = min(start + self.batch_size, sample_count)
            user_ids = self._tensor(arrays["user_ids"][start:stop], torch.long).reshape(-1, 1)
            item_ids = self._tensor(arrays["item_ids"][start:stop], torch.long).reshape(-1, 1)
            history_ids = self._tensor(arrays["history_item_ids"][start:stop], torch.long)
            labels = self._tensor(arrays["labels"][start:stop], torch.float32).reshape(-1)
            item_rows = self._feature_rows(item_ids).numpy()
            history_rows = self._feature_rows(history_ids).numpy()
            item_features: Dict[str, torch.Tensor] = {"id": item_ids}
            history_features: Dict[str, torch.Tensor] = {"id": history_ids}
            for name, values in self._modalities.items():
                item_features[name] = self._tensor(values[item_rows.reshape(-1)], torch.float32)
                history_values = values[history_rows.reshape(-1)].reshape(
                    history_ids.shape[0], history_ids.shape[1], values.shape[1]
                )
                history_features[name] = self._tensor(history_values, torch.float32)
            yield Batch(
                user_features={"id": user_ids},
                item_features=item_features,
                history_features=history_features,
                history_mask=history_ids.ne(self.manifest.padding_id),
                labels=labels,
                metadata={
                    "dataset": self.dataset_name,
                    "dataset_version": self.manifest.version,
                    "dataset_fingerprint": self.manifest.fingerprint,
                    "split": canonical_split,
                    "batch_index": batch_index,
                },
            )

    def get_data_seq(self, split: str):
        """Compatibility iterator for callers not yet migrated to `iter_batches`."""

        for batch in self.iter_batches(split):
            yield (
                dict(batch.user_features),
                dict(batch.item_features),
                dict(batch.history_features),
                batch.labels.reshape(-1, 1),
            )

    def get_data(self, split: str):
        """Compatibility iterator retaining the old combined ID field."""

        for batch in self.iter_batches(split):
            features = dict(batch.item_features)
            features["id"] = torch.cat(
                [batch.user_features["id"], batch.item_features["id"]], dim=1
            )
            yield features, dict(batch.history_features), batch.labels.reshape(-1, 1)

    def get_multi_modal(self) -> Mapping[str, np.ndarray]:
        return {name: values for name, values in self._modalities.items()}


__all__ = [
    "NamedArrayDatasetLoader",
    "aggregate_file_digest",
    "load_manifest",
    "save_manifest",
    "sha256_file",
    "source_fingerprint",
]

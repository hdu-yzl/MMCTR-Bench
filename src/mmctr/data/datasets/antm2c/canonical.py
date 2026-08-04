"""Restartable full AntM2C conversion and sharded canonical loader."""

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from mmctr.core import Batch, ContractError
from mmctr.data.datasets.arrays import load_manifest, save_manifest, sha256_file
from mmctr.data.manifest import DatasetManifest, SplitStatistics

from .array_store import LEGACY_USER_TEXT_CONTEXT
from .benchmark import LegacyAntM2CLayout, discover_legacy_records


CANONICAL_SCHEMA_VERSION = 1
CANONICAL_STORAGE_FORMAT = "sharded-named-npy-v1"
CONTEXT_INDICES = {
    "service_text": 0,
    "query_text": 1,
    "bill_text": 2,
    "entity_text": 3,
    "time_context": 5,
}


@dataclass(frozen=True)
class CanonicalConversionStatus:
    """Observable state returned after a complete or deliberately bounded conversion run."""

    complete: bool
    output_dir: Path
    processed_source_files: int
    manifest: Optional[DatasetManifest] = None


def _atomic_json(path: Path, values: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(values, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def _source_identity(
    source_dir: Path, records: Mapping[str, Sequence[Path]]
) -> Tuple[str, Mapping[str, Any]]:
    item_files = {name: source_dir / name for name in ("text_feature.npy", "image_feature.npy")}
    if any(not path.is_file() for path in item_files.values()):
        raise ContractError("legacy AntM2C item feature stores are missing")
    payload: Dict[str, Any] = {
        "record_files": {
            "{}:{}".format(split, path.name): {
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for split, paths in records.items()
            for path in paths
        },
        "item_sha256": {name: sha256_file(path) for name, path in item_files.items()},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload


def _new_state(
    source_fingerprint: str,
    source_identity: Mapping[str, Any],
    layout: LegacyAntM2CLayout,
    shard_size: int,
    read_batch_size: int,
) -> Dict[str, Any]:
    return {
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "storage_format": CANONICAL_STORAGE_FORMAT,
        "source_fingerprint": source_fingerprint,
        "source_identity": source_identity,
        "layout": layout.to_dict(),
        "conflict_policy": "shared-item-store-wins",
        "shard_size": shard_size,
        "read_batch_size": read_batch_size,
        "files": {},
        "shards": {"train": [], "val": [], "test": []},
    }


def _validate_resume_state(
    state: Mapping[str, Any],
    source_fingerprint: str,
    layout: LegacyAntM2CLayout,
    shard_size: int,
    read_batch_size: int,
) -> None:
    expected = (
        CANONICAL_SCHEMA_VERSION,
        CANONICAL_STORAGE_FORMAT,
        source_fingerprint,
        layout.to_dict(),
        "shared-item-store-wins",
        shard_size,
        read_batch_size,
    )
    actual = (
        state.get("schema_version"),
        state.get("storage_format"),
        state.get("source_fingerprint"),
        state.get("layout"),
        state.get("conflict_policy"),
        state.get("shard_size"),
        state.get("read_batch_size"),
    )
    if actual != expected:
        raise ContractError("incomplete AntM2C conversion used different inputs or options")


def _copy_item_store(source_dir: Path, stage: Path) -> None:
    feature_dir = stage / "items" / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    for source_name, target_name in (
        ("text_feature.npy", "text.npy"),
        ("image_feature.npy", "image.npy"),
    ):
        source = source_dir / source_name
        target = feature_dir / target_name
        if target.is_file() and sha256_file(target) == sha256_file(source):
            continue
        temporary = target.with_suffix(target.suffix + ".tmp")
        shutil.copyfile(str(source), str(temporary))
        os.replace(str(temporary), str(target))


def _feature_description(layout: LegacyAntM2CLayout) -> Mapping[str, Any]:
    import tensorflow as tf  # type: ignore[import-untyped]

    return {
        "text_features": tf.io.FixedLenFeature([layout.text_dimension * 6], tf.float32),
        "image_features": tf.io.FixedLenFeature([layout.image_dimension], tf.float32),
        "id_feature": tf.io.FixedLenFeature([2], tf.int64),
        "label": tf.io.FixedLenFeature([1], tf.float32),
        "domain": tf.io.FixedLenFeature([1], tf.float32),
        "user_seq": tf.io.FixedLenFeature([layout.sequence_length], tf.int64),
    }


def _item_rows(values: np.ndarray, row_count: int, minimum_item_id: int) -> np.ndarray:
    ids = np.asarray(values, dtype=np.int64)
    rows = np.where(ids == 0, 0, ids - int(minimum_item_id) + 1)
    if bool((rows < 0).any()) or bool((rows >= row_count).any()):
        raise ContractError("legacy AntM2C record references an item outside the feature store")
    return rows


def _parse_records(
    raw_records: Sequence[bytes], layout: LegacyAntM2CLayout
) -> Mapping[str, np.ndarray]:
    import tensorflow as tf

    parsed = tf.io.parse_example(tf.constant(raw_records), _feature_description(layout))
    return {name: value.numpy() for name, value in parsed.items()}


def _empty_buffer() -> Dict[str, List[np.ndarray]]:
    names = (
        "event_id",
        "user_index",
        "item_index",
        "history_item_index",
        "labels",
        "domain",
    ) + tuple(CONTEXT_INDICES)
    return {name: [] for name in names}


def _buffer_arrays(buffer: Mapping[str, List[np.ndarray]]) -> Mapping[str, np.ndarray]:
    return {
        "event_id": np.asarray(buffer["event_id"], dtype=np.int64),
        "user_index": np.asarray(buffer["user_index"], dtype=np.int64),
        "item_index": np.asarray(buffer["item_index"], dtype=np.int64),
        "history_item_index": np.asarray(buffer["history_item_index"], dtype=np.int64),
        "labels": np.asarray(buffer["labels"], dtype=np.float32),
        "domain": np.asarray(buffer["domain"], dtype=np.float32).reshape(-1, 1),
        **{name: np.asarray(buffer[name], dtype=np.float32) for name in CONTEXT_INDICES},
    }


def _write_shard(
    stage: Path,
    split: str,
    shard_index: int,
    arrays: Mapping[str, np.ndarray],
) -> Mapping[str, Any]:
    relative = Path("splits") / split / "shard-{:06d}".format(shard_index)
    target = stage / relative
    temporary = target.with_name(target.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(str(temporary))
    temporary.mkdir(parents=True)
    files: Dict[str, Any] = {}
    for name, values in arrays.items():
        path = temporary / (name + ".npy")
        with path.open("wb") as stream:
            np.save(stream, values, allow_pickle=False)
        files[name] = {
            "sha256": sha256_file(path),
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "bytes": path.stat().st_size,
        }
    os.replace(str(temporary), str(target))
    return {
        "path": relative.as_posix(),
        "samples": int(arrays["labels"].shape[0]),
        "files": files,
    }


def _logical_record_digest_update(digest: Any, serialized: bytes) -> None:
    digest.update(len(serialized).to_bytes(8, byteorder="little"))
    digest.update(serialized)


def _process_source_file(
    path: Path,
    split: str,
    source_index: int,
    stage: Path,
    state: Dict[str, Any],
    text_store: np.ndarray,
    image_store: np.ndarray,
    layout: LegacyAntM2CLayout,
    shard_size: int,
    read_batch_size: int,
    state_path: Path,
) -> None:
    import tensorflow as tf

    key = "{}:{}".format(split, path.name)
    file_state = state["files"].setdefault(
        key,
        {
            "split": split,
            "name": path.name,
            "source_index": source_index,
            "scanned_rows": 0,
            "accepted_rows": 0,
            "title_store_conflicts": 0,
            "image_store_conflicts": 0,
            "complete": False,
            "source_record_sha256": None,
        },
    )
    if file_state["complete"]:
        return
    resume_rows = int(file_state["scanned_rows"])
    scanned = resume_rows
    accepted = int(file_state["accepted_rows"])
    title_conflicts = int(file_state["title_store_conflicts"])
    image_conflicts = int(file_state["image_store_conflicts"])
    buffer = _empty_buffer()
    digest = hashlib.sha256()
    raw_cursor = 0
    raw_dataset = tf.data.TFRecordDataset([str(path)]).batch(read_batch_size)
    for raw_batch in raw_dataset:
        serialized_batch = [bytes(value) for value in raw_batch.numpy().tolist()]
        batch_start = raw_cursor
        raw_cursor += len(serialized_batch)
        for serialized in serialized_batch:
            _logical_record_digest_update(digest, serialized)
        if batch_start + len(serialized_batch) <= resume_rows:
            continue
        start_offset = max(0, resume_rows - batch_start)
        active_records = serialized_batch[start_offset:]
        parsed = _parse_records(active_records, layout)
        packed_text = parsed["text_features"].reshape(-1, 6, layout.text_dimension)
        identifiers = parsed["id_feature"]
        history_ids = parsed["user_seq"]
        item_rows = _item_rows(identifiers[:, 1], len(text_store), layout.minimum_item_id)
        history_rows = _item_rows(history_ids, len(text_store), layout.minimum_item_id)
        title_match = np.all(
            np.isclose(packed_text[:, 4], text_store[item_rows], rtol=0.0, atol=1e-6),
            axis=1,
        )
        image_match = np.all(
            np.isclose(parsed["image_features"], image_store[item_rows], rtol=0.0, atol=1e-6),
            axis=1,
        )
        for local_index in range(len(active_records)):
            source_row = batch_start + start_offset + local_index
            scanned = source_row + 1
            if not bool(title_match[local_index]):
                title_conflicts += 1
            if not bool(image_match[local_index]):
                image_conflicts += 1
            buffer["event_id"].append(np.asarray([source_index, source_row]))
            buffer["user_index"].append(identifiers[local_index, 0])
            buffer["item_index"].append(item_rows[local_index])
            buffer["history_item_index"].append(history_rows[local_index])
            buffer["labels"].append(parsed["label"][local_index, 0])
            buffer["domain"].append(parsed["domain"][local_index, 0])
            for name, packed_index in CONTEXT_INDICES.items():
                buffer[name].append(packed_text[local_index, packed_index])
            accepted += 1
            if len(buffer["labels"]) == shard_size:
                shard = _write_shard(
                    stage,
                    split,
                    len(state["shards"][split]),
                    _buffer_arrays(buffer),
                )
                state["shards"][split].append(shard)
                buffer = _empty_buffer()
                file_state.update(
                    {
                        "scanned_rows": scanned,
                        "accepted_rows": accepted,
                        "title_store_conflicts": title_conflicts,
                        "image_store_conflicts": image_conflicts,
                    }
                )
                _atomic_json(state_path, state)
    if buffer["labels"]:
        shard = _write_shard(
            stage,
            split,
            len(state["shards"][split]),
            _buffer_arrays(buffer),
        )
        state["shards"][split].append(shard)
    file_state.update(
        {
            "scanned_rows": scanned,
            "accepted_rows": accepted,
            "title_store_conflicts": title_conflicts,
            "image_store_conflicts": image_conflicts,
            "complete": True,
            "source_record_sha256": digest.hexdigest(),
        }
    )
    _atomic_json(state_path, state)


def _aggregate_digest(values: Any) -> str:
    payload = json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _finalize_manifest(stage: Path, state: Mapping[str, Any]) -> DatasetManifest:
    split_statistics: Dict[str, SplitStatistics] = {}
    domains: Dict[str, Dict[str, int]] = {}
    for split in ("train", "val", "test"):
        users: Set[int] = set()
        items: Set[int] = set()
        positives = 0
        samples = 0
        domain_counts: Dict[str, int] = {}
        for shard in state["shards"][split]:
            root = stage / shard["path"]
            user_values = np.load(root / "user_index.npy", mmap_mode="r")
            item_values = np.load(root / "item_index.npy", mmap_mode="r")
            label_values = np.load(root / "labels.npy", mmap_mode="r")
            domain_values = np.load(root / "domain.npy", mmap_mode="r").reshape(-1)
            users.update(int(value) for value in np.unique(user_values))
            items.update(int(value) for value in np.unique(item_values))
            positives += int(np.asarray(label_values).sum())
            samples += len(label_values)
            unique_domains, counts = np.unique(domain_values, return_counts=True)
            for domain, count in zip(unique_domains, counts):
                key = str(float(domain))
                domain_counts[key] = domain_counts.get(key, 0) + int(count)
        split_statistics[split] = SplitStatistics(
            samples=samples,
            positives=positives,
            users=len(users),
            items=len(items),
            sha256=_aggregate_digest(state["shards"][split]),
        )
        domains[split] = domain_counts

    title_conflicts = {
        split: sum(
            int(file_state["title_store_conflicts"])
            for file_state in state["files"].values()
            if file_state["split"] == split
        )
        for split in ("train", "val", "test")
    }
    image_conflicts = {
        split: sum(
            int(file_state["image_store_conflicts"])
            for file_state in state["files"].values()
            if file_state["split"] == split
        )
        for split in ("train", "val", "test")
    }
    item_feature_sha256 = {
        name: sha256_file(stage / "items" / "features" / (name + ".npy"))
        for name in ("text", "image")
    }
    layout = state["layout"]
    manifest = DatasetManifest(
        name="antm2c",
        version="antm2c-canonical-v1",
        storage_format=CANONICAL_STORAGE_FORMAT,
        sequence_length=int(layout["sequence_length"]),
        padding_id=0,
        id_offsets={"user": 0, "item": int(layout["minimum_item_id"]) - 1},
        feature_dimensions={
            "id": 1,
            "text": int(layout["text_dimension"]),
            "image": int(layout["image_dimension"]),
            **{name: int(layout["text_dimension"]) for name in CONTEXT_INDICES},
            "domain": 1,
        },
        splits=split_statistics,
        source_fingerprint=str(state["source_fingerprint"]),
        metadata={
            "event_id": "[source_file_index, source_row] within each split",
            "source_record_sha256": {
                key: value["source_record_sha256"] for key, value in state["files"].items()
            },
            "output_shard_sha256": _aggregate_digest(state["shards"]),
            "item_feature_sha256": item_feature_sha256,
            "title_store_conflicts": title_conflicts,
            "image_store_conflicts": image_conflicts,
            "domain_counts": domains,
            "context_features": list(CONTEXT_INDICES),
        },
    )
    save_manifest(manifest, stage / "manifest.json")
    layout_payload = {
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "storage_format": CANONICAL_STORAGE_FORMAT,
        "splits": state["shards"],
        "item_features": {
            "text": {
                "path": "items/features/text.npy",
                "sha256": item_feature_sha256["text"],
            },
            "image": {
                "path": "items/features/image.npy",
                "sha256": item_feature_sha256["image"],
            },
        },
    }
    _atomic_json(stage / "layout.json", layout_payload)
    _atomic_json(stage / "conversion_audit.json", state)
    return manifest


def convert_legacy_to_canonical(
    source_dir: Path,
    output_dir: Path,
    shard_size: int = 8_192,
    read_batch_size: int = 256,
    layout: LegacyAntM2CLayout = LegacyAntM2CLayout(),
    max_source_files: Optional[int] = None,
) -> CanonicalConversionStatus:
    """Convert the authoritative legacy split files with shard-level restartability."""

    for name, value in (("shard_size", shard_size), ("read_batch_size", read_batch_size)):
        if isinstance(value, bool) or int(value) <= 0:
            raise ContractError("{} must be a positive integer".format(name))
    if max_source_files is not None and (
        isinstance(max_source_files, bool) or int(max_source_files) <= 0
    ):
        raise ContractError("max_source_files must be a positive integer")
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    records = discover_legacy_records(source_dir)
    source_fingerprint, source_identity = _source_identity(source_dir, records)
    if output_dir.exists():
        manifest = load_manifest(output_dir / "manifest.json")
        if manifest.source_fingerprint != source_fingerprint:
            raise ContractError("existing AntM2C canonical dataset has a different source")
        return CanonicalConversionStatus(True, output_dir, 0, manifest)

    stage = output_dir.with_name(output_dir.name + ".incomplete")
    stage.mkdir(parents=True, exist_ok=True)
    state_path = stage / "conversion_state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        _validate_resume_state(
            state, source_fingerprint, layout, int(shard_size), int(read_batch_size)
        )
    else:
        state = _new_state(
            source_fingerprint,
            source_identity,
            layout,
            int(shard_size),
            int(read_batch_size),
        )
        _atomic_json(state_path, state)
    _copy_item_store(source_dir, stage)
    text_store = np.load(stage / "items" / "features" / "text.npy", mmap_mode="r")
    image_store = np.load(stage / "items" / "features" / "image.npy", mmap_mode="r")
    if text_store.shape != (image_store.shape[0], layout.text_dimension):
        raise ContractError("legacy AntM2C text store has an invalid shape")
    if image_store.ndim != 2 or image_store.shape[1] != layout.image_dimension:
        raise ContractError("legacy AntM2C image store has an invalid shape")

    processed = 0
    for split in ("train", "val", "test"):
        for source_index, path in enumerate(records[split]):
            key = "{}:{}".format(split, path.name)
            if state["files"].get(key, {}).get("complete"):
                continue
            if max_source_files is not None and processed >= int(max_source_files):
                return CanonicalConversionStatus(False, output_dir, processed)
            _process_source_file(
                path,
                split,
                source_index,
                stage,
                state,
                text_store,
                image_store,
                layout,
                int(shard_size),
                int(read_batch_size),
                state_path,
            )
            processed += 1

    manifest = _finalize_manifest(stage, state)
    state_path.unlink()
    os.replace(str(stage), str(output_dir))
    return CanonicalConversionStatus(True, output_dir, processed, manifest)


def verify_canonical_dataset(input_dir: Path) -> DatasetManifest:
    """Verify every published item/shard array against its versioned layout metadata."""

    root = Path(input_dir)
    manifest = load_manifest(root / "manifest.json")
    layout = json.loads((root / "layout.json").read_text(encoding="utf-8"))
    if (
        manifest.name != "antm2c"
        or manifest.storage_format != CANONICAL_STORAGE_FORMAT
        or layout.get("schema_version") != CANONICAL_SCHEMA_VERSION
        or layout.get("storage_format") != CANONICAL_STORAGE_FORMAT
    ):
        raise ContractError("canonical AntM2C manifest and layout are incompatible")
    manifest_item_hashes = manifest.metadata.get("item_feature_sha256")
    if not isinstance(manifest_item_hashes, Mapping) or set(manifest_item_hashes) != set(
        layout["item_features"]
    ):
        raise ContractError("canonical AntM2C manifest is missing item feature hashes")
    for name, details in layout["item_features"].items():
        path = root / details["path"]
        expected = manifest_item_hashes[name]
        if (
            not isinstance(expected, str)
            or details.get("sha256") != expected
            or not path.is_file()
            or sha256_file(path) != expected
        ):
            raise ContractError("canonical AntM2C item feature hash mismatch: {}".format(path))
    if _aggregate_digest(layout["splits"]) != manifest.metadata["output_shard_sha256"]:
        raise ContractError("canonical AntM2C output shard metadata hash mismatch")
    for split in ("train", "val", "test"):
        samples = 0
        for shard in layout["splits"][split]:
            shard_root = root / shard["path"]
            if not shard_root.is_dir():
                raise ContractError("canonical AntM2C shard directory is missing")
            for name, details in shard["files"].items():
                path = shard_root / (name + ".npy")
                if not path.is_file() or path.stat().st_size != int(details["bytes"]):
                    raise ContractError(
                        "canonical AntM2C shard file size mismatch: {}".format(path)
                    )
                if sha256_file(path) != details["sha256"]:
                    raise ContractError("canonical AntM2C shard hash mismatch: {}".format(path))
                values = np.load(path, mmap_mode="r", allow_pickle=False)
                if list(values.shape) != details["shape"] or str(values.dtype) != details["dtype"]:
                    raise ContractError("canonical AntM2C shard array metadata mismatch")
            samples += int(shard["samples"])
        if samples != manifest.splits[split].samples:
            raise ContractError("canonical AntM2C split sample count does not match manifest")
    return manifest


class AntM2CCanonicalLoader:
    """Canonical Batch loader over restartably generated AntM2C shards."""

    dataset_name = "antm2c"

    def __init__(self, data_config: Mapping[str, Any], batch_size: int) -> None:
        if isinstance(batch_size, bool) or int(batch_size) <= 0:
            raise ContractError("batch_size must be a positive integer")
        try:
            self.root = Path(data_config["data_dir"])
        except KeyError as error:
            raise ContractError("canonical AntM2C config requires data_dir") from error
        self.batch_size = int(batch_size)
        self.manifest = load_manifest(self.root / "manifest.json")
        if self.manifest.name != self.dataset_name:
            raise ContractError("dataset manifest name does not match AntM2C loader")
        if self.manifest.storage_format != CANONICAL_STORAGE_FORMAT:
            raise ContractError("unsupported canonical AntM2C storage format")
        self.layout = json.loads((self.root / "layout.json").read_text(encoding="utf-8"))
        if self.layout.get("storage_format") != CANONICAL_STORAGE_FORMAT:
            raise ContractError("AntM2C layout format does not match manifest")
        self._modalities = {
            name: np.load(self.root / details["path"], mmap_mode="r")
            for name, details in self.layout["item_features"].items()
        }
        expected_rows = None
        for name, values in self._modalities.items():
            dimension = self.manifest.feature_dimensions[name]
            if values.ndim != 2 or values.shape[1] != dimension:
                raise ContractError("AntM2C item feature {!r} has an invalid shape".format(name))
            if expected_rows is not None and values.shape[0] != expected_rows:
                raise ContractError("AntM2C item feature stores have different row counts")
            expected_rows = values.shape[0]
            if bool(np.any(values[0] != 0)):
                raise ContractError("AntM2C item feature padding row must be zero")

    @staticmethod
    def _split_name(split: str) -> str:
        value = "val" if str(split).lower() == "validation" else str(split).lower()
        if value not in {"train", "val", "test"}:
            raise ContractError("split must be one of train, val, or test")
        return value

    @staticmethod
    def _tensor(values: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(np.array(values, copy=True), dtype=dtype)

    def iter_batches(self, split: str) -> Iterator[Batch]:
        canonical_split = self._split_name(split)
        item_offset = int(self.manifest.id_offsets["item"])
        batch_index = 0
        for shard in self.layout["splits"][canonical_split]:
            root = self.root / shard["path"]
            arrays = {
                name: np.load(root / (name + ".npy"), mmap_mode="r")
                for name in (
                    "event_id",
                    "user_index",
                    "item_index",
                    "history_item_index",
                    "labels",
                    "domain",
                )
            }
            contexts = {
                name: np.load(root / (name + ".npy"), mmap_mode="r") for name in CONTEXT_INDICES
            }
            sample_count = len(arrays["labels"])
            if any(
                len(values) != sample_count
                for values in list(arrays.values()) + list(contexts.values())
            ):
                raise ContractError("AntM2C shard arrays have inconsistent sample counts")
            for start in range(0, sample_count, self.batch_size):
                stop = min(start + self.batch_size, sample_count)
                item_rows = np.asarray(arrays["item_index"][start:stop], dtype=np.int64)
                history_rows = np.asarray(arrays["history_item_index"][start:stop], dtype=np.int64)
                item_ids = np.where(item_rows == 0, 0, item_rows + item_offset)
                history_ids = np.where(history_rows == 0, 0, history_rows + item_offset)
                context_features = {
                    name: self._tensor(values[start:stop], torch.float32)
                    for name, values in contexts.items()
                }
                context_features["domain"] = self._tensor(
                    arrays["domain"][start:stop], torch.float32
                )
                context_features["text"] = torch.cat(
                    [context_features[name] for name in LEGACY_USER_TEXT_CONTEXT], dim=-1
                )
                item_features: Dict[str, torch.Tensor] = {
                    "id": self._tensor(item_ids[:, None], torch.long)
                }
                history_features: Dict[str, torch.Tensor] = {
                    "id": self._tensor(history_ids, torch.long)
                }
                for name, values in self._modalities.items():
                    item_features[name] = self._tensor(values[item_rows], torch.float32)
                    history_features[name] = self._tensor(values[history_rows], torch.float32)
                yield Batch(
                    user_features={
                        "id": self._tensor(arrays["user_index"][start:stop, None], torch.long)
                    },
                    item_features=item_features,
                    history_features=history_features,
                    history_mask=history_features["id"].ne(self.manifest.padding_id),
                    labels=self._tensor(arrays["labels"][start:stop], torch.float32),
                    context_features=context_features,
                    metadata={
                        "dataset": self.dataset_name,
                        "dataset_version": self.manifest.version,
                        "dataset_fingerprint": self.manifest.fingerprint,
                        "split": canonical_split,
                        "batch_index": batch_index,
                        "event_id": self._tensor(arrays["event_id"][start:stop], torch.long),
                        "domain": self._tensor(
                            arrays["domain"][start:stop].reshape(-1), torch.float32
                        ),
                    },
                )
                batch_index += 1

    def get_data_seq(self, split: str) -> Iterable[Tuple[Any, ...]]:
        for batch in self.iter_batches(split):
            users = dict(batch.user_features)
            users.update(batch.context_features)
            yield (
                users,
                dict(batch.item_features),
                dict(batch.history_features),
                batch.labels[:, None],
            )

    def get_data(self, split: str) -> Iterable[Tuple[Any, ...]]:
        for batch in self.iter_batches(split):
            features = dict(batch.item_features)
            features["id"] = torch.cat(
                [batch.user_features["id"], batch.item_features["id"]], dim=1
            )
            yield features, dict(batch.history_features), batch.labels[:, None]

    def get_multi_modal(self) -> Mapping[str, np.ndarray]:
        return dict(self._modalities)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Convert legacy AntM2C TFRecords to canonical-v1")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=8_192)
    parser.add_argument("--read-batch-size", type=int, default=256)
    parser.add_argument("--max-source-files", type=int)
    args = parser.parse_args(argv)
    result = convert_legacy_to_canonical(
        args.source_dir,
        args.output_dir,
        shard_size=args.shard_size,
        read_batch_size=args.read_batch_size,
        max_source_files=args.max_source_files,
    )
    print(
        json.dumps(
            {
                "complete": result.complete,
                "output_dir": str(result.output_dir),
                "processed_source_files": result.processed_source_files,
                "manifest_fingerprint": (
                    result.manifest.fingerprint if result.manifest is not None else None
                ),
            },
            sort_keys=True,
        )
    )
    return 0 if result.complete else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANONICAL_SCHEMA_VERSION",
    "CANONICAL_STORAGE_FORMAT",
    "AntM2CCanonicalLoader",
    "CanonicalConversionStatus",
    "convert_legacy_to_canonical",
    "main",
    "verify_canonical_dataset",
]

"""Fair, reproducible format benchmarks for the legacy and canonical AntM2C loaders."""

import hashlib
import json
import os
import re
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, Sequence, Tuple

import numpy as np
import torch

from mmctr.core import Batch, ContractError
from mmctr.data.datasets.arrays import sha256_file
from mmctr.data.manifest import DatasetManifest, SplitStatistics

from .array_store import AntM2CArrayLoader, InteractionTable, write_array_store
from .item_store import ItemFeatureStore, ItemIndex


@dataclass(frozen=True)
class LegacyAntM2CLayout:
    """Shape and ID rules required to decode the frozen legacy TFRecord schema."""

    text_dimension: int = 768
    image_dimension: int = 512
    sequence_length: int = 5
    minimum_item_id: int = 67_625

    def __post_init__(self) -> None:
        for name in ("text_dimension", "image_dimension", "sequence_length"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) <= 0:
                raise ContractError("{} must be a positive integer".format(name))
        if isinstance(self.minimum_item_id, bool) or int(self.minimum_item_id) <= 0:
            raise ContractError("minimum_item_id must be a positive integer")

    def to_dict(self) -> Dict[str, int]:
        return {
            "text_dimension": int(self.text_dimension),
            "image_dimension": int(self.image_dimension),
            "sequence_length": int(self.sequence_length),
            "minimum_item_id": int(self.minimum_item_id),
        }


@dataclass(frozen=True)
class BenchmarkSliceManifest:
    """Identity and exact sampled-prefix counts for one prepared comparison."""

    source_fingerprint: str
    prefix_fingerprints: Mapping[str, str]
    split_samples: Mapping[str, int]
    scanned_samples: Mapping[str, int]
    skipped_title_mismatches: Mapping[str, int]
    skipped_image_mismatches: Mapping[str, int]
    samples_per_split: int
    layout: Mapping[str, int]
    source_tfrecord_bytes: int
    source_item_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "prefix_fingerprints", MappingProxyType(dict(self.prefix_fingerprints))
        )
        object.__setattr__(self, "split_samples", MappingProxyType(dict(self.split_samples)))
        object.__setattr__(self, "scanned_samples", MappingProxyType(dict(self.scanned_samples)))
        object.__setattr__(
            self,
            "skipped_title_mismatches",
            MappingProxyType(dict(self.skipped_title_mismatches)),
        )
        object.__setattr__(
            self,
            "skipped_image_mismatches",
            MappingProxyType(dict(self.skipped_image_mismatches)),
        )
        object.__setattr__(self, "layout", MappingProxyType(dict(self.layout)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_fingerprint": self.source_fingerprint,
            "prefix_fingerprints": dict(self.prefix_fingerprints),
            "split_samples": dict(self.split_samples),
            "scanned_samples": dict(self.scanned_samples),
            "skipped_title_mismatches": dict(self.skipped_title_mismatches),
            "skipped_image_mismatches": dict(self.skipped_image_mismatches),
            "samples_per_split": self.samples_per_split,
            "layout": dict(self.layout),
            "source_tfrecord_bytes": self.source_tfrecord_bytes,
            "source_item_bytes": self.source_item_bytes,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "BenchmarkSliceManifest":
        return cls(
            source_fingerprint=str(values["source_fingerprint"]),
            prefix_fingerprints={
                str(name): str(value) for name, value in values["prefix_fingerprints"].items()
            },
            split_samples={
                str(name): int(value) for name, value in values["split_samples"].items()
            },
            scanned_samples={
                str(name): int(value) for name, value in values["scanned_samples"].items()
            },
            skipped_title_mismatches={
                str(name): int(value) for name, value in values["skipped_title_mismatches"].items()
            },
            skipped_image_mismatches={
                str(name): int(value) for name, value in values["skipped_image_mismatches"].items()
            },
            samples_per_split=int(values["samples_per_split"]),
            layout={str(name): int(value) for name, value in values["layout"].items()},
            source_tfrecord_bytes=int(values["source_tfrecord_bytes"]),
            source_item_bytes=int(values["source_item_bytes"]),
        )


@dataclass(frozen=True)
class FormatMeasurement:
    samples: int
    batches: int
    elapsed_seconds: float
    samples_per_second: float
    checksum: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "samples": self.samples,
            "batches": self.batches,
            "elapsed_seconds": self.elapsed_seconds,
            "samples_per_second": self.samples_per_second,
            "checksum": self.checksum,
        }


@dataclass(frozen=True)
class FormatBenchmarkReport:
    legacy: FormatMeasurement
    named_arrays: FormatMeasurement
    legacy_interaction_bytes: int
    named_interaction_bytes: int
    shared_item_bytes: int
    source_tfrecord_bytes: int
    source_item_bytes: int
    batch_size: int
    repeats: int
    cache_protocol: str
    gpu_wait_ms: None = None
    gpu_wait_reason: str = "CPU loader benchmark does not include host-to-device transfer"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "legacy": self.legacy.to_dict(),
            "named_arrays": self.named_arrays.to_dict(),
            "bytes": {
                "legacy_interactions": self.legacy_interaction_bytes,
                "named_interactions": self.named_interaction_bytes,
                "shared_item_store": self.shared_item_bytes,
                "full_source_tfrecords": self.source_tfrecord_bytes,
                "full_source_item_store": self.source_item_bytes,
            },
            "batch_size": self.batch_size,
            "repeats": self.repeats,
            "cache_protocol": self.cache_protocol,
            "gpu_wait_ms": self.gpu_wait_ms,
            "gpu_wait_reason": self.gpu_wait_reason,
        }


def _item_rows(item_ids: np.ndarray, row_count: int, minimum_item_id: int) -> np.ndarray:
    values = np.asarray(item_ids, dtype=np.int64)
    rows = np.where(values == 0, 0, values - int(minimum_item_id) + 1)
    if bool((rows < 0).any()) or bool((rows >= row_count).any()):
        raise ContractError("legacy AntM2C record references an item outside the feature store")
    return rows


def iter_legacy_tfrecord_batches(
    record_paths: Sequence[Path],
    item_feature_dir: Path,
    batch_size: int,
    layout: LegacyAntM2CLayout = LegacyAntM2CLayout(),
) -> Iterator[Batch]:
    """Decode frozen packed TFRecords into the canonical, slice-free batch contract.

    Positional decoding is confined to this explicit legacy boundary. Returned target and
    history modalities are both gathered from the same item feature store.
    """

    if isinstance(batch_size, bool) or int(batch_size) <= 0:
        raise ContractError("batch_size must be a positive integer")
    paths = tuple(Path(path) for path in record_paths)
    if not paths or any(not path.is_file() for path in paths):
        raise ContractError("legacy benchmark requires existing TFRecord paths")
    item_feature_dir = Path(item_feature_dir)
    text_path = item_feature_dir / "text_feature.npy"
    if not text_path.is_file():
        text_path = item_feature_dir / "text.npy"
    image_path = item_feature_dir / "image_feature.npy"
    if not image_path.is_file():
        image_path = item_feature_dir / "image.npy"
    text_store = np.load(text_path, mmap_mode="r")
    image_store = np.load(image_path, mmap_mode="r")
    if text_store.shape[0] != image_store.shape[0]:
        raise ContractError("legacy AntM2C item stores have different row counts")
    if text_store.ndim != 2 or text_store.shape[1] != layout.text_dimension:
        raise ContractError("legacy AntM2C text item store has an invalid shape")
    if image_store.ndim != 2 or image_store.shape[1] != layout.image_dimension:
        raise ContractError("legacy AntM2C image item store has an invalid shape")

    import tensorflow as tf

    description = {
        "text_features": tf.io.FixedLenFeature([layout.text_dimension * 6], tf.float32),
        "image_features": tf.io.FixedLenFeature([layout.image_dimension], tf.float32),
        "id_feature": tf.io.FixedLenFeature([2], tf.int64),
        "label": tf.io.FixedLenFeature([1], tf.float32),
        "domain": tf.io.FixedLenFeature([1], tf.float32),
        "user_seq": tf.io.FixedLenFeature([layout.sequence_length], tf.int64),
    }

    @tf.autograph.experimental.do_not_convert
    def parse_record(raw_record):
        return tf.io.parse_single_example(raw_record, description)

    dataset = tf.data.TFRecordDataset([str(path) for path in paths])
    dataset = dataset.map(parse_record, num_parallel_calls=1).batch(int(batch_size))
    context_indices = {
        "service_text": 0,
        "query_text": 1,
        "bill_text": 2,
        "entity_text": 3,
        "time_context": 5,
    }
    for batch_index, values in enumerate(dataset):
        packed_text = values["text_features"].numpy().reshape(-1, 6, layout.text_dimension)
        packed_image = values["image_features"].numpy()
        identifiers = values["id_feature"].numpy()
        history_ids = values["user_seq"].numpy()
        item_rows = _item_rows(identifiers[:, 1], len(text_store), layout.minimum_item_id)
        history_rows = _item_rows(history_ids, len(text_store), layout.minimum_item_id)
        item_text = np.asarray(text_store[item_rows], dtype=np.float32)
        item_image = np.asarray(image_store[item_rows], dtype=np.float32)
        if not bool(np.allclose(packed_text[:, 4], item_text, rtol=0.0, atol=1e-6)):
            raise ContractError("packed target text disagrees with the shared item store")
        if not bool(np.allclose(packed_image, item_image, rtol=0.0, atol=1e-6)):
            raise ContractError("packed target image disagrees with the shared item store")
        context_features = {
            name: torch.as_tensor(packed_text[:, index].copy(), dtype=torch.float32)
            for name, index in context_indices.items()
        }
        context_features["domain"] = torch.as_tensor(
            values["domain"].numpy().copy(), dtype=torch.float32
        )
        context_features["text"] = torch.cat(
            [context_features[name] for name in context_indices], dim=-1
        )
        yield Batch(
            user_features={"id": torch.as_tensor(identifiers[:, :1].copy(), dtype=torch.long)},
            item_features={
                "id": torch.as_tensor(identifiers[:, 1:2].copy(), dtype=torch.long),
                "text": torch.as_tensor(item_text.copy(), dtype=torch.float32),
                "image": torch.as_tensor(item_image.copy(), dtype=torch.float32),
            },
            history_features={
                "id": torch.as_tensor(history_ids.copy(), dtype=torch.long),
                "text": torch.as_tensor(
                    np.asarray(text_store[history_rows], dtype=np.float32).copy(),
                    dtype=torch.float32,
                ),
                "image": torch.as_tensor(
                    np.asarray(image_store[history_rows], dtype=np.float32).copy(),
                    dtype=torch.float32,
                ),
            },
            history_mask=torch.as_tensor(history_ids != 0, dtype=torch.bool),
            labels=torch.as_tensor(values["label"].numpy().reshape(-1), dtype=torch.float32),
            context_features=context_features,
            metadata={
                "dataset": "antm2c",
                "storage_format": "legacy-tfrecord-packed-v1",
                "batch_index": batch_index,
                "domain": torch.as_tensor(
                    values["domain"].numpy().reshape(-1), dtype=torch.float32
                ),
            },
        )


def _natural_shard_key(path: Path) -> Tuple[int, str]:
    match = re.search(r"(\d+)\.tfrecord$", path.name)
    return (int(match.group(1)) if match else -1, path.name)


def discover_legacy_records(source_dir: Path) -> Mapping[str, Tuple[Path, ...]]:
    """Discover exactly the split files consumed by the frozen legacy training loader."""

    source_dir = Path(source_dir)
    train = tuple(sorted(source_dir.glob("train_shuffle_*.tfrecord"), key=_natural_shard_key))
    if not train:
        train = tuple(sorted(source_dir.glob("train[0-9]*.tfrecord"), key=_natural_shard_key))
    result = {
        "train": train,
        "val": tuple(sorted(source_dir.glob("val[0-9]*.tfrecord"), key=_natural_shard_key)),
        "test": tuple(sorted(source_dir.glob("test[0-9]*.tfrecord"), key=_natural_shard_key)),
    }
    missing = [name for name, paths in result.items() if not paths]
    if missing:
        raise ContractError("legacy AntM2C split TFRecords are missing: {}".format(missing))
    return MappingProxyType(result)


def _source_fingerprint(
    records: Mapping[str, Sequence[Path]], source_dir: Path
) -> Tuple[str, int, int]:
    record_metadata = {
        "{}:{}".format(split, path.name): {
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for split, paths in records.items()
        for path in paths
    }
    item_paths = (source_dir / "text_feature.npy", source_dir / "image_feature.npy")
    if any(not path.is_file() for path in item_paths):
        raise ContractError("legacy AntM2C item feature stores are missing")
    payload = {
        "records": record_metadata,
        "items": {path.name: sha256_file(path) for path in item_paths},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return (
        hashlib.sha256(encoded).hexdigest(),
        sum(path.stat().st_size for paths in records.values() for path in paths),
        sum(path.stat().st_size for path in item_paths),
    )


def _legacy_item_mismatch(
    serialized: bytes,
    text_store: np.ndarray,
    image_store: np.ndarray,
    layout: LegacyAntM2CLayout,
) -> str:
    import tensorflow as tf

    example = tf.train.Example.FromString(serialized)
    features = example.features.feature
    identifiers = tuple(features["id_feature"].int64_list.value)
    if len(identifiers) != 2:
        raise ContractError("legacy AntM2C id_feature does not contain two values")
    row = int(_item_rows(np.asarray([identifiers[1]]), len(text_store), layout.minimum_item_id)[0])
    packed_text = np.asarray(features["text_features"].float_list.value, dtype=np.float32)
    packed_image = np.asarray(features["image_features"].float_list.value, dtype=np.float32)
    if packed_text.shape != (layout.text_dimension * 6,):
        raise ContractError("legacy AntM2C packed text has an invalid shape")
    if packed_image.shape != (layout.image_dimension,):
        raise ContractError("legacy AntM2C packed image has an invalid shape")
    target_text = packed_text.reshape(6, layout.text_dimension)[4]
    if not bool(np.allclose(target_text, text_store[row], rtol=0.0, atol=1e-6)):
        return "title"
    if not bool(np.allclose(packed_image, image_store[row], rtol=0.0, atol=1e-6)):
        return "image"
    return ""


def _copy_prefix(
    paths: Sequence[Path],
    output: Path,
    sample_count: int,
    text_store: np.ndarray,
    image_store: np.ndarray,
    layout: LegacyAntM2CLayout,
) -> Tuple[int, str, int, int, int]:
    import tensorflow as tf

    digest = hashlib.sha256()
    copied = 0
    scanned = 0
    skipped_title = 0
    skipped_image = 0
    base_quota, remainder = divmod(sample_count, len(paths))
    with tf.io.TFRecordWriter(str(output)) as writer:
        for path_index, path in enumerate(paths):
            quota = base_quota + (1 if path_index < remainder else 0)
            accepted_from_path = 0
            if quota == 0:
                continue
            for serialized in tf.data.TFRecordDataset([str(path)]):
                value = bytes(serialized.numpy())
                scanned += 1
                mismatch = _legacy_item_mismatch(value, text_store, image_store, layout)
                if mismatch == "title":
                    skipped_title += 1
                    continue
                if mismatch == "image":
                    skipped_image += 1
                    continue
                writer.write(value)
                digest.update(len(value).to_bytes(8, byteorder="little"))
                digest.update(value)
                copied += 1
                accepted_from_path += 1
                if accepted_from_path == quota:
                    break
            if accepted_from_path != quota:
                raise ContractError(
                    "legacy AntM2C shard {} has too few consistent records".format(path.name)
                )
    if copied != sample_count:
        raise ContractError(
            "legacy AntM2C split has {} records, fewer than requested {}".format(
                copied, sample_count
            )
        )
    return copied, digest.hexdigest(), scanned, skipped_title, skipped_image


def _concatenate_batches(batches: Iterable[Batch], feature_group: str, name: str) -> np.ndarray:
    values = []
    for batch in batches:
        mapping = getattr(batch, feature_group)
        values.append(mapping[name].numpy())
    return np.concatenate(values, axis=0)


def _atomic_json(path: Path, values: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(values, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def prepare_benchmark_slice(
    source_dir: Path,
    output_dir: Path,
    samples_per_split: int = 2_048,
    layout: LegacyAntM2CLayout = LegacyAntM2CLayout(),
) -> BenchmarkSliceManifest:
    """Materialize identical real-record prefixes as TFRecord and named arrays."""

    if isinstance(samples_per_split, bool) or int(samples_per_split) <= 0:
        raise ContractError("samples_per_split must be a positive integer")
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    records = discover_legacy_records(source_dir)
    fingerprint, source_tfrecord_bytes, source_item_bytes = _source_fingerprint(records, source_dir)
    manifest_path = output_dir / "benchmark_slice.json"
    if output_dir.exists():
        if not manifest_path.is_file():
            raise ContractError("AntM2C benchmark output exists without its slice manifest")
        existing = BenchmarkSliceManifest.from_dict(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
        if (
            existing.source_fingerprint != fingerprint
            or existing.samples_per_split != int(samples_per_split)
            or dict(existing.layout) != layout.to_dict()
        ):
            raise ContractError("existing AntM2C benchmark used different inputs or options")
        return existing

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=str(output_dir.parent), prefix=".{}-".format(output_dir.name)
    ) as temporary_directory:
        stage = Path(temporary_directory) / "dataset"
        legacy_dir = stage / "legacy"
        legacy_dir.mkdir(parents=True)
        prefix_fingerprints = {}
        split_samples = {}
        scanned_samples = {}
        skipped_title_mismatches = {}
        skipped_image_mismatches = {}
        sample_paths = {}
        text_store = np.load(source_dir / "text_feature.npy", mmap_mode="r")
        image_store = np.load(source_dir / "image_feature.npy", mmap_mode="r")
        for split in ("train", "val", "test"):
            sample_path = legacy_dir / (split + ".tfrecord")
            count, prefix_digest, scanned, skipped_title, skipped_image = _copy_prefix(
                records[split],
                sample_path,
                int(samples_per_split),
                text_store,
                image_store,
                layout,
            )
            sample_paths[split] = sample_path
            split_samples[split] = count
            scanned_samples[split] = scanned
            skipped_title_mismatches[split] = skipped_title
            skipped_image_mismatches[split] = skipped_image
            prefix_fingerprints[split] = prefix_digest

        tables = {}
        split_statistics = {}
        for split, sample_path in sample_paths.items():
            batches = tuple(
                iter_legacy_tfrecord_batches(
                    [sample_path], source_dir, batch_size=256, layout=layout
                )
            )
            user_ids = _concatenate_batches(batches, "user_features", "id").reshape(-1)
            global_item_ids = _concatenate_batches(batches, "item_features", "id").reshape(-1)
            global_histories = _concatenate_batches(batches, "history_features", "id")
            item_indices = _item_rows(
                global_item_ids,
                np.load(source_dir / "text_feature.npy", mmap_mode="r").shape[0],
                layout.minimum_item_id,
            )
            history_indices = _item_rows(
                global_histories,
                np.load(source_dir / "text_feature.npy", mmap_mode="r").shape[0],
                layout.minimum_item_id,
            )
            labels = np.concatenate([batch.labels.numpy() for batch in batches], axis=0)
            contexts = {
                name: np.concatenate(
                    [batch.context_features[name].numpy() for batch in batches], axis=0
                )
                for name in batches[0].context_features
                if name != "text"
            }
            tables[split] = InteractionTable(
                event_ids=tuple(
                    "benchmark-v1:{}:{:08d}".format(split, index) for index in range(len(labels))
                ),
                user_indices=user_ids,
                item_indices=item_indices,
                history_item_indices=history_indices,
                labels=labels,
                context_features=contexts,
            )
            split_statistics[split] = SplitStatistics(
                samples=len(labels),
                positives=int(labels.sum()),
                users=int(np.unique(user_ids).size),
                items=int(np.unique(item_indices).size),
                sha256=prefix_fingerprints[split],
            )

        by_index = (None,) + tuple(
            range(layout.minimum_item_id, layout.minimum_item_id + len(text_store) - 1)
        )
        item_index = ItemIndex(
            {original_id: index for index, original_id in enumerate(by_index) if index},
            by_index,
        )
        item_store = ItemFeatureStore(item_index, {"text": text_store, "image": image_store})
        feature_dimensions = {
            "text": layout.text_dimension,
            "image": layout.image_dimension,
            "service_text": layout.text_dimension,
            "query_text": layout.text_dimension,
            "bill_text": layout.text_dimension,
            "entity_text": layout.text_dimension,
            "time_context": layout.text_dimension,
            "domain": 1,
        }
        dataset_manifest = DatasetManifest(
            name="antm2c",
            version="antm2c-benchmark-v1",
            storage_format="named-npy-candidate-v1",
            sequence_length=layout.sequence_length,
            padding_id=0,
            id_offsets={"user": 0, "item": layout.minimum_item_id - 1},
            feature_dimensions=feature_dimensions,
            splits=split_statistics,
            source_fingerprint=fingerprint,
            metadata={"purpose": "format-benchmark-only", "samples_per_split": samples_per_split},
        )
        write_array_store(stage / "named_arrays", dataset_manifest, tables, item_store)
        prepared = BenchmarkSliceManifest(
            source_fingerprint=fingerprint,
            prefix_fingerprints=prefix_fingerprints,
            split_samples=split_samples,
            scanned_samples=scanned_samples,
            skipped_title_mismatches=skipped_title_mismatches,
            skipped_image_mismatches=skipped_image_mismatches,
            samples_per_split=int(samples_per_split),
            layout=layout.to_dict(),
            source_tfrecord_bytes=source_tfrecord_bytes,
            source_item_bytes=source_item_bytes,
        )
        _atomic_json(stage / "benchmark_slice.json", prepared.to_dict())
        os.replace(str(stage), str(output_dir))
    return prepared


def _batch_checksum(batch: Batch) -> float:
    values = [batch.labels]
    for mapping in (
        batch.user_features,
        batch.item_features,
        batch.history_features,
        batch.context_features,
    ):
        values.extend(mapping[name] for name in sorted(mapping))
    return sum(float(value.double().sum().item()) for value in values)


def _measure_batches(factory: Callable[[], Iterable[Batch]], repeats: int) -> FormatMeasurement:
    durations = []
    expected = None
    for _ in range(repeats):
        started = time.perf_counter()
        samples = 0
        batches = 0
        checksum = 0.0
        for batch in factory():
            samples += batch.batch_size
            batches += 1
            checksum += _batch_checksum(batch)
        elapsed = time.perf_counter() - started
        result = samples, batches, checksum
        if expected is not None and result != expected:
            raise ContractError("format benchmark produced non-deterministic batch contents")
        expected = result
        durations.append(elapsed)
    if expected is None or expected[0] == 0:
        raise ContractError("format benchmark consumed no samples")
    elapsed = float(statistics.median(durations))
    return FormatMeasurement(
        samples=expected[0],
        batches=expected[1],
        elapsed_seconds=elapsed,
        samples_per_second=expected[0] / elapsed,
        checksum=expected[2],
    )


def _tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def benchmark_prepared_slice(
    output_dir: Path,
    batch_size: int = 256,
    repeats: int = 3,
) -> FormatBenchmarkReport:
    """Measure warm-cache end-to-end CPU batch construction for both prepared formats."""

    if isinstance(batch_size, bool) or int(batch_size) <= 0:
        raise ContractError("batch_size must be a positive integer")
    if isinstance(repeats, bool) or int(repeats) <= 0:
        raise ContractError("repeats must be a positive integer")
    output_dir = Path(output_dir)
    prepared = BenchmarkSliceManifest.from_dict(
        json.loads((output_dir / "benchmark_slice.json").read_text(encoding="utf-8"))
    )
    layout = LegacyAntM2CLayout(**dict(prepared.layout))
    legacy_paths = [
        output_dir / "legacy" / (split + ".tfrecord") for split in ("train", "val", "test")
    ]
    item_dir = output_dir / "named_arrays" / "items" / "features"

    def legacy_factory() -> Iterable[Batch]:
        return iter_legacy_tfrecord_batches(
            legacy_paths, item_dir, batch_size=int(batch_size), layout=layout
        )

    named_loader = AntM2CArrayLoader.from_directory(
        output_dir / "named_arrays", batch_size=int(batch_size)
    )

    def named_factory() -> Iterable[Batch]:
        for split in ("train", "val", "test"):
            yield from named_loader.iter_batches(split)

    _measure_batches(legacy_factory, 1)
    _measure_batches(named_factory, 1)
    legacy = _measure_batches(legacy_factory, int(repeats))
    named = _measure_batches(named_factory, int(repeats))
    if legacy.samples != named.samples or abs(legacy.checksum - named.checksum) > 1e-5:
        raise ContractError("legacy and named-array benchmark contents are not equivalent")
    named_root = output_dir / "named_arrays"
    report = FormatBenchmarkReport(
        legacy=legacy,
        named_arrays=named,
        legacy_interaction_bytes=_tree_bytes(output_dir / "legacy"),
        named_interaction_bytes=_tree_bytes(named_root / "splits"),
        shared_item_bytes=_tree_bytes(named_root / "items"),
        source_tfrecord_bytes=prepared.source_tfrecord_bytes,
        source_item_bytes=prepared.source_item_bytes,
        batch_size=int(batch_size),
        repeats=int(repeats),
        cache_protocol="one full warm-up pass per format, then median of complete CPU passes",
    )
    _atomic_json(output_dir / "benchmark_report.json", report.to_dict())
    return report


__all__ = [
    "BenchmarkSliceManifest",
    "FormatBenchmarkReport",
    "FormatMeasurement",
    "LegacyAntM2CLayout",
    "benchmark_prepared_slice",
    "discover_legacy_records",
    "iter_legacy_tfrecord_batches",
    "prepare_benchmark_slice",
]

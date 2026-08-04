"""Build a versioned canonical MicroLens dataset from the public parquet files."""

import os
import tempfile
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import pyarrow.parquet as pq

from mmctr.core import ContractError
from mmctr.data.datasets.arrays import (
    aggregate_file_digest,
    load_manifest,
    save_manifest,
    source_fingerprint,
)
from mmctr.data.manifest import DatasetManifest, SplitStatistics


MICROLENS_VERSION = "microlens-canonical-v1"
ITEM_ID_OFFSET = 1_000_000
SPLIT_NAMES = ("train", "val", "test")


def _embedding_dimension(parquet: pq.ParquetFile, column: str) -> int:
    batch = next(parquet.iter_batches(batch_size=1, columns=[column]))
    values = batch.column(0).to_pylist()
    if not values:
        raise ContractError("MicroLens item feature parquet is empty")
    dimension = len(values[0])
    if dimension <= 0:
        raise ContractError("MicroLens item embeddings must be non-empty")
    return dimension


def _write_item_store(source: Path, output: Path) -> Mapping[str, int]:
    parquet = pq.ParquetFile(source)
    item_count = parquet.metadata.num_rows
    dimensions = {
        "text": _embedding_dimension(parquet, "txt_emb_BERT"),
        "image": _embedding_dimension(parquet, "img_emb_CLIPRN50"),
    }
    output.mkdir(parents=True)
    item_ids = np.lib.format.open_memmap(
        output / "item_ids.npy", mode="w+", dtype=np.int64, shape=(item_count + 1,)
    )
    item_ids[0] = 0
    features = {
        name: np.lib.format.open_memmap(
            output / "{}.npy".format(name),
            mode="w+",
            dtype=np.float32,
            shape=(item_count + 1, dimension),
        )
        for name, dimension in dimensions.items()
    }
    for values in features.values():
        values[0] = 0.0
    seen = np.zeros(item_count + 1, dtype=np.bool_)
    columns = ["item_id", "txt_emb_BERT", "img_emb_CLIPRN50"]
    for batch in parquet.iter_batches(batch_size=4096, columns=columns):
        raw_ids = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        if bool((raw_ids <= 0).any()) or bool((raw_ids > item_count).any()):
            raise ContractError("MicroLens raw item IDs must be contiguous from one")
        if bool(seen[raw_ids].any()):
            raise ContractError("MicroLens item feature parquet contains duplicate item IDs")
        seen[raw_ids] = True
        item_ids[raw_ids] = raw_ids + ITEM_ID_OFFSET
        for name, column_index in (("text", 1), ("image", 2)):
            encoded = np.asarray(batch.column(column_index).to_pylist(), dtype=np.float32)
            if encoded.shape != (len(raw_ids), dimensions[name]):
                raise ContractError("MicroLens {} embeddings have inconsistent shapes".format(name))
            if not bool(np.isfinite(encoded).all()):
                raise ContractError("MicroLens {} embeddings contain NaN or Inf".format(name))
            features[name][raw_ids] = encoded
    if not bool(seen[1:].all()):
        raise ContractError("MicroLens item feature parquet has missing item IDs")
    for values in (item_ids,) + tuple(features.values()):
        values.flush()
    return dimensions


def _pad_and_shift(histories, sequence_length: int, item_count: int) -> np.ndarray:
    result = np.zeros((len(histories), sequence_length), dtype=np.int64)
    for row, raw_history in enumerate(histories):
        selected = list(raw_history)[-sequence_length:]
        if len(selected) < sequence_length:
            selected = [0] * (sequence_length - len(selected)) + selected
        raw_ids = np.asarray(selected, dtype=np.int64)
        if bool((raw_ids < 0).any()) or bool((raw_ids > item_count).any()):
            raise ContractError("MicroLens history references an unknown item")
        result[row] = np.where(raw_ids == 0, 0, raw_ids + ITEM_ID_OFFSET)
    return result


def _allocate_splits(
    output: Path, counts: Mapping[str, int], sequence_length: int
) -> Mapping[str, Mapping[str, np.memmap]]:
    arrays: Dict[str, Mapping[str, np.memmap]] = {}
    for name, count in counts.items():
        split_root = output / name
        split_root.mkdir(parents=True)
        arrays[name] = {
            "user_ids": np.lib.format.open_memmap(
                split_root / "user_ids.npy", mode="w+", dtype=np.int64, shape=(count,)
            ),
            "item_ids": np.lib.format.open_memmap(
                split_root / "item_ids.npy", mode="w+", dtype=np.int64, shape=(count,)
            ),
            "history_item_ids": np.lib.format.open_memmap(
                split_root / "history_item_ids.npy",
                mode="w+",
                dtype=np.int64,
                shape=(count, sequence_length),
            ),
            "labels": np.lib.format.open_memmap(
                split_root / "labels.npy", mode="w+", dtype=np.float32, shape=(count,)
            ),
        }
    return arrays


def _write_splits(
    source: Path,
    output: Path,
    sequence_length: int,
    seed: int,
    item_count: int,
) -> Mapping[str, SplitStatistics]:
    parquet = pq.ParquetFile(source)
    sample_count = parquet.metadata.num_rows
    train_count = int(sample_count * 0.8)
    val_boundary = int(sample_count * 0.9)
    counts = {
        "train": train_count,
        "val": val_boundary - train_count,
        "test": sample_count - val_boundary,
    }
    # Split complete source rows, including their already-built histories; rebuilding a
    # user history after random assignment would leak interactions across these splits.
    assignments = np.concatenate(
        [np.full(counts[name], index, dtype=np.uint8) for index, name in enumerate(SPLIT_NAMES)]
    )
    np.random.RandomState(seed).shuffle(assignments)
    arrays = _allocate_splits(output, counts, sequence_length)
    positions = {name: 0 for name in SPLIT_NAMES}
    positives = {name: 0 for name in SPLIT_NAMES}
    row_offset = 0
    columns = ["user_id", "item_seq", "item_id", "label"]
    for batch in parquet.iter_batches(batch_size=32768, columns=columns):
        batch_size = batch.num_rows
        codes = assignments[row_offset : row_offset + batch_size]
        users = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        histories = batch.column(1).to_pylist()
        items = batch.column(2).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        labels = batch.column(3).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        if bool((users <= 0).any()):
            raise ContractError("MicroLens user IDs must be positive")
        if bool((items <= 0).any()) or bool((items > item_count).any()):
            raise ContractError("MicroLens target references an unknown item")
        if not bool(np.isin(labels, (0.0, 1.0)).all()):
            raise ContractError("MicroLens labels must be binary")
        shifted_history = _pad_and_shift(histories, sequence_length, item_count)
        for code, name in enumerate(SPLIT_NAMES):
            selected = np.flatnonzero(codes == code)
            if not len(selected):
                continue
            start = positions[name]
            stop = start + len(selected)
            arrays[name]["user_ids"][start:stop] = users[selected]
            arrays[name]["item_ids"][start:stop] = items[selected] + ITEM_ID_OFFSET
            arrays[name]["history_item_ids"][start:stop] = shifted_history[selected]
            arrays[name]["labels"][start:stop] = labels[selected]
            positions[name] = stop
            positives[name] += int(labels[selected].sum())
        row_offset += batch_size
    if row_offset != sample_count or positions != counts:
        raise ContractError("MicroLens split writer did not consume the expected sample count")
    statistics = {}
    for name in SPLIT_NAMES:
        for values in arrays[name].values():
            values.flush()
        split_root = output / name
        files = {path.name: path for path in sorted(split_root.glob("*.npy"))}
        statistics[name] = SplitStatistics(
            samples=counts[name],
            positives=positives[name],
            users=int(np.unique(arrays[name]["user_ids"]).size),
            items=int(np.unique(arrays[name]["item_ids"]).size),
            sha256=aggregate_file_digest(files),
        )
    return statistics


def prepare_microlens(
    raw_dir: Path,
    output_dir: Path,
    sequence_length: int = 5,
    seed: int = 42,
) -> DatasetManifest:
    """Create or verify a canonical MicroLens dataset inside ``output_dir``."""

    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    if isinstance(sequence_length, bool) or int(sequence_length) <= 0:
        raise ContractError("MicroLens sequence_length must be positive")
    sources = {
        "item_feature.parquet": raw_dir / "item_feature.parquet",
        "train.parquet": raw_dir / "train.parquet",
    }
    missing = [name for name, path in sources.items() if not path.is_file()]
    if missing:
        raise ContractError("MicroLens raw files are missing: {}".format(missing))
    fingerprint, source_digests = source_fingerprint(sources)
    if output_dir.exists():
        manifest_path = output_dir / "manifest.json"
        if not manifest_path.is_file():
            raise ContractError("MicroLens output exists without a manifest")
        manifest = load_manifest(manifest_path)
        expected = {
            "seed": int(seed),
            "split_protocol": "deterministic-random-80-10-10-v1",
        }
        if (
            manifest.source_fingerprint != fingerprint
            or manifest.sequence_length != int(sequence_length)
            or any(manifest.metadata.get(key) != value for key, value in expected.items())
        ):
            raise ContractError("existing MicroLens output was built from different inputs")
        return manifest

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=str(output_dir.parent), prefix=".{}-".format(output_dir.name)
    ) as temporary_directory:
        stage = Path(temporary_directory) / "dataset"
        stage.mkdir()
        dimensions = _write_item_store(sources["item_feature.parquet"], stage / "item_features")
        item_count = pq.ParquetFile(sources["item_feature.parquet"]).metadata.num_rows
        statistics = _write_splits(
            sources["train.parquet"],
            stage / "splits",
            int(sequence_length),
            int(seed),
            item_count,
        )
        manifest = DatasetManifest(
            name="microlens",
            version=MICROLENS_VERSION,
            storage_format="named-npy-v1",
            sequence_length=int(sequence_length),
            padding_id=0,
            id_offsets={"user": 0, "item": ITEM_ID_OFFSET},
            feature_dimensions={"id": 1, **dimensions},
            splits=statistics,
            source_fingerprint=fingerprint,
            metadata={
                "seed": int(seed),
                "split_protocol": "deterministic-random-80-10-10-v1",
                "source_sha256": dict(source_digests),
                "raw_item_id_min": 1,
                "raw_item_id_max": item_count,
            },
        )
        save_manifest(manifest, stage / "manifest.json")
        os.replace(str(stage), str(output_dir))
    return manifest


__all__ = ["MICROLENS_VERSION", "prepare_microlens"]

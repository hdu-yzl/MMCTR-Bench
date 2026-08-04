"""Build leakage-safe CTR samples from the official TikTok split files."""

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Set, Tuple

import numpy as np

from mmctr.core import ContractError
from mmctr.data.datasets.arrays import (
    aggregate_file_digest,
    load_manifest,
    save_manifest,
    source_fingerprint,
)
from mmctr.data.manifest import DatasetManifest, SplitStatistics


TIKTOK_VERSION = "tiktok-canonical-v1"
SPLIT_NAMES = ("train", "val", "test")
MODALITY_FILES = {
    "text": "text_feat.npy",
    "image": "image_feat.npy",
    "audio": "audio_feat.npy",
}


def _load_sequences(path: Path) -> Mapping[int, Tuple[int, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ContractError("TikTok split JSON must contain a user mapping")
    result: Dict[int, Tuple[int, ...]] = {}
    for raw_user, raw_items in payload.items():
        try:
            user_id = int(raw_user)
        except (TypeError, ValueError) as error:
            raise ContractError("TikTok user IDs must be integers") from error
        if user_id < 0 or not isinstance(raw_items, list):
            raise ContractError("TikTok users must be non-negative and map to item lists")
        try:
            items = tuple(int(item) for item in raw_items)
        except (TypeError, ValueError) as error:
            raise ContractError("TikTok item IDs must be integers") from error
        result[user_id] = items
    return result


def _write_item_store(
    sources: Mapping[str, Path], output: Path, item_id_offset: int
) -> Tuple[int, Mapping[str, int]]:
    arrays = {name: np.load(path, mmap_mode="r") for name, path in sources.items()}
    row_counts = {values.shape[0] for values in arrays.values()}
    if len(row_counts) != 1:
        raise ContractError("TikTok modality arrays must have the same number of rows")
    row_count = row_counts.pop()
    if row_count < 2:
        raise ContractError("TikTok modality arrays require padding plus item rows")
    item_count = row_count - 1
    dimensions = {}
    output.mkdir(parents=True)
    item_ids = np.lib.format.open_memmap(
        output / "item_ids.npy", mode="w+", dtype=np.int64, shape=(row_count,)
    )
    item_ids[0] = 0
    item_ids[1:] = np.arange(item_id_offset, item_id_offset + item_count, dtype=np.int64)
    item_ids.flush()
    for name, source in arrays.items():
        if source.ndim != 2 or source.shape[1] <= 0:
            raise ContractError("TikTok modality {!r} must be a rank-two array".format(name))
        if not np.issubdtype(source.dtype, np.floating):
            raise ContractError("TikTok modality {!r} must be floating point".format(name))
        if not bool(np.all(source[0] == 0)):
            raise ContractError(
                "TikTok modality {!r} must reserve row zero for padding".format(name)
            )
        destination = np.lib.format.open_memmap(
            output / "{}.npy".format(name),
            mode="w+",
            dtype=np.float32,
            shape=source.shape,
        )
        for start in range(0, row_count, 4096):
            stop = min(start + 4096, row_count)
            values = np.asarray(source[start:stop], dtype=np.float32)
            if not bool(np.isfinite(values).all()):
                raise ContractError("TikTok modality {!r} contains NaN or Inf".format(name))
            destination[start:stop] = values
        destination.flush()
        dimensions[name] = int(source.shape[1])
    return item_count, dimensions


def _full_positive_items(
    splits: Mapping[str, Mapping[int, Tuple[int, ...]]],
) -> Mapping[int, Set[int]]:
    result: Dict[int, Set[int]] = {}
    for values in splits.values():
        for user_id, items in values.items():
            result.setdefault(user_id, set()).update(items)
    return result


def _left_padded_history(
    history: Sequence[int], sequence_length: int, item_id_offset: int
) -> np.ndarray:
    selected = list(history[-sequence_length:])
    result = np.zeros(sequence_length, dtype=np.int64)
    if selected:
        result[-len(selected) :] = np.asarray(selected, dtype=np.int64) + item_id_offset
    return result


def _sample_negatives(
    random_state: np.random.RandomState,
    item_count: int,
    positive_items: Set[int],
    count: int,
) -> np.ndarray:
    available = item_count - len(positive_items)
    if available < count:
        raise ContractError("TikTok user has too few unobserved items for negative sampling")
    if count == 0:
        return np.empty(0, dtype=np.int64)
    candidates = np.asarray(
        [item for item in range(item_count) if item not in positive_items], dtype=np.int64
    )
    return random_state.choice(candidates, size=count, replace=False)


def _allocate_split(output: Path, sample_count: int, sequence_length: int):
    output.mkdir(parents=True)
    return {
        "user_ids": np.lib.format.open_memmap(
            output / "user_ids.npy", mode="w+", dtype=np.int64, shape=(sample_count,)
        ),
        "item_ids": np.lib.format.open_memmap(
            output / "item_ids.npy", mode="w+", dtype=np.int64, shape=(sample_count,)
        ),
        "history_item_ids": np.lib.format.open_memmap(
            output / "history_item_ids.npy",
            mode="w+",
            dtype=np.int64,
            shape=(sample_count, sequence_length),
        ),
        "labels": np.lib.format.open_memmap(
            output / "labels.npy", mode="w+", dtype=np.float32, shape=(sample_count,)
        ),
    }


def _write_splits(
    splits: Mapping[str, Mapping[int, Tuple[int, ...]]],
    output: Path,
    sequence_length: int,
    negative_ratio: int,
    seed: int,
    item_count: int,
    item_id_offset: int,
) -> Mapping[str, SplitStatistics]:
    positive_items = _full_positive_items(splits)
    for user_items in positive_items.values():
        if any(item < 0 or item >= item_count for item in user_items):
            raise ContractError("TikTok split references an item outside the modality store")
    histories: Dict[int, List[int]] = {user_id: [] for user_id in positive_items}
    random_state = np.random.RandomState(seed)
    statistics = {}
    for split_name in SPLIT_NAMES:
        positive_count = sum(len(items) for items in splits[split_name].values())
        sample_count = positive_count * (negative_ratio + 1)
        arrays = _allocate_split(output / split_name, sample_count, sequence_length)
        position = 0
        split_users = set()
        split_items: Set[int] = set()
        for user_id in sorted(splits[split_name]):
            split_users.add(user_id)
            history = histories.setdefault(user_id, [])
            for positive_item in splits[split_name][user_id]:
                # Snapshot before append: each target sees only earlier official events,
                # while state intentionally carries forward from train to val to test.
                history_values = _left_padded_history(history, sequence_length, item_id_offset)
                negatives = _sample_negatives(
                    random_state,
                    item_count,
                    positive_items[user_id],
                    negative_ratio,
                )
                targets = np.concatenate([np.asarray([positive_item], dtype=np.int64), negatives])
                count = len(targets)
                stop = position + count
                arrays["user_ids"][position:stop] = user_id + 1
                arrays["item_ids"][position:stop] = targets + item_id_offset
                arrays["history_item_ids"][position:stop] = history_values
                arrays["labels"][position:stop] = 0.0
                arrays["labels"][position] = 1.0
                split_items.update(int(item) for item in targets)
                position = stop
                history.append(positive_item)
        if position != sample_count:
            raise ContractError("TikTok split writer produced an unexpected sample count")
        for values in arrays.values():
            values.flush()
        split_root = output / split_name
        files = {path.name: path for path in sorted(split_root.glob("*.npy"))}
        statistics[split_name] = SplitStatistics(
            samples=sample_count,
            positives=positive_count,
            users=len(split_users),
            items=len(split_items),
            sha256=aggregate_file_digest(files),
        )
    return statistics


def prepare_tiktok(
    raw_dir: Path,
    output_dir: Path,
    sequence_length: int = 5,
    negative_ratio: int = 5,
    seed: int = 42,
) -> DatasetManifest:
    """Create or verify leakage-safe TikTok CTR splits inside ``output_dir``."""

    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    if isinstance(sequence_length, bool) or int(sequence_length) <= 0:
        raise ContractError("TikTok sequence_length must be positive")
    if isinstance(negative_ratio, bool) or int(negative_ratio) < 0:
        raise ContractError("TikTok negative_ratio must be non-negative")
    sources = {
        **{"{}.json".format(name): raw_dir / "{}.json".format(name) for name in SPLIT_NAMES},
        **{filename: raw_dir / filename for filename in MODALITY_FILES.values()},
    }
    missing = [name for name, path in sources.items() if not path.is_file()]
    if missing:
        raise ContractError("TikTok raw files are missing: {}".format(missing))
    fingerprint, source_digests = source_fingerprint(sources)
    expected_metadata = {
        "seed": int(seed),
        "negative_ratio": int(negative_ratio),
        "split_protocol": "official-split-causal-prefix-negative-v1",
    }
    if output_dir.exists():
        manifest_path = output_dir / "manifest.json"
        if not manifest_path.is_file():
            raise ContractError("TikTok output exists without a manifest")
        manifest = load_manifest(manifest_path)
        if (
            manifest.source_fingerprint != fingerprint
            or manifest.sequence_length != int(sequence_length)
            or any(manifest.metadata.get(key) != value for key, value in expected_metadata.items())
        ):
            raise ContractError("existing TikTok output was built from different inputs")
        return manifest

    splits = {name: _load_sequences(sources["{}.json".format(name)]) for name in SPLIT_NAMES}
    user_ids = set().union(*(set(values) for values in splits.values()))
    if not user_ids:
        raise ContractError("TikTok split files contain no users")
    max_user_id = max(user_ids)
    item_id_offset = max_user_id + 2
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=str(output_dir.parent), prefix=".{}-".format(output_dir.name)
    ) as temporary_directory:
        stage = Path(temporary_directory) / "dataset"
        stage.mkdir()
        modality_sources = {name: sources[filename] for name, filename in MODALITY_FILES.items()}
        item_count, dimensions = _write_item_store(
            modality_sources, stage / "item_features", item_id_offset
        )
        statistics = _write_splits(
            splits,
            stage / "splits",
            int(sequence_length),
            int(negative_ratio),
            int(seed),
            item_count,
            item_id_offset,
        )
        manifest = DatasetManifest(
            name="tiktok",
            version=TIKTOK_VERSION,
            storage_format="named-npy-v1",
            sequence_length=int(sequence_length),
            padding_id=0,
            id_offsets={"user": 1, "item": item_id_offset},
            feature_dimensions={"id": 1, **dimensions},
            splits=statistics,
            source_fingerprint=fingerprint,
            metadata={
                **expected_metadata,
                "source_sha256": dict(source_digests),
                "raw_user_id_min": min(user_ids),
                "raw_user_id_max": max_user_id,
                "raw_item_id_min": 0,
                "raw_item_id_max": item_count - 1,
            },
        )
        save_manifest(manifest, stage / "manifest.json")
        os.replace(str(stage), str(output_dir))
    return manifest


__all__ = ["TIKTOK_VERSION", "prepare_tiktok"]

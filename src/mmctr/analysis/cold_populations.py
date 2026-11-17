"""Train-only support counts and fixed validation/test cold populations."""

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from mmctr.core import ContractError
from .benchmark_protocols import cold_selection, population_summary

COLD_PROTOCOL = "paper-mixed-graph-oov-v2"


def _field_arrays(root, split, field):
    layout_path = root / "layout.json"
    if layout_path.exists():
        layout = json.loads(layout_path.read_text())
        if layout.get("storage_format") == "sharded-named-npy-v1":
            name = {
                "user_ids": "user_index",
                "item_ids": "item_index",
                "history_ids": "history_item_index",
            }.get(field, field)
            # Exactly the loader's layout order; item offsets do not affect support counts.
            for shard in layout["splits"][split]:
                yield np.load(root / shard["path"] / (name + ".npy"), mmap_mode="r")
            return
    name = "history_item_ids" if field == "history_ids" else field
    yield np.load(root / "splits" / split / (name + ".npy"), mmap_mode="r")


def _load_field(root, split, field):
    return np.concatenate(
        [np.asarray(array).reshape(-1) for array in _field_arrays(root, split, field)]
    )


def training_exposure(root):
    """Candidate frequencies plus graph exposure, including valid history item IDs.

    History exposure is a seen-set, not extra candidate interactions: contexts are
    repeated for sampled negatives and must not inflate few-shot frequencies.
    """
    root = Path(root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    sharded = manifest.get("name") == "antm2c"
    counts = {}
    for field in ("user_ids", "item_ids"):
        support = np.zeros(0, dtype=np.int64)
        minimum = 0 if field == "user_ids" and sharded else 1
        for array in _field_arrays(root, "train", field):
            values = np.asarray(array).reshape(-1)
            if not np.issubdtype(values.dtype, np.integer) or np.any(values < minimum):
                raise ContractError("training target IDs are outside the dataset domain")
            chunk = np.bincount(values)
            if len(chunk) > len(support):
                support = np.pad(support, (0, len(chunk) - len(support)))
            support[: len(chunk)] += chunk
        counts[field] = support
    seen = {name: value > 0 for name, value in counts.items()}
    for array in _field_arrays(root, "train", "history_ids"):
        values = np.asarray(array)
        if not np.issubdtype(values.dtype, np.integer) or np.any(values < 0):
            raise ContractError("training history IDs must be nonnegative integers")
        ids = np.unique(values)
        ids = ids[ids != 0]
        if len(ids):
            size = int(ids[-1]) + 1
            if size > len(seen["item_ids"]):
                seen["item_ids"] = np.pad(seen["item_ids"], (0, size - len(seen["item_ids"])))
            seen["item_ids"][ids] = True
    # AntM2C stores compact item rows; Batch IDs include the disjoint field offset.
    offset = int(manifest.get("id_offsets", {}).get("item", 0)) if sharded else 0
    return counts, seen, offset


def build_cold_populations(
    root, target: str, regime: str, exposure=None, *, splits=("val", "test")
) -> Tuple[Dict, Dict]:
    """Return selection masks and auditable populations without changing data."""
    root = Path(root)
    supports, seen, _ = training_exposure(root) if exposure is None else exposure
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    allow_zero_user = manifest.get("name") == "antm2c"
    if regime == "zero":
        supports = seen
    masks, summaries = {}, {}
    for split in splits:
        counts = {}
        labels = _load_field(root, split, "labels").reshape(-1)
        for field, support in supports.items():
            minimum = 0 if field == "user_ids" and allow_zero_user else 1
            values = _load_field(root, split, field).reshape(-1)
            if len(values) != len(labels) or np.any(values < minimum):
                raise ContractError("evaluation IDs must align with labels and the dataset domain")
            valid = values < len(support)
            result = np.zeros(len(values), dtype=np.int64)
            result[valid] = support[values[valid]]
            counts[field] = result
        masks[split] = cold_selection(counts["user_ids"], counts["item_ids"], target, regime)
        summaries[split] = population_summary(labels[masks[split]])
    return masks, summaries

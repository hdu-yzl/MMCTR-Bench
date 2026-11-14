"""Publish a history-length variant without modifying its parent or refitting features."""

import os
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from mmctr.core import ContractError
from mmctr.data.datasets.arrays import aggregate_file_digest, save_manifest, sha256_file
from mmctr.data.manifest import DatasetManifest
from .preprocessing import _read_histories, validate_ebnerd


def resize_history(source_dir, raw_dir, output_dir, sequence_length=20):
    source, raw, output = [Path(p).resolve() for p in (source_dir, raw_dir, output_dir)]
    if (
        isinstance(sequence_length, bool)
        or not isinstance(sequence_length, int)
        or sequence_length < 1
    ):
        raise ContractError("sequence length must be a positive integer")
    if source == output or source in output.parents or output in source.parents:
        raise ContractError("history source/output must be separate directories")
    parent = validate_ebnerd(source)
    for name in ("train", "validation"):
        relative = "ebnerd_small/" + name + "/history.parquet"
        if sha256_file(raw / relative) != parent.metadata["source_sha256"][relative]:
            raise ContractError("history source hash mismatch")
    identity = {"parent_fingerprint": parent.fingerprint, "sequence_length": sequence_length}
    if output.exists():
        manifest = validate_ebnerd(output)
        if manifest.metadata.get("history_variant") != identity:
            raise ContractError("existing history variant has different inputs")
        return manifest
    users = np.load(source / "mappings/original_user_ids.npy")
    items = np.load(source / "item_features/original_item_ids.npy")
    item_ids = np.load(source / "item_features/item_ids.npy")
    mapping = dict(zip(items.tolist(), item_ids.tolist()))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(output.parent), prefix=".ebnerd-history-") as tmp:
        stage = Path(tmp) / "dataset"
        shutil.copytree(source, stage)
        stats = dict(parent.splits)
        for official, splits in (("train", ("train", "val")), ("validation", ("test",))):
            first = min(parent.metadata["time_bounds_us"][s][0] for s in splits)
            histories, _ = _read_histories(
                raw / "ebnerd_small" / official / "history.parquet", first, sequence_length
            )
            table = np.zeros((len(users), sequence_length), dtype=np.int64)
            for user_index, original in enumerate(users):
                if user_index == 0 or int(original) not in histories:
                    continue
                selected = histories[int(original)]
                if selected:
                    table[user_index, -len(selected) :] = [mapping[i] for i in selected]
            for split in splits:
                path = stage / "splits" / split
                encoded = np.load(source / "splits" / split / "user_ids.npy", mmap_mode="r")
                dest = np.lib.format.open_memmap(
                    path / "history_item_ids.npy",
                    mode="w+",
                    dtype=np.int64,
                    shape=(len(encoded), sequence_length),
                )
                old = np.load(source / "splits" / split / "history_item_ids.npy", mmap_mode="r")
                shared = min(sequence_length, parent.sequence_length)
                for start in range(0, len(encoded), 8192):
                    stop = min(len(encoded), start + 8192)
                    dest[start:stop] = table[encoded[start:stop]]
                    if not np.array_equal(dest[start:stop, -shared:], old[start:stop, -shared:]):
                        raise ContractError("history variant changed the shared suffix")
                dest.flush()
                stats[split] = replace(
                    stats[split],
                    sha256=aggregate_file_digest({p.name: p for p in sorted(path.glob("*.npy"))}),
                )
        metadata = dict(parent.metadata)
        metadata["history_variant"] = identity
        metadata["output_sha256"] = {
            p.relative_to(stage).as_posix(): sha256_file(p) for p in sorted(stage.rglob("*.npy"))
        }
        for name, digest in parent.metadata["output_sha256"].items():
            if (
                not name.endswith("/history_item_ids.npy")
                and metadata["output_sha256"][name] != digest
            ):
                raise ContractError("non-history array changed")
        payload = parent.to_dict(include_fingerprint=False)
        payload.update(sequence_length=sequence_length, splits=stats, metadata=metadata)
        result = DatasetManifest(**payload)
        save_manifest(result, stage / "manifest.json")
        os.replace(str(stage), str(output))
    return result

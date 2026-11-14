"""Deterministic whole-impression subsampling of an audited EB-NeRD test split."""

import math
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from mmctr.core import ContractError
from mmctr.data.datasets.arrays import aggregate_file_digest, save_manifest, sha256_file
from mmctr.data.manifest import DatasetManifest, SplitStatistics

from .preprocessing import VERSION, validate_ebnerd

SAMPLED_VERSION = "ebnerd-small-canonical-v2"


def sample_ebnerd_test(
    source_dir: Path,
    output_dir: Path,
    fraction: float = 0.1,
    seed: int = 2025,
) -> DatasetManifest:
    """Publish a new version without changing the parent, train/val rows or ID mappings."""
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not math.isfinite(fraction)
        or not 0 < fraction <= 1
    ):
        raise ContractError("test fraction must be finite and in (0, 1]")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ContractError("sampling seed must be an integer in [0, 2**32)")
    source, output = Path(source_dir).resolve(), Path(output_dir).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ContractError("sampling source and output must be separate directories")
    parent = validate_ebnerd(source)
    if parent.version != VERSION:
        raise ContractError("test sampling requires the full canonical-v1 parent")
    identity = {
        "method": "uniform-whole-impression-without-replacement-v1",
        "requested_fraction": float(fraction),
        "seed": seed,
        "parent_fingerprint": parent.fingerprint,
    }
    if output.exists():
        manifest = validate_ebnerd(output)
        existing = manifest.metadata.get("test_sampling", {})
        if manifest.version != SAMPLED_VERSION or any(
            existing.get(k) != v for k, v in identity.items()
        ):
            raise ContractError("existing sampled output uses different inputs or parameters")
        return manifest

    test_root = source / "splits/test"
    impression_ids = np.load(test_root / "impression_ids.npy", mmap_mode="r")
    unique = np.unique(impression_ids)
    if not len(unique):
        raise ContractError("cannot sample an empty test set")
    selected_count = max(1, int(math.ceil(len(unique) * fraction)))
    selected = np.random.RandomState(seed).choice(unique, selected_count, replace=False)
    keep = np.isin(impression_ids, selected)
    count = int(keep.sum())
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(output.parent), prefix=".ebnerd-sample-") as temporary:
        stage = Path(temporary) / "dataset"
        stage.mkdir()
        for relative in parent.metadata["output_sha256"]:
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if Path(relative).parts[:2] == ("splits", "test"):
                values = np.load(source / relative, mmap_mode="r")
                if len(values) != len(keep):
                    raise ContractError("test source arrays have inconsistent row counts")
                np.save(destination, values[keep])
            else:
                shutil.copy2(source / relative, destination)
        sampled_root = stage / "splits/test"
        labels = np.load(sampled_root / "labels.npy", mmap_mode="r")
        positives = int(labels.sum())
        if positives == 0 or positives == count:
            raise ContractError("sampled test must retain both CTR classes")
        statistics = dict(parent.splits)
        statistics["test"] = SplitStatistics(
            samples=count,
            positives=positives,
            users=int(np.unique(np.load(sampled_root / "user_ids.npy")).size),
            items=int(np.unique(np.load(sampled_root / "item_ids.npy")).size),
            sha256=aggregate_file_digest({p.name: p for p in sorted(sampled_root.glob("*.npy"))}),
        )
        metadata = dict(parent.metadata)
        metadata["split_protocol"] += "+sampled-test-impressions-v1"
        metadata["test_sampling"] = {
            **identity,
            "original_impressions": int(len(unique)),
            "selected_impressions": selected_count,
            "original_samples": int(len(keep)),
            "selected_samples": count,
            "actual_candidate_fraction": count / len(keep),
            "count_rounding": "ceil",
        }
        metadata["impressions"] = {**metadata["impressions"], "test": selected_count}
        times = np.load(sampled_root / "timestamps.npy", mmap_mode="r")
        metadata["time_bounds_us"] = {
            **metadata["time_bounds_us"],
            "test": [int(times.min()), int(times.max())],
        }
        metadata["output_sha256"] = {
            p.relative_to(stage).as_posix(): sha256_file(p) for p in sorted(stage.rglob("*.npy"))
        }
        payload = parent.to_dict(include_fingerprint=False)
        payload.update(version=SAMPLED_VERSION, splits=statistics, metadata=metadata)
        manifest = DatasetManifest(**payload)
        save_manifest(manifest, stage / "manifest.json")
        os.replace(str(stage), str(output))
    return manifest

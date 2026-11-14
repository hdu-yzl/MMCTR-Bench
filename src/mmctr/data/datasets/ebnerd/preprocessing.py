"""Versioned, impression-preserving EB-NeRD conversion with causal fixed histories."""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pyarrow.parquet as pq

from mmctr.core import ContractError
from mmctr.data.datasets.arrays import (
    aggregate_file_digest,
    load_manifest,
    save_manifest,
    sha256_file,
    source_fingerprint,
)
from mmctr.data.manifest import DatasetManifest, SplitStatistics

VERSION = "ebnerd-small-canonical-v1"
PROTOCOL = "official-train-time-90-10-official-validation-as-local-test-v1"
EMBEDDINGS = {
    "text": (
        "artifacts/google_bert_base_multilingual_cased/bert_base_multilingual_cased.parquet",
        "google-bert/bert-base-multilingual-cased",
    ),
    "image": (
        "artifacts/Ekstra_Bladet_image_embeddings/image_embeddings.parquet",
        "image_embedding",
    ),
}


def _records(path: Path, columns):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=4096, columns=columns):
        yield from batch.to_pylist()


def _positive_id(value, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError("{} must be a positive integer".format(field))
    return value


def _timestamp(value) -> int:
    if value is None:
        raise ContractError("EB-NeRD timestamps must not be null")
    result = np.datetime64(value, "us")
    if np.isnat(result):
        raise ContractError("EB-NeRD timestamps must be finite")
    return int(result.astype(np.int64))


def _read_behaviors(path: Path, audit: Optional[Dict[str, int]] = None):
    columns = [
        "impression_id",
        "user_id",
        "impression_time",
        "article_ids_inview",
        "article_ids_clicked",
    ]
    rows = []
    seen = set()
    for row in _records(path, columns):
        impression = _positive_id(row["impression_id"], "impression_id")
        if impression in seen:
            raise ContractError("duplicate EB-NeRD impression")
        seen.add(impression)
        user = _positive_id(row["user_id"], "user_id")
        candidates = row["article_ids_inview"]
        clicks = row["article_ids_clicked"]
        if not candidates or clicks is None:
            raise ContractError("EB-NeRD requires nonempty candidates and observed click labels")
        candidates = [_positive_id(x, "candidate article_id") for x in candidates]
        clicks = [_positive_id(x, "clicked article_id") for x in clicks]
        if len(candidates) != len(set(candidates)):
            raise ContractError("duplicate EB-NeRD candidate article")
        if audit is not None:
            audit["repeated_click_entries_collapsed"] += len(clicks) - len(set(clicks))
        if not set(clicks).issubset(candidates):
            raise ContractError("clicked article is absent from impression candidates")
        rows.append((impression, user, _timestamp(row["impression_time"]), candidates, set(clicks)))
    if not rows:
        raise ContractError("EB-NeRD behaviors are empty")
    return sorted(rows, key=lambda row: (row[2], row[0]))


def _read_histories(path: Path, first_time: int, sequence_length: int):
    histories = {}
    referenced = set()
    for row in _records(path, ["user_id", "article_id_fixed", "impression_time_fixed"]):
        user = _positive_id(row["user_id"], "history user_id")
        if user in histories:
            raise ContractError("duplicate EB-NeRD history user")
        items, times = row["article_id_fixed"], row["impression_time_fixed"]
        if items is None or times is None or len(items) != len(times):
            raise ContractError("EB-NeRD history IDs and timestamps must align")
        events = []
        for order, (item, time) in enumerate(zip(items, times)):
            item = _positive_id(item, "history article_id")
            time = _timestamp(time)
            if time >= first_time:
                raise ContractError("EB-NeRD history contains future or same-time events")
            referenced.add(item)
            events.append((time, order, item))
        histories[user] = [entry[2] for entry in sorted(events)[-sequence_length:]]
    return histories, referenced


def _allocate(path: Path, shape, dtype):
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", shape=shape, dtype=dtype)


def _write_features(raw: Path, stage: Path, original_ids, item_offset: int):
    root = stage / "item_features"
    root.mkdir()
    count = len(original_ids)
    mapping = {item: index + 1 for index, item in enumerate(original_ids)}
    np.save(root / "original_item_ids.npy", np.asarray([0] + original_ids, dtype=np.int64))
    np.save(root / "item_ids.npy", np.asarray([0] + list(range(item_offset, item_offset + count))))
    dimensions, missing = {}, {}
    for name, (relative, column) in EMBEDDINGS.items():
        output = None
        present = np.zeros(count + 1, dtype=np.bool_)
        seen = set()
        for batch in pq.ParquetFile(raw / relative).iter_batches(
            batch_size=2048, columns=["article_id", column]
        ):
            for row in batch.to_pylist():
                item = _positive_id(row["article_id"], "embedding article_id")
                if item not in mapping:
                    continue
                if item in seen:
                    raise ContractError("duplicate {} embedding".format(name))
                seen.add(item)
                values = np.asarray(row[column], dtype=np.float32)
                if values.ndim != 1 or not values.size or not np.isfinite(values).all():
                    raise ContractError("invalid {} embedding".format(name))
                if output is None:
                    dimensions[name] = int(values.size)
                    output = _allocate(root / (name + ".npy"), (count + 1, values.size), np.float32)
                    output[:] = 0
                if values.size != dimensions[name]:
                    raise ContractError("inconsistent {} embedding dimension".format(name))
                output[mapping[item]] = values
                present[mapping[item]] = True
        if output is None:
            raise ContractError("no matching {} embeddings".format(name))
        output.flush()
        np.save(root / (name + "_present.npy"), present)
        missing[name] = int((~present[1:]).sum())
    return dimensions, missing


def _write_split(stage, name, rows, histories, user_map, item_map, sequence_length):
    count = sum(len(row[3]) for row in rows)
    if count == 0:
        raise ContractError("EB-NeRD split must not be empty")
    root = stage / "splits" / name
    arrays = {
        key: _allocate(root / (key + ".npy"), (count,), dtype)
        for key, dtype in (
            ("user_ids", np.int64),
            ("item_ids", np.int64),
            ("labels", np.float32),
            ("impression_ids", np.int64),
            ("timestamps", np.int64),
            ("candidate_positions", np.int32),
        )
    }
    arrays["history_item_ids"] = _allocate(
        root / "history_item_ids.npy", (count, sequence_length), np.int64
    )
    position, positives = 0, 0
    users, items = set(), set()
    for impression, user, time, candidates, clicks in rows:
        if user not in histories:
            raise ContractError("EB-NeRD behavior user has no fixed history record")
        history = [item_map[item] for item in histories[user]]
        padded = np.zeros(sequence_length, dtype=np.int64)
        if history:
            padded[-len(history) :] = history
        stop = position + len(candidates)
        arrays["user_ids"][position:stop] = user_map[user]
        arrays["item_ids"][position:stop] = [item_map[item] for item in candidates]
        arrays["history_item_ids"][position:stop] = padded
        arrays["labels"][position:stop] = [float(item in clicks) for item in candidates]
        arrays["impression_ids"][position:stop] = impression
        arrays["timestamps"][position:stop] = time
        arrays["candidate_positions"][position:stop] = np.arange(len(candidates))
        position = stop
        positives += len(clicks)
        users.add(user)
        items.update(candidates)
    if positives == 0 or positives == count:
        raise ContractError("EB-NeRD CTR splits must contain both classes")
    for array in arrays.values():
        array.flush()
    return SplitStatistics(
        samples=count,
        positives=positives,
        users=len(users),
        items=len(items),
        sha256=aggregate_file_digest({p.name: p for p in sorted(root.glob("*.npy"))}),
    )


def validate_ebnerd(output_dir: Path) -> DatasetManifest:
    """Verify all generated files against the manifest's closed set of SHA-256 hashes."""
    root = Path(output_dir)
    manifest = load_manifest(root / "manifest.json")
    if manifest.name != "ebnerd" or manifest.version not in (
        VERSION,
        "ebnerd-small-canonical-v2",
        "ebnerd-small-canonical-v3",
        "ebnerd-small-canonical-v4",
    ):
        raise ContractError("not a supported EB-NeRD manifest")
    hashes = manifest.metadata.get("output_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ContractError("EB-NeRD manifest lacks output hashes")
    if set(hashes) != {p.relative_to(root).as_posix() for p in root.rglob("*.npy")}:
        raise ContractError("EB-NeRD output file set mismatch")
    for relative, expected in hashes.items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ContractError("EB-NeRD output hash mismatch: {}".format(relative))
    return manifest


def dataset_config(manifest: DatasetManifest, data_dir: Path) -> Dict[str, Any]:
    """Derive model dimensions and ID bounds from the actual generated store."""
    dims = {"id": 0, **{k: v for k, v in manifest.feature_dimensions.items() if k != "id"}}
    return {
        "name": "ebnerd",
        "data_dir": str(data_dir),
        "version": manifest.version,
        "storage_format": manifest.storage_format,
        "fingerprint": manifest.fingerprint,
        "padding_id": 0,
        "id_offsets": dict(manifest.id_offsets),
        "id_feature_num": manifest.metadata["max_encoded_id"],
        "multimodel_num": 2,
        "mm_dims": dims,
        "mm_seq_dims": dims,
        **({"id_embedding_dim": 16} if "id_features" in manifest.metadata else {}),
        "id_fields_num": 2 + manifest.metadata.get("id_features", {}).get("extra_fields", 0),
        "extra_id_fields": manifest.metadata.get("id_features", {}).get("extra_fields", 0),
        "id_field_widths": [
            f["width"] for f in manifest.metadata.get("id_features", {}).get("fields", [])
        ],
        "seq_len": manifest.sequence_length,
        "min_item_id": manifest.id_offsets["item"],
        "min_user_id": 1,
        "user_features": ["id"],
        "user_features_dim": {"id": 0},
        "use_mm_features": ["id", "image", "text"],
        "use_mm_seq_features": ["id", "image", "text"],
        "use_seq": True,
        "use_mm_rq": False,
        "using_local_data": False,
        **(
            {
                "structured_feature_dim": manifest.metadata["structured_features"]["dimension"],
                "structured_feature_protocol": "residual-mlp-v1",
            }
            if "structured_features" in manifest.metadata and "id_features" not in manifest.metadata
            else {}
        ),
    }


def prepare_ebnerd(raw_dir: Path, output_dir: Path, sequence_length: int = 5) -> DatasetManifest:
    """Atomically build the user-approved 90/10/local-test protocol, or verify a prior build."""
    if (
        isinstance(sequence_length, bool)
        or not isinstance(sequence_length, int)
        or sequence_length < 1
    ):
        raise ContractError("EB-NeRD sequence_length must be a positive integer")
    raw, output = Path(raw_dir), Path(output_dir)
    names = ["ebnerd_small/articles.parquet"]
    names += [
        "ebnerd_small/{}/{}.parquet".format(s, f)
        for s in ("train", "validation")
        for f in ("behaviors", "history")
    ]
    names += [value[0] for value in EMBEDDINGS.values()]
    sources = {name: raw / name for name in names}
    if any(not p.is_file() for p in sources.values()):
        raise ContractError("EB-NeRD requires all seven official Parquet source files")
    fingerprint, source_hashes = source_fingerprint(sources)
    if output.exists():
        manifest = validate_ebnerd(output)
        if (
            manifest.source_fingerprint != fingerprint
            or manifest.sequence_length != sequence_length
            or manifest.metadata.get("split_protocol") != PROTOCOL
        ):
            raise ContractError("existing EB-NeRD output uses different inputs or parameters")
        return manifest

    source_audit = {"repeated_click_entries_collapsed": 0}
    train = _read_behaviors(raw / "ebnerd_small/train/behaviors.parquet", source_audit)
    test = _read_behaviors(raw / "ebnerd_small/validation/behaviors.parquet", source_audit)
    if len(train) < 2 or train[-1][2] >= test[0][2]:
        raise ContractError("official EB-NeRD training and validation must be time-disjoint")
    if {row[0] for row in train} & {row[0] for row in test}:
        raise ContractError("official EB-NeRD splits share impression IDs")
    cutoff = train[int(len(train) * 0.9)][2]
    splits = {
        "train": [r for r in train if r[2] < cutoff],
        "val": [r for r in train if r[2] >= cutoff],
        "test": test,
    }
    histories = {}
    referenced = {item for rows in splits.values() for r in rows for item in r[3]}
    for source, first in (("train", train[0][2]), ("validation", test[0][2])):
        histories[source], items = _read_histories(
            raw / ("ebnerd_small/" + source + "/history.parquet"), first, sequence_length
        )
        referenced.update(items)
    articles = pq.read_table(raw / "ebnerd_small/articles.parquet", columns=["article_id"])
    raw_articles = articles.column(0).to_pylist()
    if len(raw_articles) != len(set(raw_articles)) or not referenced.issubset(raw_articles):
        raise ContractError("EB-NeRD articles must uniquely cover candidates and histories")
    original_items = sorted(referenced)
    original_users = sorted({row[1] for rows in splits.values() for row in rows})
    user_map = {user: index + 1 for index, user in enumerate(original_users)}
    offset = len(user_map) + 1
    item_map = {item: offset + index for index, item in enumerate(original_items)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(output.parent), prefix=".ebnerd-") as temporary:
        stage = Path(temporary) / "dataset"
        stage.mkdir()
        dims, missing = _write_features(raw, stage, original_items, offset)
        (stage / "mappings").mkdir()
        np.save(stage / "mappings/original_user_ids.npy", np.asarray([0] + original_users))
        statistics = {
            name: _write_split(
                stage,
                name,
                rows,
                histories["validation" if name == "test" else "train"],
                user_map,
                item_map,
                sequence_length,
            )
            for name, rows in splits.items()
        }
        output_hashes = {
            p.relative_to(stage).as_posix(): sha256_file(p) for p in sorted(stage.rglob("*.npy"))
        }
        manifest = DatasetManifest(
            name="ebnerd",
            version=VERSION,
            storage_format="named-npy-v1",
            sequence_length=sequence_length,
            padding_id=0,
            id_offsets={"user": 1, "item": offset},
            feature_dimensions={"id": 1, **dims},
            splits=statistics,
            source_fingerprint=fingerprint,
            metadata={
                "split_protocol": PROTOCOL,
                "validation_cutoff_us": cutoff,
                "history_protocol": "official-fixed-before-source-split-left-pad-last-L-v1",
                "candidate_protocol": "all-inview-official-click-binary-no-negative-sampling",
                "missing_modality_policy": "zero-vector-with-explicit-presence-mask",
                "missing_modality_items": missing,
                "source_sha256": dict(source_hashes),
                "output_sha256": output_hashes,
                "user_count": len(user_map),
                "item_count": len(item_map),
                "max_encoded_id": offset + len(item_map) - 1,
                "impressions": {name: len(rows) for name, rows in splits.items()},
                "time_bounds_us": {
                    name: [rows[0][2], rows[-1][2]] for name, rows in splits.items()
                },
                "test_is_official_challenge_test": False,
                "source_audit": source_audit,
            },
        )
        save_manifest(manifest, stage / "manifest.json")
        os.replace(str(stage), str(output))
    return manifest


__all__ = ["prepare_ebnerd", "validate_ebnerd", "dataset_config"]

"""Label-independent EB-NeRD snapshot metadata and strictly prior fixed-history features."""

import hashlib
import math
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from mmctr.core import ContractError
from mmctr.data.datasets.arrays import save_manifest, sha256_file
from mmctr.data.manifest import DatasetManifest

from .preprocessing import validate_ebnerd

VERSION = "ebnerd-small-canonical-v3"
# Fixed, namespaced hashing does not fit vocabularies or statistics on held-out data.
ARTICLE_GROUPS = (
    ("category", 32),
    ("subcategory", 32),
    ("topics", 64),
    ("ner_clusters", 64),
    ("entity_groups", 16),
    ("article_type", 8),
    ("sentiment_label", 4),
)
ARTICLE_NUMERIC = (
    "premium",
    "sentiment_score",
    "sentiment_missing",
    "title_length",
    "subtitle_length",
    "body_length",
    "image_count",
    "published_missing",
)
ARTICLE_NAMES = [
    "{}_hash_{}".format(name, i) for name, width in ARTICLE_GROUPS for i in range(width)
] + list(ARTICLE_NUMERIC)
ARTICLE_DIM = len(ARTICLE_NAMES)
PROFILE_NAMES = ["history_mean_" + n for n in ARTICLE_NAMES] + [
    "history_count",
    "history_unique_count",
    "history_unique_fraction",
    "history_span_days",
    "history_read_mean",
    "history_read_max",
    "history_read_missing",
    "history_scroll_mean",
    "history_scroll_max",
    "history_scroll_missing",
    "history_recent_day_fraction",
    "history_recent_week_fraction",
]
CONTEXT_GROUPS = (("device_type", 8), ("gender", 4), ("postcode", 16))
CONTEXT_NAMES = [
    "{}_hash_{}".format(name, i) for name, width in CONTEXT_GROUPS for i in range(width)
] + [
    "is_sso_user",
    "age",
    "age_missing",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "candidate_count",
    "history_last_age_days",
    "history_first_age_days",
]
CROSS_NAMES = [
    "article_age_hours",
    "article_not_yet_published",
    "article_published_missing",
    "history_article_fraction",
    "category_overlap",
    "subcategory_overlap",
    "topics_overlap",
    "ner_clusters_overlap",
    "entity_groups_overlap",
    "article_type_overlap",
    "sentiment_label_overlap",
]
FEATURE_NAMES = ["item_" + n for n in ARTICLE_NAMES] + PROFILE_NAMES + CONTEXT_NAMES + CROSS_NAMES
EXCLUDED = [
    "article_ids_clicked",
    "read_time",
    "scroll_percentage",
    "next_read_time",
    "next_scroll_percentage",
    "total_inviews",
    "total_pageviews",
    "total_read_time",
    "last_modified_time",
    "is_subscriber",
    "session_id",
    "candidate_positions",
    "article_id (current page)",
    "url",
]


def rows(path):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=8192):
        yield from batch.to_pylist()


def timestamp(value):
    if value is None:
        return 0
    return int(np.datetime64(value, "us").astype(np.int64))


def scaled(value, maximum):
    return math.log1p(min(max(float(value or 0), 0), maximum)) / math.log1p(maximum)


def hashed(name, value, width):
    result = np.zeros(width, dtype=np.float32)
    values = value if isinstance(value, (tuple, list)) else [value]
    tokens = set(str(v) for v in values if v is not None)
    if not tokens:
        tokens = {"<missing>"}
    for token in sorted(tokens):
        digest = hashlib.sha256((name + ":" + token).encode("utf-8")).digest()
        result[int.from_bytes(digest[:8], "big") % width] += 1 / len(tokens)
    return result


def article_vector(row):
    result = [hashed(name, row.get(name), width) for name, width in ARTICLE_GROUPS]
    score = row.get("sentiment_score")
    result.append(
        np.array(
            [
                float(bool(row.get("premium"))),
                float(score or 0),
                float(score is None),
                scaled(len(row.get("title") or ""), 1000),
                scaled(len(row.get("subtitle") or ""), 5000),
                scaled(len(row.get("body") or ""), 100000),
                scaled(len(row.get("image_ids") or []), 100),
                float(row.get("published_time") is None),
            ],
            dtype=np.float32,
        )
    )
    return np.concatenate(result)


def history_profile(row, articles, cached=None):
    ids = row["article_id_fixed"]
    times = np.asarray([timestamp(t) for t in row["impression_time_fixed"]], dtype=np.int64)
    if not len(ids) or len(times) != len(ids) or (times <= 0).any():
        raise ContractError("invalid fixed history for enrichment")
    vectors = np.stack(
        [
            cached[int(i)]
            if cached is not None and int(i) in cached
            else article_vector(articles.get(int(i), {}))
            for i in ids
        ]
    )
    stats = []
    for key, maximum in (("read_time_fixed", 3600), ("scroll_percentage_fixed", 100)):
        values = row.get(key) or [None] * len(ids)
        if len(values) != len(ids):
            raise ContractError("unaligned fixed history side features")
        valid = np.array([float(v) for v in values if v is not None and math.isfinite(v)])
        stats.extend(
            [
                scaled(valid.mean(), maximum) if len(valid) else 0,
                scaled(valid.max(), maximum) if len(valid) else 0,
                1 - len(valid) / len(ids),
            ]
        )
    last, first = int(times.max()), int(times.min())
    extra = [
        scaled(len(ids), 10000),
        scaled(len(set(ids)), 10000),
        len(set(ids)) / len(ids),
        scaled((last - first) / 86400e6, 365),
        *stats,
        float(np.mean(times >= last - 86400e6)),
        float(np.mean(times >= last - 7 * 86400e6)),
    ]
    unique, counts = np.unique(ids, return_counts=True)
    return (
        np.concatenate([vectors.mean(axis=0), np.asarray(extra, dtype=np.float32)]),
        first,
        last,
        dict(zip(unique.tolist(), (counts / len(ids)).tolist())),
    )


def context_vector(row, first, last):
    time = timestamp(row["impression_time"])
    if last >= time:
        raise ContractError("fixed history must strictly precede exposure")
    dt = row["impression_time"]
    hour = dt.hour + dt.minute / 60 + dt.second / 3600
    age = row.get("age")
    extra = [
        float(bool(row.get("is_sso_user"))),
        min(max(float(age or 0), 0), 120) / 120,
        float(age is None),
        math.sin(2 * math.pi * hour / 24),
        math.cos(2 * math.pi * hour / 24),
        math.sin(2 * math.pi * dt.weekday() / 7),
        math.cos(2 * math.pi * dt.weekday() / 7),
        scaled(len(row["article_ids_inview"]), 1000),
        scaled((time - last) / 86400e6, 365),
        scaled((time - first) / 86400e6, 365),
    ]
    return np.concatenate(
        [
            *[hashed(n, row.get(n), w) for n, w in CONTEXT_GROUPS],
            np.asarray(extra, dtype=np.float32),
        ]
    )


def enrich_ebnerd(source_dir, raw_dir, output_dir):
    """Publish v3 atomically, keeping every v2 array byte-identical."""
    source, raw, output = map(lambda p: Path(p).resolve(), (source_dir, raw_dir, output_dir))
    if source == output or source in output.parents or output in source.parents:
        raise ContractError("enrichment source/output must be separate")
    parent = validate_ebnerd(source)
    if parent.version not in ("ebnerd-small-canonical-v1", "ebnerd-small-canonical-v2"):
        raise ContractError("enrichment requires an unenriched parent")
    # Match the exact raw source snapshot previously audited by the parent.
    for relative, expected in parent.metadata["source_sha256"].items():
        if sha256_file(raw / relative) != expected:
            raise ContractError("enrichment source hash mismatch: " + relative)
    if output.exists():
        manifest = validate_ebnerd(output)
        if manifest.metadata.get("structured_features", {}).get("parent_fingerprint") != (
            parent.fingerprint
        ):
            raise ContractError("existing enrichment has a different parent")
        return manifest
    articles = {int(r["article_id"]): r for r in rows(raw / "ebnerd_small/articles.parquet")}
    cached = {i: article_vector(r) for i, r in articles.items()}
    original = np.load(source / "item_features/original_item_ids.npy")
    item_vectors = np.stack([article_vector(articles.get(int(i), {})) for i in original])
    item_vectors[0] = 0
    published = np.array(
        [timestamp(articles.get(int(i), {}).get("published_time")) for i in original],
        dtype=np.int64,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(output.parent), prefix=".ebnerd-enrich-") as tmp:
        stage = Path(tmp) / "dataset"
        shutil.copytree(source, stage)
        store = stage / "structured"
        store.mkdir()
        np.save(store / "articles.npy", item_vectors)
        np.save(store / "published.npy", published)
        for official, splits in (("train", ("train", "val")), ("validation", ("test",))):
            profiles = {}
            for h in rows(raw / "ebnerd_small" / official / "history.parquet"):
                profiles[int(h["user_id"])] = history_profile(h, articles, cached)
            profile_users = sorted(profiles)
            user_index = {u: i for i, u in enumerate(profile_users)}
            profile_matrix = np.stack([profiles[u][0] for u in profile_users])
            np.save(store / (official + "_profiles.npy"), profile_matrix)
            for split in splits:
                sr = stage / "splits" / split
                impression_ids = np.load(sr / "impression_ids.npy", mmap_mode="r")
                unique, inverse = np.unique(impression_ids, return_inverse=True)
                lookup = {int(i): j for j, i in enumerate(unique)}
                context = np.zeros((len(unique), len(CONTEXT_NAMES)), dtype=np.float32)
                profile_index = np.zeros(len(unique), dtype=np.int64)
                frequency = np.zeros(len(impression_ids), dtype=np.float32)
                found = set()
                # Map each event using impression identity, never candidate order as a feature.
                frequencies = {}
                for r in rows(raw / "ebnerd_small" / official / "behaviors.parquet"):
                    iid = int(r["impression_id"])
                    if iid not in lookup:
                        continue
                    j = lookup[iid]
                    if iid in found:
                        raise ContractError("duplicate raw impression")
                    found.add(iid)
                    uid = int(r["user_id"])
                    _, first, last, freq = profiles[uid]
                    profile_index[j] = user_index[uid]
                    context[j] = context_vector(r, first, last)
                    frequencies[j] = freq
                if len(found) != len(unique):
                    raise ContractError("missing raw impression")
                encoded = np.load(sr / "item_ids.npy", mmap_mode="r").reshape(-1)
                target_original = original[encoded - parent.id_offsets["item"] + 1]
                for j, (idx, item) in enumerate(zip(inverse, target_original)):
                    frequency[j] = frequencies[int(idx)].get(int(item), 0)
                dest = store / split
                dest.mkdir()
                for name, values in (
                    ("context", context),
                    ("profile_index", profile_index),
                    ("impression_index", inverse),
                    ("history_frequency", frequency),
                ):
                    np.save(dest / (name + ".npy"), values)
                print("Enriched", split, len(impression_ids), flush=True)
        metadata = dict(parent.metadata)
        metadata["structured_features"] = {
            "protocol": "hashed-snapshot-fixed-history-residual-v1",
            "parent_fingerprint": parent.fingerprint,
            "dimension": len(FEATURE_NAMES),
            "names": FEATURE_NAMES,
            "article_dimension": ARTICLE_DIM,
            "profile_dimension": len(PROFILE_NAMES),
            "context_dimension": len(CONTEXT_NAMES),
            "excluded": EXCLUDED,
            "hashing": "sha256-first8-big-endian-modulo; normalized multi-hot",
            "history": "all official fixed clicks strictly before exposure; no label updates",
            "snapshot_caveat": "article metadata and pretrained embeddings are official snapshots",
            "model_protocol": "original backbone + structured Linear( D,128)-ReLU-Linear(128,1)",
        }
        metadata["output_sha256"] = {
            p.relative_to(stage).as_posix(): sha256_file(p) for p in sorted(stage.rglob("*.npy"))
        }
        payload = parent.to_dict(include_fingerprint=False)
        payload.update(version=VERSION, metadata=metadata)
        manifest = DatasetManifest(**payload)
        save_manifest(manifest, stage / "manifest.json")
        os.replace(str(stage), str(output))
    return manifest


def assemble_features(item, profile, context, published, times, frequency):
    """Vectorized candidate-profile interactions; all time differences use exposure time."""
    age = (times - published) / 3600e6
    extras = [
        np.log1p(np.clip(age, 0, 8760)) / np.log1p(8760),
        ((age < 0) & (published != 0)).astype(np.float32),
        (published == 0).astype(np.float32),
        frequency,
    ]
    extras[0] = np.where(published != 0, extras[0], 0)
    offset = 0
    for _, width in ARTICLE_GROUPS:
        extras.append(
            (item[:, offset : offset + width] * profile[:, offset : offset + width]).sum(1)
        )
        offset += width
    return np.concatenate([item, profile, context, np.stack(extras, axis=1)], axis=1).astype(
        np.float32
    )

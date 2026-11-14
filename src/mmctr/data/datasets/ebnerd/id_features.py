"""Train-vocabulary categorical IDs and fixed numeric buckets, without dense model inputs."""

import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, Set

import numpy as np

from mmctr.core import ContractError
from mmctr.data.datasets.arrays import save_manifest, sha256_file
from mmctr.data.manifest import DatasetManifest

from .enrichment import (
    ARTICLE_DIM,
    ARTICLE_GROUPS,
    ARTICLE_NUMERIC,
    CONTEXT_NAMES,
    CROSS_NAMES,
    PROFILE_NAMES,
    rows,
)
from .preprocessing import validate_ebnerd

VERSION = "ebnerd-small-canonical-v4"
BAG_WIDTH = 16
ARTICLE_CATS = [n for n, _ in ARTICLE_GROUPS]
CONTEXT_CATS = ["device_type", "gender", "postcode", "hour", "weekday"]
CAT_NAMES = ARTICLE_CATS + ["history_" + n for n in ARTICLE_CATS] + CONTEXT_CATS
CONTEXT_KEEP = [0, 1, 2, 7, 8, 9]
NUM_NAMES = (
    ["item_" + n for n in ARTICLE_NUMERIC]
    + PROFILE_NAMES[ARTICLE_DIM:]
    + [CONTEXT_NAMES[28 + i] for i in CONTEXT_KEEP]
    + CROSS_NAMES
)
# Values were transformed by fixed clipping/log scales in the audited v3 materialization.
# Binning those values is equivalent to fixed non-uniform boundaries in original units.
EDGES = [0.0, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]


def tokens(value):
    values = value if isinstance(value, (list, tuple)) else [value]
    return sorted(set(str(v) for v in values if v is not None))


def categorical_bag(value, vocabulary, missing, unknown, width=BAG_WIDTH):
    result = np.zeros(width, dtype=np.int64)
    values = tokens(value)
    if not values:
        result[0] = missing
    else:
        # No hash collisions; unknown and missing are distinct from padding.
        mapped = list(dict.fromkeys(vocabulary.get(v, unknown) for v in values))
        result[: min(len(mapped), width)] = mapped[:width]
    return result


def numeric_values(dense):
    # v3 layout: article 228, profile 240, context 38, interactions 11.
    profile_start = ARTICLE_DIM
    context_start = ARTICLE_DIM + len(PROFILE_NAMES)
    return np.concatenate(
        [
            dense[:, ARTICLE_DIM - len(ARTICLE_NUMERIC) : ARTICLE_DIM],
            dense[:, profile_start + ARTICLE_DIM : context_start],
            dense[:, [context_start + 28 + i for i in CONTEXT_KEEP]],
            dense[:, -len(CROSS_NAMES) :],
        ],
        axis=1,
    )


def numeric_ids(values, offsets):
    if not np.isfinite(values).all():
        raise ContractError("non-finite numeric feature before bucketization")
    return np.searchsorted(np.asarray(EDGES), values, side="right") + np.asarray(offsets)


def prepare_id_features(source_dir, raw_dir, output_dir):
    source, raw, output = [Path(p).resolve() for p in (source_dir, raw_dir, output_dir)]
    if source == output or source in output.parents or output in source.parents:
        raise ContractError("ID feature source/output must be separate")
    parent = validate_ebnerd(source)
    if parent.version != "ebnerd-small-canonical-v3":
        raise ContractError("ID conversion requires audited v3 feature materialization")
    for name, digest in parent.metadata["source_sha256"].items():
        if sha256_file(raw / name) != digest:
            raise ContractError("raw source hash mismatch: " + name)
    if output.exists():
        result = validate_ebnerd(output)
        if result.metadata.get("id_features", {}).get("parent_fingerprint") != parent.fingerprint:
            raise ContractError("ID output has a different parent")
        return result
    articles = {int(r["article_id"]): r for r in rows(raw / "ebnerd_small/articles.parquet")}
    original = np.load(source / "item_features/original_item_ids.npy")
    item_offset = parent.id_offsets["item"]
    selected_items = set(
        original[
            np.unique(np.load(source / "splits/train/item_ids.npy")) - item_offset + 1
        ].tolist()
    )
    users = np.load(source / "mappings/original_user_ids.npy")
    train_encoded_users = np.unique(np.load(source / "splits/train/user_ids.npy"))
    # The user mapping has a padding entry at position zero.
    selected_users = set(users[train_encoded_users].tolist())
    histories = {}
    profile_categories = {}
    for official in ("train", "validation"):
        history = {
            int(r["user_id"]): r for r in rows(raw / "ebnerd_small" / official / "history.parquet")
        }
        histories[official] = history
        cats = {}
        for uid, h in history.items():
            counters: Dict[str, Counter] = {name: Counter() for name in ARTICLE_CATS}
            for aid in h["article_id_fixed"]:
                a = articles.get(int(aid), {})
                for name in ARTICLE_CATS:
                    counters[name].update(tokens(a.get(name)))
            cats[uid] = {
                name: [v for v, _ in sorted(c.items(), key=lambda p: (-p[1], p[0]))[:BAG_WIDTH]]
                for name, c in counters.items()
            }
            if official == "train" and uid in selected_users:
                selected_items.update(h["article_id_fixed"])
        profile_categories[official] = cats
    vocab_values: Dict[str, Set[str]] = {name: set() for name in CAT_NAMES}
    for aid in selected_items:
        for name in ARTICLE_CATS:
            vocab_values[name].update(tokens(articles.get(int(aid), {}).get(name)))
    for uid in selected_users:
        for name in ARTICLE_CATS:
            vocab_values["history_" + name].update(profile_categories["train"][uid][name])
    train_impressions = set(np.unique(np.load(source / "splits/train/impression_ids.npy")).tolist())
    for r in rows(raw / "ebnerd_small/train/behaviors.parquet"):
        if int(r["impression_id"]) in train_impressions:
            r.update(hour=r["impression_time"].hour, weekday=r["impression_time"].weekday())
            for name in CONTEXT_CATS:
                vocab_values[name].update(tokens(r.get(name)))
    cursor = parent.metadata["max_encoded_id"] + 1
    fields = []
    vocabularies = {}
    for name in CAT_NAMES:
        values = sorted(vocab_values[name])
        mapping = {v: cursor + 2 + i for i, v in enumerate(values)}
        fields.append(
            {
                "name": name,
                "kind": "categorical",
                "start": cursor,
                "missing": cursor,
                "unknown": cursor + 1,
                "end": cursor + 1 + len(values),
                "width": BAG_WIDTH,
            }
        )
        vocabularies[name] = mapping
        cursor += 2 + len(values)
    offsets = []
    for name in NUM_NAMES:
        offsets.append(cursor)
        fields.append(
            {
                "name": name,
                "kind": "bucket",
                "start": cursor,
                "end": cursor + len(EDGES),
                "width": 1,
            }
        )
        cursor += len(EDGES) + 1

    def encode(name, value):
        f = fields[CAT_NAMES.index(name)]
        return categorical_bag(value, vocabularies[name], f["missing"], f["unknown"])

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(output.parent), prefix=".ebnerd-ids-") as temp:
        stage = Path(temp) / "dataset"
        shutil.copytree(source, stage)
        dest = stage / "categorical"
        dest.mkdir()
        article_ids = np.stack(
            [
                np.stack([encode(n, articles.get(int(i), {}).get(n)) for n in ARTICLE_CATS])
                for i in original
            ]
        )
        article_ids[0] = 0
        np.save(dest / "articles.npy", article_ids)
        for official, splits in (("train", ("train", "val")), ("validation", ("test",))):
            profile_ids = np.stack(
                [
                    np.stack(
                        [
                            encode("history_" + n, profile_categories[official][uid][n])
                            for n in ARTICLE_CATS
                        ]
                    )
                    for uid in sorted(histories[official])
                ]
            )
            np.save(dest / (official + "_profiles.npy"), profile_ids)
            for split in splits:
                unique = np.unique(np.load(source / "splits" / split / "impression_ids.npy"))
                lookup = {int(v): i for i, v in enumerate(unique)}
                context_ids = np.zeros((len(unique), len(CONTEXT_CATS), BAG_WIDTH), dtype=np.int64)
                for r in rows(raw / "ebnerd_small" / official / "behaviors.parquet"):
                    iid = int(r["impression_id"])
                    if iid in lookup:
                        r.update(
                            hour=r["impression_time"].hour, weekday=r["impression_time"].weekday()
                        )
                        context_ids[lookup[iid]] = np.stack(
                            [encode(n, r.get(n)) for n in CONTEXT_CATS]
                        )
                np.save(dest / (split + "_context.npy"), context_ids)
        metadata = dict(parent.metadata)
        metadata["base_max_encoded_id"] = metadata["max_encoded_id"]
        metadata["max_encoded_id"] = cursor - 1
        metadata["id_features"] = {
            "protocol": "field-buckets-shared-embedding-v1",
            "fields": fields,
            "vocabularies": vocabularies,
            "numeric_edges": EDGES,
            "numeric_offsets": offsets,
            "parent_fingerprint": parent.fingerprint,
            "extra_fields": len(fields),
            "fit_scope": "local training candidates, training users' prior fixed history and training impressions",
            "bag_policy": "unique lexical candidate tokens, top-frequency history tokens; cap 16; masked mean embedding",
            "embedding": "same global table and latent_dim as user_id and item_id",
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

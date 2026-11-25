"""EB-NeRD source semantics, leakage boundaries and downstream contracts."""

from datetime import datetime, timedelta

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from mmctr.core import ContractError
from mmctr.data.datasets.ebnerd.loader import EBNeRDLoader
from mmctr.data.datasets.ebnerd.preprocessing import (
    dataset_config,
    prepare_ebnerd,
    validate_ebnerd,
)
from mmctr.models.common.registry import create_model


def _parquet(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


@pytest.fixture
def raw(tmp_path):
    root = tmp_path / "raw"
    origin = datetime(2023, 5, 18)
    _parquet(root / "ebnerd_small/articles.parquet", [{"article_id": x} for x in (10, 20, 30)])
    for split, offset, count in (("train", 0, 20), ("validation", 30, 4)):
        _parquet(
            root / ("ebnerd_small/" + split + "/behaviors.parquet"),
            [
                {
                    "impression_id": i + 1 + offset * 10,
                    "user_id": 7,
                    "impression_time": origin + timedelta(hours=i + offset),
                    "article_ids_inview": [20, 10, 30],
                    "article_ids_clicked": [10, 30] if i % 2 else [20],
                }
                for i in range(count)
            ],
        )
        _parquet(
            root / ("ebnerd_small/" + split + "/history.parquet"),
            [
                {
                    "user_id": 7,
                    "article_id_fixed": [30, 10, 20],
                    "impression_time_fixed": [
                        origin - timedelta(days=1),
                        origin - timedelta(days=3),
                        origin - timedelta(days=2),
                    ],
                }
            ],
        )
    _parquet(
        root / "artifacts/google_bert_base_multilingual_cased/bert_base_multilingual_cased.parquet",
        [
            {"article_id": x, "google-bert/bert-base-multilingual-cased": [float(x), 1.0]}
            for x in (30, 10, 20)
        ],
    )
    _parquet(
        root / "artifacts/Ekstra_Bladet_image_embeddings/image_embeddings.parquet",
        [{"article_id": x, "image_embedding": [float(x), 2.0, 3.0]} for x in (20, 10)],
    )
    return root


def test_impressions_time_labels_histories_and_missing_modalities(raw, tmp_path):
    out = tmp_path / "canonical"
    manifest = prepare_ebnerd(raw, out, sequence_length=2)
    assert {k: v.samples for k, v in manifest.splits.items()} == {
        "train": 54,
        "val": 6,
        "test": 12,
    }
    assert manifest.metadata["missing_modality_items"] == {"text": 0, "image": 1}
    assert manifest.id_offsets["item"] == 2
    loader = EBNeRDLoader(dataset_config(manifest, out), batch_size=4)
    split_impressions = []
    for name in ("train", "val", "test"):
        split_impressions.append(set(np.load(out / "splits" / name / "impression_ids.npy")))
        batches = list(loader.iter_batches(name))
        assert sum(b.batch_size for b in batches) == manifest.splits[name].samples
        for b in batches:
            assert torch.equal(b.history_mask, b.history_features["id"].ne(0))
            # Sort upstream timestamps; the last two historical articles are 20, 30.
            assert b.history_features["id"].tolist() == [[3, 4]] * b.batch_size
            assert (
                b.metadata["modality_presence"]["image"]["history"].tolist()
                == [[True, False]] * b.batch_size
            )
        ids = np.load(out / "splits" / name / "item_ids.npy")
        assert ids[:3].tolist() == [3, 2, 4]  # Preserve original candidate order.
    assert all(
        not (a & b) for i, a in enumerate(split_impressions) for b in split_impressions[i + 1 :]
    )
    first = next(loader.iter_batches("train"))
    assert first.labels[:3].tolist() == [1.0, 0.0, 0.0]
    assert first.item_features["text"][:3, 0].tolist() == [20.0, 10.0, 30.0]
    assert torch.equal(first.item_features["image"][2], torch.zeros(3))
    assert validate_ebnerd(out).fingerprint == manifest.fingerprint
    assert prepare_ebnerd(raw, out, 2).fingerprint == manifest.fingerprint
    with pytest.raises(ContractError, match="different inputs"):
        prepare_ebnerd(raw, out, 3)
    changed = np.load(out / "item_features/image.npy")
    changed[1, 0] += 1
    np.save(out / "item_features/image.npy", changed)
    with pytest.raises(ContractError, match="hash mismatch"):
        prepare_ebnerd(raw, out, 2)


def test_time_ties_stay_together(raw, tmp_path):
    path = raw / "ebnerd_small/train/behaviors.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[-2]["impression_time"] = rows[-3]["impression_time"]
    _parquet(path, rows)
    manifest = prepare_ebnerd(raw, tmp_path / "canonical")
    bounds = manifest.metadata["time_bounds_us"]
    assert bounds["train"][1] < bounds["val"][0] < bounds["test"][0]
    assert manifest.metadata["impressions"]["val"] == 3


@pytest.mark.parametrize(
    "problem",
    [
        "future_history",
        "duplicate_candidate",
        "unknown_click",
        "missing_history",
        "overlapping_impression",
    ],
)
def test_invalid_sources_rejected_without_publishing(raw, tmp_path, problem):
    behavior = raw / "ebnerd_small/train/behaviors.parquet"
    if problem == "future_history":
        path = raw / "ebnerd_small/train/history.parquet"
        rows = pq.read_table(path).to_pylist()
        rows[0]["impression_time_fixed"][0] = datetime(2025, 1, 1)
    elif problem == "missing_history":
        path = raw / "ebnerd_small/train/history.parquet"
        rows = pq.read_table(path).to_pylist()
        rows[0]["user_id"] = 99
    elif problem == "overlapping_impression":
        path = raw / "ebnerd_small/validation/behaviors.parquet"
        rows = pq.read_table(path).to_pylist()
        rows[0]["impression_id"] = 1
    else:
        path = behavior
        rows = pq.read_table(path).to_pylist()
        if problem == "duplicate_candidate":
            rows[0]["article_ids_inview"] = [10, 10, 20]
        else:
            rows[0]["article_ids_clicked"] = [99]
    _parquet(path, rows)
    out = tmp_path / "canonical"
    with pytest.raises(ContractError):
        prepare_ebnerd(raw, out)
    assert not out.exists()


def test_models_consume_canonical_batch(raw, tmp_path):
    out = tmp_path / "canonical"
    manifest = prepare_ebnerd(raw, out)
    config = dataset_config(manifest, out)
    loader = EBNeRDLoader(config, batch_size=4)
    batch = next(loader.iter_batches("train"))
    assert batch.history_features["id"][0].tolist() == [0, 0, 2, 3, 4]
    for name in ("dnn", "din", "dnn_mm_seq", "lmf"):
        model = create_model(
            name,
            {
                "latent_dim": 4,
                "projection_dim": 4,
                "mlp_dims": [8],
                "dropout": 0.0,
                "batch_norm": False,
                "modal_fusion_method": "add",
            },
            config,
        )
        output = model(batch)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(output.logits, batch.labels)
        (loss + output.auxiliary_loss()).backward()
        assert torch.isfinite(output.logits).all()
    with pytest.raises(ContractError, match="config/manifest"):
        EBNeRDLoader({**config, "seq_len": 99}, 4)


def test_repeated_clicks_form_one_binary_label(raw, tmp_path):
    path = raw / "ebnerd_small/train/behaviors.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0]["article_ids_clicked"] = [20, 20]
    _parquet(path, rows)
    out = tmp_path / "canonical"
    manifest = prepare_ebnerd(raw, out)
    assert manifest.metadata["source_audit"]["repeated_click_entries_collapsed"] == 1
    assert np.load(out / "splits/train/labels.npy")[:3].tolist() == [1.0, 0.0, 0.0]


def test_test_sampling_preserves_impressions_and_training(raw, tmp_path):
    from mmctr.data.datasets.ebnerd.sampling import sample_ebnerd_test

    full = tmp_path / "full"
    parent = prepare_ebnerd(raw, full)
    sampled = tmp_path / "sampled"
    manifest = sample_ebnerd_test(full, sampled, fraction=0.5, seed=2025)
    assert manifest.version == "ebnerd-small-canonical-v2"
    assert manifest.splits["test"].samples == 6
    assert manifest.metadata["test_sampling"]["selected_impressions"] == 2
    assert manifest.metadata["test_sampling"]["parent_fingerprint"] == parent.fingerprint
    assert manifest.fingerprint != parent.fingerprint
    before = np.load(full / "splits/test/impression_ids.npy")
    after = np.load(sampled / "splits/test/impression_ids.npy")
    selection = np.isin(before, np.unique(after))
    for path in (full / "splits/test").glob("*.npy"):
        assert np.array_equal(
            np.load(sampled / "splits/test" / path.name), np.load(path)[selection]
        )
    for relative, digest in parent.metadata["output_sha256"].items():
        if not relative.startswith("splits/test/"):
            assert manifest.metadata["output_sha256"][relative] == digest
    again = sample_ebnerd_test(full, tmp_path / "repeat", 0.5, 2025)
    assert again.fingerprint == manifest.fingerprint
    assert sample_ebnerd_test(full, sampled, 0.5, 2025).fingerprint == manifest.fingerprint
    assert validate_ebnerd(full).fingerprint == parent.fingerprint
    assert (
        next(EBNeRDLoader(dataset_config(manifest, sampled), 4).iter_batches("test")).batch_size
        == 4
    )
    with pytest.raises(ContractError, match="different inputs"):
        sample_ebnerd_test(full, sampled, 0.5, 42)
    with pytest.raises(ContractError, match="separate"):
        sample_ebnerd_test(full, full)
    with pytest.raises(ContractError, match="canonical-v1"):
        sample_ebnerd_test(sampled, tmp_path / "recursive")
    for invalid in (0, -1, 1.1, float("nan"), True):
        with pytest.raises(ContractError, match="fraction"):
            sample_ebnerd_test(full, tmp_path / "invalid", invalid)
    for invalid in (-1, True, 2**32):
        with pytest.raises(ContractError, match="seed"):
            sample_ebnerd_test(full, tmp_path / "invalid", seed=invalid)


def test_enrichment_preserves_events_excludes_outcomes_and_uses_full_history(raw, tmp_path):
    from mmctr.data.datasets.ebnerd.enrichment import (
        enrich_ebnerd,
        history_profile,
        article_vector,
        FEATURE_NAMES,
    )

    parent_dir = tmp_path / "parent"
    parent = prepare_ebnerd(raw, parent_dir, sequence_length=2)
    root = tmp_path / "enriched"
    manifest = enrich_ebnerd(parent_dir, raw, root)
    assert manifest.metadata["structured_features"]["dimension"] == len(FEATURE_NAMES)
    for relative, digest in parent.metadata["output_sha256"].items():
        assert manifest.metadata["output_sha256"][relative] == digest
    assert validate_ebnerd(root).fingerprint == manifest.fingerprint
    loader = EBNeRDLoader(dataset_config(manifest, root), 5)
    for split in ("train", "val", "test"):
        for batch in loader.iter_batches(split):
            features = batch.context_features["structured"]
            assert features.shape == (batch.batch_size, len(FEATURE_NAMES))
            assert torch.isfinite(features).all()
    row = {"article_id": 10, "category": 2, "total_inviews": 1}
    original = article_vector(row)
    row.update(total_inviews=999999, total_pageviews=800, last_modified_time=datetime(2030, 1, 1))
    assert np.array_equal(original, article_vector(row))
    h = next(iter(pq.read_table(raw / "ebnerd_small/train/history.parquet").to_pylist()))
    profile, first, last, counts = history_profile(h, {})
    assert len(counts) == 3  # Full history, not the two retained sequence tokens.
    assert first < last
    with pytest.raises(ContractError, match="dimension"):
        config = dataset_config(manifest, root)
        config.pop("structured_feature_dim")
        EBNeRDLoader(config, 5)


def test_enrichment_context_ignores_labels_postclick_and_checks_causality():
    from mmctr.data.datasets.ebnerd.enrichment import context_vector, timestamp

    dt = datetime(2023, 5, 18, 10)
    row = dict(impression_time=dt, article_ids_inview=[1, 2], article_ids_clicked=[1])
    now = timestamp(dt)
    expected = context_vector(row, now - 2000000, now - 1000000)
    row.update(
        article_ids_clicked=[2],
        read_time=123,
        next_read_time=500,
        scroll_percentage=100,
        next_scroll_percentage=1,
        is_subscriber=True,
    )
    assert np.array_equal(expected, context_vector(row, now - 2000000, now - 1000000))
    with pytest.raises(ContractError, match="strictly precede"):
        context_vector(row, now - 1000000, now)


def test_bucketed_fields_share_embedding_and_change_original_prediction(raw, tmp_path):
    from dataclasses import replace
    from pathlib import Path
    import yaml
    from mmctr.data.datasets.ebnerd.enrichment import enrich_ebnerd
    from mmctr.data.datasets.ebnerd.id_features import prepare_id_features, numeric_ids, EDGES

    parent, enriched, root = [tmp_path / n for n in ("parent", "enriched", "ids")]
    prepare_ebnerd(raw, parent)
    enrich_ebnerd(parent, raw, enriched)
    manifest = prepare_id_features(enriched, raw, root)
    config = dataset_config(manifest, root)
    assert "structured_feature_dim" not in config
    assert config["id_fields_num"] == config["extra_id_fields"] + 2
    batch = next(EBNeRDLoader(config, 4).iter_batches("train"))
    ids = batch.user_features["attribute_ids"]
    assert ids.dtype == torch.long
    assert not batch.context_features
    for index, field in enumerate(manifest.metadata["id_features"]["fields"]):
        nonzero = ids[:, index][ids[:, index] != 0]
        assert nonzero.min() >= field["start"]
        assert nonzero.max() <= field["end"]
    catalog = yaml.safe_load(
        (Path(__file__).parents[2] / "configs/models/ebnerd/dnn.yaml").read_text()
    )
    model = create_model("dnn", catalog["model"], config)
    assert not hasattr(model, "structured_head")
    model.eval()
    values = model.embed_user_fields(batch)
    assert values.shape[1] == model.latent_dim * (config["extra_id_fields"] + 1)
    # All categorical/bucket IDs use exactly the existing identity embedding table.
    expected = model.embedding(ids[:, 0])
    mask = ids[:, 0].ne(0).unsqueeze(-1)
    expected = (expected * mask).sum(1) / mask.sum(1).clamp_min(1)
    assert torch.allclose(values[:, model.latent_dim : 2 * model.latent_dim], expected)
    original = model(batch).logits
    zero = replace(
        batch, user_features={**batch.user_features, "attribute_ids": torch.zeros_like(ids)}
    )
    assert not torch.equal(original, model(zero).logits)
    original.sum().backward()
    used = ids[ids > 0].unique()
    assert model.embedding.embedding.grad[used].abs().sum() > 0
    assert model.embedding.embedding.grad[0].abs().sum() == 0
    with pytest.raises(ContractError, match="ID fields"):
        model(replace(batch, user_features={"id": batch.user_features["id"]}))
    assert numeric_ids(
        np.array([[EDGES[3] - 1e-8], [EDGES[3]], [EDGES[3] + 1e-8]]), [50]
    ).ravel().tolist() == [53, 54, 54]


def test_category_unknown_missing_padding_and_vocab_train_only(raw, tmp_path):
    from mmctr.data.datasets.ebnerd.enrichment import enrich_ebnerd
    from mmctr.data.datasets.ebnerd.id_features import prepare_id_features, categorical_bag

    parent, enriched, root = [tmp_path / n for n in ("parent", "enriched", "ids")]
    # Device 99 occurs only in official validation and must remain OOV.
    for split, value in (("train", 1), ("validation", 99)):
        path = raw / "ebnerd_small" / split / "behaviors.parquet"
        records = pq.read_table(path).to_pylist()
        for row in records:
            row["device_type"] = value
        _parquet(path, records)
    prepare_ebnerd(raw, parent)
    enrich_ebnerd(parent, raw, enriched)
    manifest = prepare_id_features(enriched, raw, root)
    meta = manifest.metadata["id_features"]
    assert "99" not in meta["vocabularies"]["device_type"]
    field = next(f for f in meta["fields"] if f["name"] == "device_type")
    a = categorical_bag(99, meta["vocabularies"]["device_type"], field["missing"], field["unknown"])
    b = categorical_bag(
        None, meta["vocabularies"]["device_type"], field["missing"], field["unknown"]
    )
    assert a[0] == field["unknown"] != b[0] == field["missing"]
    assert np.all(a[1:] == 0)
    assert validate_ebnerd(root).fingerprint == manifest.fingerprint


def test_dataset_embedding_override_is_scoped_and_nonmutating():
    from mmctr.models.common.registry import resolve_model_config

    original = {"latent_dim": 128, "projection_dim": 128, "ebnerd": {"latent_dim": 128}}
    resolved = resolve_model_config(original, {"name": "ebnerd", "id_embedding_dim": 16})
    assert resolved["latent_dim"] == resolved["ebnerd"]["latent_dim"] == 16
    assert resolved["projection_dim"] == 128
    assert original["latent_dim"] == original["ebnerd"]["latent_dim"] == 128
    assert resolve_model_config(original, {"name": "tiktok"}) == original
    with pytest.raises(ValueError, match="positive"):
        resolve_model_config(original, {"id_embedding_dim": 0})


def test_history_variant_preserves_parent_and_all_non_history_arrays(raw, tmp_path):
    from mmctr.data.datasets.ebnerd.history_variant import resize_history

    parent_path = tmp_path / "parent"
    parent = prepare_ebnerd(raw, parent_path, sequence_length=2)
    target = tmp_path / "longer"
    manifest = resize_history(parent_path, raw, target, 5)
    assert manifest.sequence_length == 5
    assert validate_ebnerd(parent_path).fingerprint == parent.fingerprint
    assert validate_ebnerd(target).fingerprint == manifest.fingerprint
    assert resize_history(parent_path, raw, target, 5).fingerprint == manifest.fingerprint
    for name, digest in parent.metadata["output_sha256"].items():
        if name.endswith("/history_item_ids.npy"):
            old = np.load(parent_path / name)
            new = np.load(target / name)
            assert new.shape == (len(old), 5)
            assert np.array_equal(new[:, -2:], old)
            assert np.all(new[:, :2] == 0)
            assert np.all(np.count_nonzero(new, axis=1) == 3)
        else:
            assert manifest.metadata["output_sha256"][name] == digest
    batch = next(EBNeRDLoader(dataset_config(manifest, target), 3).iter_batches("train"))
    assert batch.history_mask.shape == (3, 5)
    with pytest.raises(ContractError, match="different inputs"):
        resize_history(parent_path, raw, target, 4)


def test_prepare_cli_matches_stages_and_resumes_after_failure(raw, tmp_path, monkeypatch):
    from mmctr.config import load_yaml_mapping
    from mmctr.data.datasets.ebnerd import id_features
    from mmctr.data.datasets.ebnerd.__main__ import main
    from mmctr.data.datasets.ebnerd.enrichment import enrich_ebnerd
    from mmctr.data.datasets.ebnerd.sampling import sample_ebnerd_test

    manual = tmp_path / "manual"
    prepare_ebnerd(raw, manual / "impressions", 2)
    sample_ebnerd_test(manual / "impressions", manual / "sampled", 0.5, 42)
    enrich_ebnerd(manual / "sampled", raw, manual / "attributes")
    expected = id_features.prepare_id_features(manual / "attributes", raw, manual / "ids")
    work, output, config = tmp_path / "work", tmp_path / "ids", tmp_path / "catalog.yaml"
    args = [
        "--prepare",
        "--raw-dir",
        str(raw),
        "--output-dir",
        str(output),
        "--work-dir",
        str(work),
        "--sequence-length",
        "2",
        "--test-fraction",
        "0.5",
        "--sampling-seed",
        "42",
        "--config-output",
        str(config),
    ]
    fit = id_features.prepare_id_features

    def fail(*args, **kwargs):
        raise RuntimeError("interrupted before ID materialization")

    monkeypatch.setattr(id_features, "prepare_id_features", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        main(args)
    assert not output.exists() and not config.exists()
    before = {p: p.stat().st_mtime_ns for p in work.rglob("*.npy")}
    monkeypatch.setattr(id_features, "prepare_id_features", fit)
    assert main(args) == 0
    assert all(p.stat().st_mtime_ns == value for p, value in before.items())
    actual = validate_ebnerd(output)
    assert actual.fingerprint == expected.fingerprint
    for relative in actual.metadata["output_sha256"]:
        np.testing.assert_array_equal(
            np.load(output / relative), np.load(manual / "ids" / relative)
        )
    assert load_yaml_mapping(config)["ebnerd"] == dataset_config(actual, output)
    assert main(args) == 0
    assert main(["--verify-only", "--output-dir", str(output)]) == 0
    with pytest.raises(SystemExit):
        main([*args, "--verify-only"])

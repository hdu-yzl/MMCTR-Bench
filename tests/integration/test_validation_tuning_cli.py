"""Real canonical preprocessing -> CLI search -> frozen final evaluation."""

import json

import numpy as np
import pytest
import yaml

from mmctr.cli.app import main
from mmctr.config import load_yaml_mapping
from mmctr.data.datasets.tiktok import prepare_tiktok
from mmctr.training.entrypoint import PROJECT_ROOT


def prepare_fixture(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for split, values in {
        "train": {"0": [0, 1], "1": [1, 2]},
        "val": {"0": [2], "1": [3], "2": [3]},
        "test": {"0": [3], "1": [4], "2": [4]},
    }.items():
        (raw / (split + ".json")).write_text(json.dumps(values))
    for name, width in (("text", 3), ("image", 2), ("audio", 1)):
        values = np.random.RandomState(width).normal(size=(9, width)).astype(np.float32)
        values[0] = 0
        np.save(raw / (name + "_feat.npy"), values)
    root = tmp_path / "canonical"
    manifest = prepare_tiktok(raw, root, sequence_length=2, negative_ratio=1)
    config = load_yaml_mapping(PROJECT_ROOT / "configs/datasets/catalog.yaml")["tiktok"]
    config.update(
        data_dir=str(root),
        fingerprint=manifest.fingerprint,
        seq_len=2,
        id_offsets=dict(manifest.id_offsets),
        id_feature_num=int(np.load(root / "item_features/item_ids.npy").max()),
        mm_dims={"text": 3, "image": 2, "audio": 1},
        mm_seq_dims={"text": 3, "image": 2, "audio": 1},
    )
    catalog = tmp_path / "datasets.yaml"
    catalog.write_text(yaml.safe_dump({"tiktok": config}))
    return root, catalog, config


def test_tune_cli_uses_validation_trials_and_tests_only_frozen_winner(tmp_path):
    _, catalog, _ = prepare_fixture(tmp_path)
    candidates = tmp_path / "candidates.yaml"
    candidates.write_text(
        yaml.safe_dump(
            {
                "candidates": [
                    {
                        "model": {
                            "latent_dim": 4,
                            "projection_dim": 4,
                            "mlp_dims": [8],
                            "batch_norm": False,
                            "attention_dim": 4,
                        },
                        "training": {
                            "lr": lr,
                            "max_epochs": 2,
                            "early_stop_patience": 1,
                            "batch_size": 4,
                        },
                    }
                    for lr in (0.01, 0.001)
                ]
            }
        )
    )
    output = tmp_path / "outputs"
    assert (
        main(
            [
                "tune",
                "--dataset-name",
                "tiktok",
                "--model-name",
                "dmf",
                "--dataset-config",
                str(catalog),
                "--candidates",
                str(candidates),
                "--study-id",
                "dmf-search",
                "--output-root",
                str(output),
                "--cuda",
                "-1",
                "--num-threads",
                "1",
            ]
        )
        == 0
    )
    history = json.loads((output / "dmf-search/trial_history.json").read_text())
    assert len(history["trials"]) == 2
    selected = json.loads((output / "dmf-search/frozen_selection.json").read_text())
    assert selected["validation_auc"] == max(t["metrics"]["val_auc"] for t in history["trials"])
    assert "test_auc" not in json.dumps(history)
    final = json.loads((output / "dmf-search/final_test_result.json").read_text())
    assert final["selected_run_id"] == selected["selected_run_id"]
    metrics = [
        json.loads(line)
        for path in output.rglob("metrics.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert sum(value["split"] == "test" for value in metrics) == 1
    assert not list(output.rglob("*.pt"))


def diagnostic_candidates(tmp_path, model_overrides=None):
    candidates = tmp_path / "diagnostic-candidates.yaml"
    candidates.write_text(
        yaml.safe_dump(
            {
                "candidates": [
                    {
                        "model": {
                            "latent_dim": 4,
                            "projection_dim": 4,
                            "mlp_dims": [8],
                            "batch_norm": False,
                            "attention_dim": 4,
                            **(model_overrides or {}),
                        },
                        "training": {
                            "lr": lr,
                            "max_epochs": 1,
                            "early_stop_patience": 1,
                            "batch_size": 4,
                        },
                    }
                    for lr in (0.01, 0.001)
                ]
            }
        )
    )
    return candidates


def run_diagnostic(tmp_path, catalog, family, args, model_overrides=None):
    from mmctr.analysis.diagnostic_execution import main as diagnostic_main

    output = tmp_path / "analysis"
    assert (
        diagnostic_main(
            [
                family,
                "--dataset-name",
                "tiktok",
                "--model-name",
                "dmf",
                "--dataset-config",
                str(catalog),
                "--candidates",
                str(diagnostic_candidates(tmp_path, model_overrides)),
                "--study-id",
                family,
                "--output-root",
                str(output),
                "--cuda",
                "-1",
                "--num-threads",
                "1",
            ]
            + args
        )
        == 0
    )
    return output / family


@pytest.mark.parametrize("evaluable", [True, False])
def test_paper_cold_execution_selects_on_full_validation_then_compares_same_weights(
    tmp_path, monkeypatch, evaluable
):
    from mmctr.analysis.benchmark_protocols import population_summary
    from mmctr.analysis import cold_populations

    root, catalog, _ = prepare_fixture(tmp_path)
    original = cold_populations.build_cold_populations
    _, expected = original(root, "union", "zero", splits=("test",))
    if not evaluable:
        expected["test"] = population_summary(np.array([], dtype=int))
    inspected = []

    def audit(*args, **kwargs):
        assert (tmp_path / "analysis/cold-start/frozen_selection.json").is_file()
        inspected.extend(kwargs["splits"])
        masks, populations = original(*args, **kwargs)
        if not evaluable:
            masks["test"][:] = False
            populations["test"] = expected["test"]
        return masks, populations

    monkeypatch.setattr(cold_populations, "build_cold_populations", audit)
    study = run_diagnostic(tmp_path, catalog, "cold-start", ["--regime", "zero"])
    frozen = json.loads((study / "frozen_selection.json").read_text())
    final = json.loads((study / "final_test_result.json").read_text())
    assert inspected == ["test"]
    assert final["selected_run_id"] == frozen["selected_run_id"]
    assert final["metadata"]["test_subset_population"] == expected["test"]
    full = {"test_full_auc", "test_full_loss"}
    if evaluable:
        assert final["metrics"].keys() == full | {"test_auc", "test_loss"}
        assert final["metadata"]["comparison"]["auc_delta"] == (
            final["metrics"]["test_auc"] - final["metrics"]["test_full_auc"]
        )
    else:
        assert final["metrics"].keys() == full
        assert final["metadata"]["subset_status"] == "not_evaluable"
        assert final["metadata"]["comparison"]["auc_delta"] is None
    test_runs = []
    for path in study.rglob("metrics.jsonl"):
        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert all(r["samples"] == 6 for r in records if r["split"] == "val")
        samples = [r["samples"] for r in records if r["split"] == "test"]
        if samples:
            test_runs.append((path.parent.name, samples))
    expected_samples = [6, expected["test"]["samples"]] if evaluable else [6]
    assert test_runs == [(final["run_id"], expected_samples)]
    assert not list(study.rglob("*.pt"))


def test_diagnostic_cli_tunes_one_setting_and_its_control_without_main_table(tmp_path):
    _, catalog, _ = prepare_fixture(tmp_path)
    for family, args in (
        ("robustness", ["--probability", "0.3"]),
        ("alignment", ["--method", "cosine", "--weight", "0.1"]),
        ("fusion", ["--method", "cat"]),
        ("control", []),
    ):
        study = run_diagnostic(
            tmp_path, catalog, family, args, {"similarity_source": "legacy_projected"}
        )
        trials = json.loads((study / "trial_history.json").read_text())["trials"]
        frozen = json.loads((study / "frozen_selection.json").read_text())
        final = json.loads((study / "final_test_result.json").read_text())
        assert len(trials) == 2
        assert all(set(t["metrics"]).isdisjoint({"test_auc", "test_loss"}) for t in trials)
        assert frozen["validation_auc"] == max(t["metrics"]["val_auc"] for t in trials)
        assert frozen["frozen_config"]["candidate"]["diagnostic"]["family"] == family
        assert final["selected_run_id"] == frozen["selected_run_id"]
        metrics = [
            json.loads(line)
            for path in study.rglob("metrics.jsonl")
            for line in path.read_text().splitlines()
        ]
        assert sum(value["split"] == "test" for value in metrics) == 1
        assert not list(study.rglob("*.pt"))

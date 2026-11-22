"""Saved defaults are isolated by pair and tied to validation/data provenance."""

import json

import pytest
import yaml

from mmctr.config import load_tuned_config, write_tuned_config


def selection_file(root, dataset="tiktok", model="dnn", lr=0.0001):
    root.mkdir(parents=True, exist_ok=True)
    path = root / "frozen_selection.json"
    path.write_text(
        json.dumps(
            {
                "selection_split": "val",
                "objective": "val_auc",
                "data_fingerprint": dataset + "-sha",
                "study_id": root.name,
                "selected_trial_id": "1",
                "selected_run_id": "winner",
                "seeds": [2025],
                "validation_auc": 0.8,
                "validation_log_loss": 0.4,
                "frozen_config": {
                    "data": {dataset: {"fingerprint": dataset + "-sha"}},
                    "model": {"model_name": model, "dropout": 0.3},
                    "training": {
                        "lr": lr,
                        "l2": 1e-6,
                        "batch_size": 512,
                        "optim": "adam",
                        "max_epochs": 20,
                        "early_stop_patience": 5,
                        "seed": 2025,
                        "cuda": 0,
                        "output_root": "/old/machine/output",
                    },
                },
            }
        )
    )
    return path


def test_independent_pairs_and_replacement_preserve_other_defaults(tmp_path):
    directory = tmp_path / "configs"
    source = selection_file(tmp_path / "first")
    destination = write_tuned_config(source, directory)
    write_tuned_config(selection_file(tmp_path / "other-model", model="din"), directory)
    write_tuned_config(selection_file(tmp_path / "other-data", dataset="antm2c"), directory)
    write_tuned_config(selection_file(tmp_path / "new", lr=1e-5), directory)

    selected = load_tuned_config(directory, "tiktok", "dnn", "tiktok-sha")
    assert selected["training"]["lr"] == 1e-5
    assert selected["selection"]["study_id"] == "new"
    assert load_tuned_config(directory, "tiktok", "din", "tiktok-sha")["training"]["lr"] == 1e-4
    assert load_tuned_config(directory, "antm2c", "dnn", "antm2c-sha")["training"]["lr"] == 1e-4
    assert load_tuned_config(directory, "microlens", "dnn", "absent") is None
    assert "seed" not in selected["training"]
    assert "output_root" not in selected["training"]
    assert "model_name" not in selected["model"]
    assert {path.name for path in destination.parent.iterdir()} == {"dnn.yaml", "din.yaml"}
    assert not list(directory.rglob("*.tmp"))


def test_stale_data_and_runtime_overrides_are_rejected(tmp_path):
    directory = tmp_path / "configs"
    path = write_tuned_config(selection_file(tmp_path / "study"), directory)
    with pytest.raises(ValueError, match="data_fingerprint mismatch"):
        load_tuned_config(directory, "tiktok", "dnn", "new-sha")
    payload = yaml.safe_load(path.read_text())
    payload["training"]["seed"] = 2025
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="unsupported tuned training"):
        load_tuned_config(directory, "tiktok", "dnn", "tiktok-sha")


def test_nonvalidation_selection_cannot_overwrite_defaults(tmp_path):
    directory = tmp_path / "configs"
    source = selection_file(tmp_path / "study")
    path = write_tuned_config(source, directory)
    before = path.read_bytes()
    payload = json.loads(source.read_text())
    payload["selection_split"] = "test"
    source.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="validation-AUC"):
        write_tuned_config(source, directory)
    assert path.read_bytes() == before


def test_selection_names_cannot_escape_config_directory(tmp_path):
    source = selection_file(tmp_path / "study", model="../escape")
    with pytest.raises(ValueError, match="invalid tuned config name"):
        write_tuned_config(source, tmp_path / "configs")
    assert not (tmp_path / "escape.yaml").exists()

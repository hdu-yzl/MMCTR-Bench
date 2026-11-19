"""Real canonical preprocessing -> CLI search -> frozen final evaluation."""

import json

import numpy as np
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

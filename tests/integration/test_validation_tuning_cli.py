"""Real canonical preprocessing -> CLI search -> frozen final evaluation."""

import json
import shutil

import numpy as np
import pytest
import yaml

from mmctr.cli.app import main
from mmctr.config import load_yaml_mapping
from mmctr.data.datasets.tiktok import prepare_tiktok
from mmctr.training.entrypoint import PROJECT_ROOT


@pytest.fixture(autouse=True)
def isolated_project_config(tmp_path, monkeypatch):
    """Exercise default write-back without changing the real repository configs."""
    from mmctr.training import entrypoint

    root = tmp_path / "project"
    shutil.copytree(PROJECT_ROOT / "configs", root / "configs")
    monkeypatch.setattr(entrypoint, "PROJECT_ROOT", root)
    return root


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


@pytest.mark.parametrize(
    "model_name", ["dcn", "lmf", "diff_msin", "marn", "dmf", "simcen", "make", "em3", "gmmf"]
)
def test_sampled_model_ranges_construct_and_train(model_name, tmp_path):
    pytest.importorskip("optuna")
    import torch

    from mmctr.experiments.tuning import load_tuning_search
    from mmctr.training.entrypoint import Trainer

    torch.set_num_threads(1)
    _, catalog, _ = prepare_fixture(tmp_path)
    fixed = tmp_path / "fixed.yaml"
    fixed.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "latent_dim": 4,
                    "projection_dim": 8,
                    "mlp_dims": [8],
                    "batch_norm": False,
                }
            }
        )
    )
    candidates, _ = load_tuning_search(fixed, model_name=model_name)
    # One production training pass exercises sampled architectures and objectives.
    candidate = candidates[0]
    trainer = Trainer(
        dataset_name="tiktok",
        model_name=model_name,
        dataset_config_path=str(catalog),
        cuda=-1,
        output_root=str(tmp_path / "runs"),
        save_checkpoints=False,
        model_overrides=candidate["model"],
        training_overrides={
            **candidate["training"],
            "max_epochs": 1,
            "early_stop_patience": 1,
            "batch_size": 4,
        },
    )
    if model_name == "gmmf":
        groups = {group["phase"]: group for group in trainer.engine.optimizer.param_groups}
        assert groups["main"]["lr"] == candidate["training"]["lr"]
        assert groups["discriminator"]["lr"] == candidate["model"]["lr_disc"]
        assert groups["generator"]["lr"] == candidate["model"]["lr_gen"]
        assert all(
            group["weight_decay"] == candidate["training"]["l2"] for group in groups.values()
        )
    result = trainer.run(evaluate_test=False)
    assert result.status == "completed"
    assert "test_auc" not in result.metrics


@pytest.mark.parametrize("model_name", ["qarm", "psrq"])
def test_quantized_tuning_overrides_reach_dataset_config_and_keep_artifacts_frozen(
    model_name, tmp_path, monkeypatch
):
    import torch

    from mmctr.training import entrypoint
    from mmctr.quantization import (
        PSRQPretrainer,
        ResidualQuantizer,
        psrq_artifact_path,
        rq_artifact_path,
    )
    from mmctr.config import load_training_config

    torch.set_num_threads(1)
    _, catalog, _ = prepare_fixture(tmp_path)
    artifacts = tmp_path / "quantizers"
    original_load = load_training_config

    def local_training(*args, **kwargs):
        config = original_load(*args, **kwargs).to_dict()
        config["quantization_artifact_dir"] = str(artifacts)
        return entrypoint.TrainingConfig.from_mapping(config, PROJECT_ROOT)

    monkeypatch.setattr(entrypoint, "load_training_config", local_training)
    data = load_yaml_mapping(catalog)["tiktok"]
    pretrain = {
        "projection_dim": 8,
        "psrq_dims": [6],
        "n_levels": 2,
        "codebook_size": 3,
        "dropout": 0.0,
        "batch_norm": True,
    }
    if model_name == "qarm":
        for name, dimension in data["mm_dims"].items():
            quantizer = ResidualQuantizer(n_levels=2, codebook_size=3, dimension=dimension)
            quantizer.set_codebooks(torch.randn(2, 3, dimension))
            quantizer.save(rq_artifact_path(artifacts, "tiktok", name))
    else:
        quantizer = PSRQPretrainer(pretrain, data)
        for module in quantizer.modules():
            if hasattr(module, "initted"):
                module.initted.fill_(True)
        quantizer.save(psrq_artifact_path(artifacts, "tiktok"))
    trainer = entrypoint.Trainer(
        dataset_name="tiktok",
        model_name=model_name,
        dataset_config_path=str(catalog),
        cuda=-1,
        output_root=str(tmp_path / "runs"),
        save_checkpoints=False,
        model_overrides={
            **pretrain,
            "latent_dim": 4,
            "mlp_dims": [8],
            "dropout": 0.3,
            "cross_num": 5,
        },
        training_overrides={"max_epochs": 1, "early_stop_patience": 1, "batch_size": 4},
    )
    assert trainer.model.dropout == 0.3
    assert trainer.model_config["tiktok"]["dropout"] == 0.3
    if model_name == "qarm":
        assert trainer.model.cross_layers == 5
    else:
        assert trainer.model.quantizer.dropout == 0.0
        assert trainer.model.quantizer.training is False
        assert all(
            not parameter.requires_grad for parameter in trainer.model.quantizer.parameters()
        )
    assert trainer.run(evaluate_test=False).status == "completed"

    from mmctr.config import write_tuned_config

    selection_path = tmp_path / "frozen_selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "selection_split": "val",
                "objective": "val_auc",
                "data_fingerprint": data["fingerprint"],
                "study_id": "quantized",
                "selected_trial_id": "0",
                "selected_run_id": trainer.run_context.run_id,
                "seeds": [2025],
                "validation_auc": 0.8,
                "validation_log_loss": 0.4,
                "frozen_config": {
                    "model": trainer.model_config,
                    "training": trainer.train_config,
                    "data": trainer.data_config,
                },
            }
        )
    )
    write_tuned_config(selection_path, entrypoint.PROJECT_ROOT / "configs/tuned")
    restored = entrypoint.Trainer(
        dataset_name="tiktok",
        model_name="mcca" if model_name == "psrq" else model_name,
        dataset_config_path=str(catalog),
        cuda=-1,
        seed=2029,
        output_root=str(tmp_path / "restored"),
        save_checkpoints=False,
    )
    assert restored.model_config == trainer.model_config
    assert restored.train_config["seed"] == 2029
    assert restored.model.dropout == 0.3
    if model_name == "psrq":
        assert restored.model.quantizer.dropout == 0.0
    assert restored.run(evaluate_test=False).status == "completed"


def test_default_cli_search_records_model_space_and_tests_one_frozen_winner(tmp_path, monkeypatch):
    from mmctr.training import entrypoint

    pytest.importorskip("optuna")
    _, catalog, _ = prepare_fixture(tmp_path)
    fixed = tmp_path / "fixed.yaml"
    fixed.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "latent_dim": 4,
                    "projection_dim": 4,
                    "mlp_dims": [8],
                    "batch_norm": False,
                }
            }
        )
    )
    original_load = entrypoint.load_training_config

    def short_training(*args, **kwargs):
        config = original_load(*args, **kwargs).to_dict()
        config.update(max_epochs=1, early_stop_patience=1, batch_size=4)
        return entrypoint.TrainingConfig.from_mapping(config, PROJECT_ROOT)

    monkeypatch.setattr(entrypoint, "load_training_config", short_training)
    output = tmp_path / "outputs"
    arguments = [
        "--dataset-name",
        "tiktok",
        "--model-name",
        "dcn",
        "--dataset-config",
        str(catalog),
        "--candidates",
        str(fixed),
        "--study-id",
        "search",
        "--output-root",
        str(output),
        "--cuda",
        "-1",
        "--num-threads",
        "1",
    ]
    assert main(["tune"] + arguments) == 0
    history = json.loads((output / "search/trial_history.json").read_text())
    frozen = json.loads((output / "search/frozen_selection.json").read_text())
    assert len(history["trials"]) == 50
    assert history["stop_reason"] == "max_trials"
    assert history["search_protocol"]["space"]["model.cross_num"] == [3, 4, 5]
    assert frozen["search_protocol"] == history["search_protocol"]
    assert {trial["config"]["model"]["cross_num"] for trial in history["trials"]} == {3, 4, 5}
    metrics = [
        json.loads(line)
        for path in output.rglob("metrics.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert sum(metric["split"] == "test" for metric in metrics) == 1


def test_tune_cli_uses_validation_trials_and_tests_only_frozen_winner(
    tmp_path, isolated_project_config
):
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

    saved_path = isolated_project_config / "configs/tuned/tiktok/dmf.yaml"
    saved = load_yaml_mapping(saved_path)
    assert saved["model"] == {
        k: v for k, v in selected["frozen_config"]["model"].items() if k != "model_name"
    }
    assert saved["training"]["lr"] == selected["frozen_config"]["training"]["lr"]
    assert saved["selection"]["run_id"] == selected["selected_run_id"]
    assert "test_auc" not in saved_path.read_text()
    assert "seed" not in saved["training"]
    assert "output_root" not in saved["training"]

    # The exact command shape used by the main table consumes the new defaults.
    from mmctr.experiments.main_table import command_for

    command = command_for(
        {"dataset": "tiktok", "model": "dmf", "seed": 2029},
        {"num_threads": 1, "use_local_data": False, "no_checkpoints": True},
        tmp_path / "main-table-job",
    )
    command[command.index("--cuda") + 1] = "-1"
    assert main(command[3:] + ["--dataset-config", str(catalog)]) == 0
    run = next((tmp_path / "main-table-job").rglob("resolved_config.yaml"))
    resolved = load_yaml_mapping(run)
    assert resolved["model"] == selected["frozen_config"]["model"]
    for key, value in saved["training"].items():
        assert resolved["train"][key] == value
    assert resolved["train"]["seed"] == 2029
    metadata = json.loads((run.parent / "run_metadata.json").read_text())
    assert metadata["tuned_selection"]["run_id"] == selected["selected_run_id"]


def test_shared_cli_runs_fifty_public_trials_then_tests_one_winner(tmp_path):
    pytest.importorskip("optuna")
    _, catalog, _ = prepare_fixture(tmp_path)
    fixed = tmp_path / "fixed.yaml"
    fixed.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "latent_dim": 4,
                    "projection_dim": 4,
                    "mlp_dims": [8],
                    "batch_norm": False,
                }
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
                "dnn",
                "--dataset-config",
                str(catalog),
                "--candidates",
                str(fixed),
                "--study-id",
                "shared",
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
    study = output / "shared"
    history = json.loads((study / "trial_history.json").read_text())
    selected = json.loads((study / "frozen_selection.json").read_text())
    final = json.loads((study / "final_test_result.json").read_text())
    assert len(history["trials"]) == 50
    assert all(t["status"] == "completed" for t in history["trials"])
    assert history["search_protocol"]["mode"] == "shared_random"
    assert history["stop_reason"] == "max_trials"
    assert history["budget"]["gpu_hours"] == 0
    assert history["budget"]["cpu_hours"] > 0
    assert selected["validation_auc"] == max(t["metrics"]["val_auc"] for t in history["trials"])
    assert final["selected_run_id"] == selected["selected_run_id"]
    metrics = [
        json.loads(line)
        for path in output.rglob("metrics.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert sum(value["split"] == "test" for value in metrics) == 1
    assert not list(output.rglob("*.pt"))


def test_write_back_opt_out_failed_search_and_retuning(tmp_path, isolated_project_config):
    _, catalog, config = prepare_fixture(tmp_path)
    candidates = tmp_path / "candidates.yaml"
    candidate = {
        "model": {
            "latent_dim": 4,
            "projection_dim": 4,
            "mlp_dims": [8],
            "dropout": 0.3,
            "batch_norm": False,
        },
        "training": {"lr": 0.001, "max_epochs": 1, "early_stop_patience": 1, "batch_size": 4},
    }
    candidates.write_text(yaml.safe_dump({"candidates": [candidate]}))
    base = [
        "tune",
        "--dataset-name",
        "tiktok",
        "--model-name",
        "dnn",
        "--dataset-config",
        str(catalog),
        "--candidates",
        str(candidates),
        "--output-root",
        str(tmp_path / "outputs"),
        "--cuda",
        "-1",
        "--num-threads",
        "1",
    ]
    saved_path = isolated_project_config / "configs/tuned/tiktok/dnn.yaml"
    assert main(base + ["--study-id", "skip", "--no-write-back"]) == 0
    assert not saved_path.exists()
    assert main(base + ["--study-id", "publish"]) == 0
    before = saved_path.read_bytes()
    assert main(base + ["--study-id", "skip-existing", "--no-write-back"]) == 0
    assert saved_path.read_bytes() == before

    candidate["training"]["lr"] = -1
    candidates.write_text(yaml.safe_dump({"candidates": [candidate]}))
    with pytest.raises(SystemExit):
        main(base + ["--study-id", "failed"])
    assert saved_path.read_bytes() == before

    # A new search must not inherit saved dropout or reject an old saved fingerprint.
    saved = load_yaml_mapping(saved_path)
    saved["data_fingerprint"] = "obsolete-data"
    saved_path.write_text(yaml.safe_dump(saved))
    candidate["training"]["lr"] = 0.0001
    del candidate["model"]["dropout"]
    candidates.write_text(yaml.safe_dump({"candidates": [candidate]}))
    assert main(base + ["--study-id", "retune"]) == 0
    updated = load_yaml_mapping(saved_path)
    assert updated["data_fingerprint"] == config["fingerprint"]
    assert updated["model"]["dropout"] == 0.0
    assert updated["training"]["lr"] == 0.0001

    from mmctr.training.entrypoint import Trainer

    # Saved defaults can be bypassed, and explicit caller overrides still take priority.
    normal = Trainer(
        dataset_name="tiktok",
        model_name="dnn",
        dataset_config_path=str(catalog),
        cuda=-1,
        output_root=str(tmp_path / "ordinary"),
        save_checkpoints=False,
        model_overrides={"dropout": 0.1},
        training_overrides={"lr": 0.002},
    )
    assert normal.model_config["dropout"] == 0.1
    assert normal.train_config["lr"] == 0.002
    assert normal.run(evaluate_test=False).status == "completed"
    saved_path.write_text(yaml.safe_dump(saved))
    with pytest.raises(ValueError, match="data_fingerprint mismatch"):
        Trainer(dataset_name="tiktok", model_name="dnn", dataset_config_path=str(catalog))
    normal = Trainer(
        dataset_name="tiktok",
        model_name="dnn",
        dataset_config_path=str(catalog),
        cuda=-1,
        output_root=str(tmp_path / "bypass"),
        save_checkpoints=False,
        use_tuned_config=False,
        model_overrides=candidate["model"],
        training_overrides=candidate["training"],
    )
    assert normal.model_config["dropout"] == 0.0
    assert normal.train_config["lr"] == 0.0001
    assert normal.run(evaluate_test=False).status == "completed"

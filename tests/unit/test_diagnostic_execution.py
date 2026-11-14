import copy
import json
from types import SimpleNamespace

import pytest

from mmctr.analysis.diagnostic_execution import (
    assert_frozen_settings,
    baseline_auc,
    main_table_complete,
    validate_task,
)
from mmctr.core import ContractError


def fixture():
    task = {
        "task_id": "tiktok-naml-alignment-cosine-0.1-seed2025",
        "dataset": "tiktok",
        "model": "naml",
        "seed": 2025,
        "family": "alignment",
        "baseline_test_auc": 0.8,
        "intervention": {"method": "cosine", "weight": 0.1},
    }
    baseline = {
        "jobs": [
            {
                "dataset": "tiktok",
                "model": "naml",
                "seed": 2025,
                "status": "completed",
                "metrics": {"test_auc": 0.8},
            }
        ],
        "data": {"tiktok": {"name": "tiktok"}},
        "model_catalog": {"naml": {"projection_dim": 128}},
        "training_defaults": {"seed": 2025, "lr": 0.001, "cuda": 0},
    }
    return task, baseline


def test_same_seed_baseline_rejects_missing_duplicate_or_wrong_value():
    task, baseline = fixture()
    assert baseline_auc(task, baseline) == 0.8
    changed = dict(task, seed=2026)
    with pytest.raises(ContractError):
        baseline_auc(changed, baseline)
    with pytest.raises(ContractError):
        baseline_auc(dict(task, baseline_test_auc=0.81), baseline)
    baseline["jobs"] *= 2
    with pytest.raises(ContractError):
        baseline_auc(task, baseline)


def test_fixed_grid_and_pending_single_pair_are_enforced():
    task, baseline = fixture()
    validate_task(task, baseline)
    for intervention in (
        {"method": "cosine", "weight": 0.2},
        {"method": "cosine", "weight": 0.1, "epochs": 1},
    ):
        with pytest.raises(ContractError):
            validate_task(dict(task, intervention=intervention), baseline)
    with pytest.raises(ContractError):
        validate_task(dict(task, task_id="../other"), baseline)


def test_hyperparameter_drift_rejected_but_device_and_output_may_change():
    task, baseline = fixture()
    trainer = SimpleNamespace(
        model_config={"projection_dim": 128, "model_name": "naml"},
        data_config=copy.deepcopy(baseline["data"]),
        train_config={"seed": 2025, "lr": 0.001, "cuda": 7, "output_root": "/tmp/example"},
    )
    assert_frozen_settings(trainer, task, baseline)
    trainer.train_config["lr"] = 0.002
    with pytest.raises(ContractError, match="training hyperparameters"):
        assert_frozen_settings(trainer, task, baseline)
    trainer.train_config["lr"] = 0.001
    trainer.model_config["projection_dim"] = 16
    with pytest.raises(ContractError, match="model hyperparameters"):
        assert_frozen_settings(trainer, task, baseline)


def test_main_table_completion_gate(tmp_path):
    path = tmp_path / "main.json"
    baseline = {"source": str(path)}
    jobs = [
        {"dataset": "fixture", "model": str(i), "seed": 2025, "status": "completed"}
        for i in range(420)
    ]
    path.write_text(json.dumps({"jobs": jobs[:-1]}))
    assert not main_table_complete(baseline)
    jobs[-1]["status"] = "running"
    path.write_text(json.dumps({"jobs": jobs}))
    assert not main_table_complete(baseline)
    jobs[-1]["status"] = "completed"
    path.write_text(json.dumps({"jobs": jobs}))
    assert main_table_complete(baseline)

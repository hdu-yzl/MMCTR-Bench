"""Exercise scheduling with real child processes, without real data or GPUs."""

import argparse
import json
import statistics
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from mmctr.experiments import main_table


WORKER = r"""
import json, os, sys, time
from pathlib import Path
root, shared, dataset, model, seed, fail = sys.argv[1:]
root, shared, seed = Path(root), Path(shared), int(seed)
gpu = os.environ['CUDA_VISIBLE_DEVICES']
# Exclusive marker detects two simultaneous processes assigned to the same GPU.
marker = shared / ('gpu-' + gpu)
fd = os.open(str(marker), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
os.close(fd)
try:
    assert os.environ['PYTHONHASHSEED'] == str(seed)
    time.sleep(0.15)
    (root / 'worker.json').write_text(json.dumps({'gpu':gpu, 'seed':seed}))
    if fail == 'yes':
        raise RuntimeError('synthetic worker failure')
    run = root / 'training' / dataset / model / 'run-1'
    run.mkdir(parents=True)
    (run / 'run_metadata.json').write_text(json.dumps({'status':'completed','seed':seed}))
    (run / 'resolved_config.yaml').write_text(json.dumps(
        {'train':{'seed':seed}, 'model':{'model_name':model}}))
    (run / 'summary.json').write_text(json.dumps({'test_auc':seed/10, 'test_loss':seed/5}))
finally:
    marker.unlink()
"""


def plan():
    return {
        "datasets": ["fixture"],
        "models": ["dnn", "din"],
        "seeds": [1, 2, 3, 4, 5],
        "sources": {},
        "num_threads": 1,
        "use_local_data": False,
        "missing_inputs": [],
    }


def test_eight_process_slots_failure_isolation_and_resume(tmp_path):
    spec = plan()
    jobs = main_table.new_jobs(spec)

    def command(job, _plan, output):
        fail = job["model"] == "dnn" and job["seed"] == 3 and len(job["attempts"]) == 1
        return [
            sys.executable,
            "-c",
            WORKER,
            str(output),
            str(tmp_path),
            job["dataset"],
            job["model"],
            str(job["seed"]),
            "yes" if fail else "no",
        ]

    with patch.object(main_table, "source_hashes", return_value={}), patch.object(
        main_table, "command_for", side_effect=command
    ):
        assert main_table.execute(spec, jobs, tmp_path, list(map(str, range(8)))) == 1
        report = json.loads((tmp_path / "main_table.json").read_text())
        partial, complete = report["table"]
        assert partial["n_success"] == 4 and partial["n_failed"] == 1
        assert partial["test_auc_mean"] is None
        assert complete["test_auc_mean"] == pytest.approx(0.3)
        assert complete["test_auc_std"] == pytest.approx(
            statistics.stdev([0.1, 0.2, 0.3, 0.4, 0.5])
        )
        assert {j["attempts"][0]["gpu"] for j in jobs} == set(map(str, range(8)))
        assert "synthetic worker failure" in Path(jobs[2]["attempts"][0]["log"]).read_text()
        assert main_table.execute(spec, jobs, tmp_path, list(map(str, range(8)))) == 0
    assert sum(len(job["attempts"]) for job in jobs) == 11
    assert all(row["status"] == "complete" for row in main_table.aggregate(spec, jobs))
    assert not list(tmp_path.glob("gpu-*"))
    assert "test_auc_seed_5" in (tmp_path / "main_table.csv").read_text()
    with patch.object(main_table, "build_plan", return_value=spec), patch.object(
        main_table.subprocess, "check_output", return_value="0, GPU-0\n"
    ), patch.object(main_table.subprocess, "Popen", side_effect=AssertionError("must reuse")):
        assert main_table.main(["--gpus", "0", "--output-dir", str(tmp_path), "--resume"]) == 0
        report = json.loads((tmp_path / "main_table.json").read_text())
        report["jobs"].pop()
        (tmp_path / "main_table.json").write_text(json.dumps(report))
        with pytest.raises(SystemExit) as error:
            main_table.main(["--gpus", "0", "--output-dir", str(tmp_path), "--resume"])
        assert error.value.code == 2


def test_nonfinite_and_wrong_seed_results_are_rejected(tmp_path):
    root = tmp_path / "training" / "fixture" / "dnn" / "run-1"
    root.mkdir(parents=True)
    (root / "resolved_config.yaml").write_text('{"train":{"seed":1},"model":{"model_name":"dnn"}}')
    (root / "run_metadata.json").write_text('{"status":"completed","seed":2}')
    (root / "summary.json").write_text('{"test_auc":0.8,"test_loss":0.5}')
    job = {"dataset": "fixture", "model": "dnn", "seed": 1}
    attempt = {"output": str(tmp_path)}
    with pytest.raises(ValueError, match="match"):
        main_table.collect_result(job, attempt)
    (root / "run_metadata.json").write_text('{"status":"completed","seed":1}')
    (root / "summary.json").write_text('{"test_auc":NaN,"test_loss":0.5}')
    with pytest.raises(ValueError, match="nonfinite"):
        main_table.collect_result(job, attempt)


def test_default_plan_and_cli_seed_command():
    args = argparse.Namespace(
        datasets="antm2c,microlens,tiktok",
        models="paper",
        seeds="2025,2026,2027,2028,2029",
        num_threads=8,
        use_local_data=False,
    )
    with patch.object(main_table, "_digest", return_value="hash"):
        spec = main_table.build_plan(args)
    assert len(spec["models"]) == 21
    assert "qarm" in spec["models"] and "psrq" in spec["models"]
    assert "dnn_mm" not in spec["models"] and "dnn_mm_seq" not in spec["models"]
    assert len(main_table.new_jobs(spec)) == 315
    command = main_table.command_for(main_table.new_jobs(spec)[0], spec, Path("/runs"))
    assert command[command.index("--seed") + 1] == "2025"
    assert command[command.index("--cuda") + 1] == "0"
    args.seeds = "1,01,2,3,4"
    with pytest.raises(ValueError, match="distinct"):
        main_table.build_plan(args)


def test_interruption_reaps_all_children(tmp_path):
    spec = plan()
    jobs = main_table.new_jobs(spec)
    processes = []
    real_popen = main_table.subprocess.Popen
    real_sleep = main_table.time.sleep
    interrupted = False

    def sleep_once(seconds):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        real_sleep(seconds)

    def launch(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        processes.append(process)
        return process

    with patch.object(main_table, "source_hashes", return_value={}), patch.object(
        main_table,
        "command_for",
        return_value=[sys.executable, "-c", "import time; time.sleep(30)"],
    ), patch.object(main_table.subprocess, "Popen", side_effect=launch), patch.object(
        main_table.time, "sleep", side_effect=sleep_once
    ):
        with pytest.raises(KeyboardInterrupt):
            main_table.execute(spec, jobs, tmp_path, ["0", "1"])
    assert len(processes) == 2
    assert all(process.poll() is not None for process in processes)
    assert jobs[0]["status"] == jobs[1]["status"] == "interrupted"


def test_trainer_seed_reaches_initialization_and_run_artifacts(tmp_path):
    from mmctr.core import Batch, ModelOutput
    from mmctr.training.entrypoint import Trainer
    from tests.unit.test_training_engine import TinyLoader

    class SeededModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)

        def forward(self, batch: Batch):
            return ModelOutput(self.linear(batch.item_features["score"]))

    initial_weights = []
    for seed in (2025, 2025, 2026):
        with patch("mmctr.training.entrypoint.get_data_loader", return_value=TinyLoader()), patch(
            "mmctr.training.entrypoint.ensure_canonical_loader", side_effect=lambda *a, **k: a[1]
        ), patch("mmctr.training.entrypoint.create_model", side_effect=lambda *a: SeededModel()):
            trainer = Trainer(
                dataset_name="tiktok",
                model_name="dnn",
                cuda=-1,
                seed=seed,
                output_root=str(tmp_path),
            )
        initial_weights.append(trainer.model.linear.weight.detach().clone())
        resolved = main_table.load_yaml_mapping(trainer.run_context.resolved_config_path)
        assert resolved["train"]["seed"] == seed
        assert json.loads(trainer.run_context.metadata_path.read_text())["seed"] == seed
        # Small synthetic smoke only; formal launcher never changes the training budget.
        trainer.train_config["max_epochs"] = 1
        trainer.train_config["early_stop_patience"] = 1
        trainer.run()
        assert json.loads(trainer.run_context.metadata_path.read_text())["status"] == "completed"
    assert torch.equal(initial_weights[0], initial_weights[1])
    assert not torch.equal(initial_weights[0], initial_weights[2])


def test_resume_refuses_changed_plan_and_second_launcher(tmp_path, monkeypatch):
    spec = plan()
    spec["missing_inputs"] = []
    monkeypatch.setattr(main_table, "build_plan", lambda args: spec)
    monkeypatch.setattr(main_table.subprocess, "check_output", lambda *a, **k: "0, GPU-0\n")
    report = {"plan": dict(spec, seeds=[5, 6, 7, 8, 9]), "jobs": []}
    (tmp_path / "main_table.json").write_text(json.dumps(report))
    with pytest.raises(SystemExit) as error:
        main_table.main(["--gpus", "0", "--output-dir", str(tmp_path), "--resume"])
    assert error.value.code == 2
    with (tmp_path / ".launcher.lock").open("w") as lock:
        main_table.fcntl.flock(lock, main_table.fcntl.LOCK_EX | main_table.fcntl.LOCK_NB)
        with pytest.raises(SystemExit) as error:
            main_table.main(["--gpus", "0", "--output-dir", str(tmp_path), "--resume"])
        assert error.value.code == 2


def test_public_train_cli_forwards_seed():
    from mmctr.cli.app import main

    with patch("mmctr.training.entrypoint.Trainer") as trainer, patch(
        "mmctr.cli.app._configure_threads"
    ):
        assert main(["train", "--seed", "2029"]) == 0
    assert trainer.call_args.kwargs["seed"] == 2029
    trainer.return_value.run.assert_called_once()


def test_two_workers_per_gpu_are_independent(tmp_path):
    spec = dict(plan(), jobs_per_gpu=2, no_checkpoints=True)
    jobs = main_table.new_jobs(spec)
    # Slot-specific markers reject duplicate assignments while permitting GPU sharing.
    worker = WORKER.replace("('gpu-' + gpu)", "('gpu-' + gpu + '-' + os.environ['MMCTR_GPU_SLOT'])")

    def command(job, _plan, output):
        return [
            sys.executable,
            "-c",
            worker,
            str(output),
            str(tmp_path),
            job["dataset"],
            job["model"],
            str(job["seed"]),
            "no",
        ]

    with patch.object(main_table, "source_hashes", return_value={}), patch.object(
        main_table, "command_for", side_effect=command
    ):
        assert main_table.execute(spec, jobs, tmp_path, ["0", "1"]) == 0
    assert {(j["attempts"][0]["gpu"], j["attempts"][0]["slot"]) for j in jobs} == {
        ("0", "0"),
        ("1", "0"),
        ("0", "1"),
        ("1", "1"),
    }
    assert all(j["status"] == "completed" for j in jobs)
    assert "--no-checkpoints" in main_table.command_for(jobs[0], spec, tmp_path)


def test_memory_checkpoint_matches_disk_best_and_writes_nothing(tmp_path):
    from tests.unit.test_training_engine import TinyLoader, TinyModel
    from mmctr.training import TrainingEngine, CheckpointManager, build_optimizer

    outputs = []
    for persist in (True, False):
        torch.manual_seed(2025)
        model = TinyModel()
        manager = CheckpointManager(tmp_path / str(persist), persist=persist)
        engine = TrainingEngine(
            model, build_optimizer(model, "Adam", 0.1), torch.device("cpu"), manager
        )
        result = engine.fit(TinyLoader(), 4, 2, "fixture")
        metric = engine.evaluate(TinyLoader(), "test", 0)
        outputs.append((result.metrics, metric.metrics, model.linear.weight.detach().clone()))
        if not persist:
            assert list(manager.directory.iterdir()) == []
            with pytest.raises(ValueError, match="cannot resume"):
                engine.resume()
            expected = model.linear.weight.detach().clone()
            with torch.no_grad():
                model.linear.weight.add_(10)
            manager.restore("best", model)
            assert torch.equal(expected, model.linear.weight)
    assert outputs[0][0] == outputs[1][0]
    assert outputs[0][1] == outputs[1][1]
    assert torch.equal(outputs[0][2], outputs[1][2])


def test_waiting_quantizer_does_not_block_other_jobs(tmp_path):
    spec = dict(
        plan(),
        wait_for_inputs=True,
        inputs={},
        training_defaults={"quantization_artifact_dir": str(tmp_path / "artifacts")},
        data={"fixture": {"data_dir": str(tmp_path / "data"), "use_mm_features": ["id", "text"]}},
    )
    path = tmp_path / "artifacts/rq/fixture/text.npz"
    spec["missing_inputs"] = [str(path)]
    assert main_table.job_ready({"dataset": "fixture", "model": "dnn"}, spec)
    assert not main_table.job_ready({"dataset": "fixture", "model": "qarm"}, spec)
    main_table.hydrate_inputs(spec)
    assert spec["missing_inputs"] == [str(path)]
    path.parent.mkdir(parents=True)
    path.write_bytes(b"atomic-test-artifact")
    main_table.hydrate_inputs(spec)
    assert spec["missing_inputs"] == []
    assert spec["inputs"][str(path)] == main_table._digest(path)
    assert main_table.job_ready({"dataset": "fixture", "model": "qarm"}, spec)

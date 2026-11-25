import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_single_model_dry_run_validates_and_prints_the_canonical_command() -> None:
    result = subprocess.run(
        [
            str(PROJECT_ROOT / "scripts/train_model.sh"),
            "--dataset",
            "tiktok",
            "--model",
            "dnn_mm_seq",
            "--gpu",
            "3",
            "--python",
            sys.executable,
            "--output-root",
            "outputs/one-click-test",
            "--num-threads",
            "4",
            "--use-local-data",
            "--dry-run",
        ],
        cwd="/tmp",
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "-m mmctr.cli train" in result.stdout
    assert "--dataset-name tiktok" in result.stdout
    assert "--model-name dnn_mm_seq" in result.stdout
    assert "--cuda 3" in result.stdout
    assert "--output-root outputs/one-click-test" in result.stdout
    assert "--num-threads 4" in result.stdout
    assert "--use-local-data" in result.stdout


def test_all_models_dry_run_includes_quantized_models_and_round_robins_gpus() -> None:
    result = subprocess.run(
        [
            str(PROJECT_ROOT / "scripts/train_all_models.sh"),
            "--dataset",
            "antm2c",
            "--gpus",
            "0,2",
            "--python",
            sys.executable,
            "--dry-run",
        ],
        cwd="/tmp",
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    commands = [line for line in result.stdout.splitlines() if line.startswith("Dry run:")]
    assert len(commands) == 23
    assert any("--model-name qarm" in line for line in commands)
    assert any("--model-name psrq" in line for line in commands)
    assert sum("--cuda 0" in line for line in commands) == 12
    assert sum("--cuda 2" in line for line in commands) == 11


def test_quantizer_pretraining_dry_run_prints_rq_then_psrq() -> None:
    result = subprocess.run(
        [
            str(PROJECT_ROOT / "scripts/pretrain_quantizers.sh"),
            "--dataset",
            "tiktok",
            "--gpu",
            "5",
            "--python",
            sys.executable,
            "--use-local-data",
            "--dry-run",
        ],
        cwd="/tmp",
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    commands = [line for line in result.stdout.splitlines() if line.startswith("Dry run:")]
    assert len(commands) == 2
    assert "-m mmctr.quantization.rq_entrypoint" in commands[0]
    assert "--dataset-name tiktok" in commands[0]
    assert "-m mmctr.quantization.psrq_entrypoint" in commands[1]
    assert "--dataset-name tiktok" in commands[1]
    assert "--cuda 5" in commands[1]
    assert all("--use-local-data" in line for line in commands)


def test_all_models_rejects_duplicate_gpu_workers() -> None:
    """Reject aliases that would launch concurrent workers on one physical GPU."""
    result = subprocess.run(
        [
            str(PROJECT_ROOT / "scripts/train_all_models.sh"),
            "--gpus",
            "0,0",
            "--models",
            "dnn",
            "--python",
            sys.executable,
            "--dry-run",
        ],
        cwd="/tmp",
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "duplicate CUDA index" in result.stderr


@pytest.mark.parametrize("pretraining_status", [0, 9])
def test_batch_prepares_before_training_and_aborts_on_pretraining_failure(
    tmp_path, pretraining_status
):
    import shutil

    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("train_all_models.sh", "common.sh"):
        shutil.copy2(PROJECT_ROOT / "scripts" / name, scripts / name)
    interpreter = tmp_path / "python"
    interpreter.write_text(
        "#!/usr/bin/env bash\n"
        'cd "$(dirname "$0")"\n'
        'if [[ "$2" == "mmctr.quantization.ensure" ]]; then\n'
        '  echo "prepare $*" >> events\n'
        "  exit " + str(pretraining_status) + "\n"
        'elif [[ "$3" == "list-datasets" ]]; then\n'
        "  echo tiktok\n"
        'elif [[ "$3" == "list-models" ]]; then\n'
        "  printf 'qarm\\npsrq\\n'\n"
        'elif [[ "$3" == "train" ]]; then\n'
        '  echo "train $*" >> events\n'
        "else\n  exit 99\nfi\n"
    )
    interpreter.chmod(0o755)
    result = subprocess.run(
        [
            str(scripts / "train_all_models.sh"),
            "--dataset",
            "tiktok",
            "--gpus",
            "2,3",
            "--models",
            "qarm,psrq",
            "--python",
            str(interpreter),
            "--use-local-data",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    events = (tmp_path / "events").read_text().splitlines()
    assert events[0].startswith("prepare ")
    assert "--models qarm,psrq --cuda 2" in events[0]
    assert "--use-local-data" in events[0]
    assert result.returncode == pretraining_status, result.stderr
    if pretraining_status:
        assert len(events) == 1
    else:
        assert len(events) == 3
        assert any("--model-name qarm --cuda 2" in line for line in events[1:])
        assert any("--model-name psrq --cuda 3" in line for line in events[1:])


def test_trainer_seed_reaches_initialization_and_run_artifacts(tmp_path):
    import json
    from unittest.mock import patch

    import torch

    from mmctr.config import load_yaml_mapping
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
        resolved = load_yaml_mapping(trainer.run_context.resolved_config_path)
        assert resolved["train"]["seed"] == seed
        assert json.loads(trainer.run_context.metadata_path.read_text())["seed"] == seed
        # Keep this seed regression small.
        trainer.train_config["max_epochs"] = 1
        trainer.train_config["early_stop_patience"] = 1
        trainer.run()
        assert json.loads(trainer.run_context.metadata_path.read_text())["status"] == "completed"
    assert torch.equal(initial_weights[0], initial_weights[1])
    assert not torch.equal(initial_weights[0], initial_weights[2])


def test_batch_reports_worker_failure_without_skipping_other_gpus(tmp_path):
    import shutil

    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("train_all_models.sh", "common.sh"):
        shutil.copy2(PROJECT_ROOT / "scripts" / name, scripts / name)
    interpreter = tmp_path / "python"
    interpreter.write_text(
        '#!/usr/bin/env bash\ncd "$(dirname "$0")"\n'
        '[[ "$2" != mmctr.quantization.ensure ]] || exit 0\n'
        'echo "$*" >> events\n'
        '[[ " $* " != *" --model-name dnn "* ]] || exit 9\n'
    )
    interpreter.chmod(0o755)
    result = subprocess.run(
        [
            str(scripts / "train_all_models.sh"),
            "--models",
            "dnn,din,deepfm",
            "--gpus",
            "0,2",
            "--python",
            str(interpreter),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    events = (tmp_path / "events").read_text()
    assert "--model-name dnn --cuda 0" in events
    assert "--model-name din --cuda 2" in events
    assert "--model-name deepfm" not in events

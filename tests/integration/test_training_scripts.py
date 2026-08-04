import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_single_model_script_exposes_a_stable_help_interface() -> None:
    result = subprocess.run(
        ["bash", str(PROJECT_ROOT / "scripts/train_model.sh"), "--help"],
        cwd="/tmp",
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--dataset" in result.stdout
    assert "--model" in result.stdout
    assert "--gpu" in result.stdout
    assert "--dry-run" in result.stdout


def test_single_model_dry_run_validates_and_prints_the_canonical_command() -> None:
    result = subprocess.run(
        [
            "bash",
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


def test_all_models_dry_run_excludes_quantized_models_and_round_robins_gpus() -> None:
    result = subprocess.run(
        [
            "bash",
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
    assert len(commands) == 21
    assert all("--model-name qarm" not in line for line in commands)
    assert all("--model-name psrq" not in line for line in commands)
    assert sum("--cuda 0" in line for line in commands) == 11
    assert sum("--cuda 2" in line for line in commands) == 10


def test_quantizer_pretraining_dry_run_prints_rq_then_psrq() -> None:
    result = subprocess.run(
        [
            "bash",
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
            "bash",
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

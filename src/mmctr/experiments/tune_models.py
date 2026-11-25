"""Select dataset/model pairs and run budgeted Optuna searches sequentially."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Optional, Sequence
from uuid import uuid4

from mmctr.config import load_model_config
from mmctr.experiments.search_protocol import TuningBudget


ROOT = Path(__file__).resolve().parents[3]
DATASETS = ("antm2c", "microlens", "tiktok", "ebnerd")


def _names(value):
    names = [part.strip().lower() for part in value.split(",")]
    if not all(names) or len(set(names)) != len(names):
        raise ValueError("names must be nonempty, unique and comma-separated")
    return names


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", required=True, help="comma-separated datasets, or all")
    parser.add_argument("--models", required=True, help="comma-separated models, paper, or all")
    parser.add_argument("--max-trials", type=int, default=50)
    parser.add_argument("--max-gpu-hours", type=float, default=50)
    parser.add_argument("--max-hours", type=float, help="CPU/GPU trial-hour limit per pair")
    parser.add_argument("--cuda", type=int, default=0, help="GPU index; -1 for CPU")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--dataset-config", type=Path)
    parser.add_argument("--model-config-dir", type=Path, default=ROOT / "configs/models")
    parser.add_argument("--candidates", type=Path, help="optional search overrides YAML")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/tuning")
    parser.add_argument("--study-prefix", help="optional unique batch name")
    parser.add_argument("--no-write-back", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print commands without tuning")
    return parser


def commands_for(args):
    from mmctr.models import available_models
    from mmctr.models.common.registry import model_spec

    TuningBudget(
        max_trials=args.max_trials, max_gpu_hours=args.max_gpu_hours, max_hours=args.max_hours
    )
    if args.num_threads < 1 or not 0 <= args.seed < 2**32 or args.cuda < -1:
        raise ValueError("invalid thread count, seed or CUDA index")
    datasets = list(DATASETS) if args.datasets == "all" else _names(args.datasets)
    if set(datasets) - set(DATASETS):
        raise ValueError("unknown dataset")
    if args.models in ("all", "paper"):
        models = list(available_models())
        if args.models == "paper":
            models = [m for m in models if m not in ("dnn_mm", "dnn_mm_seq")]
    else:
        models = [model_spec(name).name for name in _names(args.models)]
    if len(models) != len(set(models)):
        raise ValueError("model aliases select the same model more than once")
    prefix = args.study_prefix or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    )
    if Path(prefix).name != prefix or prefix in (".", ".."):
        raise ValueError("study-prefix must be a single directory name")
    commands = []
    for dataset in datasets:
        for model in models:
            # Validate every pair before starting the first expensive search.
            load_model_config(args.model_config_dir, dataset, model)
            command = [
                sys.executable,
                "-m",
                "mmctr.cli",
                "tune",
                "--dataset-name",
                dataset,
                "--model-name",
                model,
                "--study-id",
                "{}-{}-{}".format(prefix, dataset, model),
            ]
            for name in (
                "max_trials",
                "max_gpu_hours",
                "cuda",
                "seed",
                "num_threads",
                "output_root",
                "model_config_dir",
                "max_hours",
                "dataset_config",
                "candidates",
            ):
                value = getattr(args, name)
                if value is not None:
                    command.extend(["--" + name.replace("_", "-"), str(value)])
            if args.no_write_back:
                command.append("--no-write-back")
            commands.append(command)
    return commands


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        commands = commands_for(args)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(
        "{} searches; budgets apply separately to each dataset/model pair.".format(len(commands)),
        flush=True,
    )
    for command in commands:
        print(" ".join(shlex.quote(part) for part in command), flush=True)
        if not args.dry_run:
            result = subprocess.run(command, check=False)
            if result.returncode:
                return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

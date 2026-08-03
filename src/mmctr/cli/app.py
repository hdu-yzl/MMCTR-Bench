"""Dependency-light command-line interface for MMCTR-Bench."""

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Sequence

from mmctr import __version__


THREAD_ENVIRONMENT_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _configure_threads(num_threads: int) -> None:
    if num_threads < 1:
        raise ValueError("num_threads must be >= 1")
    value = str(num_threads)
    for key in THREAD_ENVIRONMENT_KEYS:
        os.environ[key] = value


def _run_train(arguments) -> int:
    _configure_threads(arguments.num_threads)
    from trainers.Trainers import Trainer

    trainer = Trainer(
        dataset_name=arguments.dataset_name,
        model_name=arguments.model_name,
        use_local_data=arguments.use_local_data,
        cuda=arguments.cuda,
        output_root=arguments.output_root,
    )
    trainer.run()
    return 0


def _validate_config(arguments) -> int:
    from mmctr.config import load_training_config

    config = load_training_config(arguments.config, project_root=arguments.project_root)
    print(json.dumps(config.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _list_models(_arguments) -> int:
    from mmctr.models import available_models

    print("\n".join(available_models()))
    return 0


def _list_datasets(_arguments) -> int:
    from mmctr.data import available_datasets

    print("\n".join(available_datasets()))
    return 0


def _list_quantizers(_arguments) -> int:
    from mmctr.quantization import available_quantizers

    print("\n".join(available_quantizers()))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mmctr", description="MMCTR-Bench command line")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="run an isolated training task")
    train.add_argument("--dataset-name", default="antm2c")
    train.add_argument("--model-name", default="dnn")
    train.add_argument("--use-local-data", action="store_true")
    train.add_argument("--cuda", type=int, default=None)
    train.add_argument("--output-root", default=None)
    train.add_argument("--num-threads", type=int, default=8)
    train.set_defaults(handler=_run_train)

    validate = subparsers.add_parser(
        "validate-config", help="validate and resolve a training YAML file"
    )
    validate.add_argument("--config", type=Path, required=True)
    validate.add_argument("--project-root", type=Path, default=None)
    validate.set_defaults(handler=_validate_config)

    models = subparsers.add_parser("list-models", help="list registered model names")
    models.set_defaults(handler=_list_models)

    quantizers = subparsers.add_parser(
        "list-quantizers", help="list registered quantization premodels"
    )
    quantizers.set_defaults(handler=_list_quantizers)

    datasets = subparsers.add_parser("list-datasets", help="list registered dataset names")
    datasets.set_defaults(handler=_list_datasets)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        return int(arguments.handler(arguments))
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    return 2

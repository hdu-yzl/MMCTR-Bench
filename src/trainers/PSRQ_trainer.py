"""Explicit entry point for fitting a canonical PSRQ artifact."""

import argparse
from pathlib import Path

import torch

from mmctr.config import (
    ConfigValidationError,
    load_local_paths,
    load_training_config,
    resolve_dataset_config,
)
from mmctr.quantization import PSRQPretrainer, fit_psrq, psrq_artifact_path
from mmctr.training import build_optimizer
from mmctr.utils import helper


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def train(dataset_name: str, cuda=None, use_local_data: bool = False) -> None:
    dataset = dataset_name.lower()
    model_catalog = helper.load_yaml(PROJECT_ROOT / "config/model.yaml")
    data_catalog = helper.load_yaml(PROJECT_ROOT / "config/seq_data.yaml")
    training = load_training_config(PROJECT_ROOT / "config/train.yaml")
    helper.setup_seed(training.seed)
    local_paths = None
    if use_local_data:
        local_paths = load_local_paths(PROJECT_ROOT / "configs/local/paths.yaml")
        if dataset not in local_paths.datasets:
            raise ConfigValidationError(
                ["local path for dataset {!r} is missing".format(dataset)]
            )
    data_config = resolve_dataset_config(
        dataset,
        data_catalog[dataset],
        project_root=PROJECT_ROOT,
        local_paths=local_paths,
    )
    config = model_catalog["psrq"]
    config = dict(config.get(dataset, config))
    loader = helper.getDataLoader(dataset, data_config, training.batch_size)
    raw_features = loader.get_multi_modal()
    modalities = tuple(name for name in data_config["use_mm_features"] if name != "id")
    features = {
        name: torch.from_numpy(raw_features[name]).to(dtype=torch.float32)
        for name in modalities
    }
    model = PSRQPretrainer(config, data_config)
    optimizer = build_optimizer(model, training.optim, training.lr, training.l2)
    requested_cuda = training.cuda if cuda is None else int(cuda)
    device = helper.getDevice(requested_cuda)
    loss = fit_psrq(
        model,
        features,
        optimizer,
        training.max_epochs,
        max(training.batch_size, model.codebook_size),
        device,
    )
    destination = psrq_artifact_path(training.quantization_artifact_dir, dataset)
    model.save(destination, metadata={"dataset": dataset})
    print("saved {} (final_loss={:.6f})".format(destination, loss))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Fit canonical MMCTR PSRQ artifact")
    parser.add_argument("--dataset-name", default="antm2c")
    parser.add_argument("--cuda", type=int)
    parser.add_argument("--use-local-data", action="store_true")
    arguments = parser.parse_args(argv)
    train(arguments.dataset_name, arguments.cuda, arguments.use_local_data)


if __name__ == "__main__":
    main()

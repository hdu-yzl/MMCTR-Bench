"""Explicit entry point for fitting a canonical PSRQ artifact."""

import argparse
from pathlib import Path


from mmctr.data import get_data_loader
from mmctr.quantization import PSRQPretrainer, fit_psrq, psrq_artifact_path
from mmctr.quantization.training import copy_feature_tables
from mmctr.training import build_optimizer
from mmctr.utils import helper

from .preparation import load_pretraining_config


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def train(dataset_name: str, cuda=None, use_local_data: bool = False) -> None:
    dataset = dataset_name.lower()
    training, data_config, config = load_pretraining_config(
        PROJECT_ROOT, dataset, "psrq", use_local_data
    )
    loader = get_data_loader(dataset, data_config, training.batch_size)
    raw_features = loader.get_multi_modal()
    modalities = tuple(name for name in data_config["use_mm_features"] if name != "id")
    features = copy_feature_tables(raw_features, modalities)
    # CTR dropout can be tuned independently of the frozen pretrainer.
    config = dict(config, dropout=config.get("quantizer_dropout", config.get("dropout", 0.0)))
    model = PSRQPretrainer(config, data_config)
    optimizer = build_optimizer(model, training.optim, training.lr, training.l2)
    requested_cuda = training.cuda if cuda is None else int(cuda)
    device = helper.get_device(requested_cuda)
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

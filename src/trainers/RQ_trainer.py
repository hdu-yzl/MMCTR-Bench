"""Explicit entry point for fitting canonical per-modality RQ artifacts."""

import argparse
from pathlib import Path

import numpy as np

from mmctr.config import (
    ConfigValidationError,
    load_local_paths,
    load_training_config,
    resolve_dataset_config,
)
from mmctr.quantization import ResidualQuantizer, rq_artifact_path
from mmctr.utils import helper


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def train(dataset_name: str, use_local_data: bool = False) -> None:
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
    config = model_catalog["rq"]
    config = dict(config.get(dataset, config))
    loader = helper.getDataLoader(dataset, data_config, training.batch_size)
    modalities = tuple(name for name in data_config["use_mm_features"] if name != "id")
    feature_table = loader.get_multi_modal()
    for modality in modalities:
        values = np.asarray(feature_table[modality], dtype=np.float32)
        denominator = np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)
        modality_config = dict(config)
        modality_config["dimension"] = int(values.shape[1])
        quantizer = ResidualQuantizer(modality_config).fit(values / denominator)
        destination = rq_artifact_path(
            training.quantization_artifact_dir, dataset, modality
        )
        quantizer.save(
            destination,
            metadata={
                "dataset": dataset,
                "modality": modality,
                "normalization": "l2",
            },
        )
        print("saved {}".format(destination))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Fit canonical MMCTR RQ artifacts")
    parser.add_argument("--dataset-name", default="antm2c")
    parser.add_argument("--use-local-data", action="store_true")
    arguments = parser.parse_args(argv)
    train(arguments.dataset_name, arguments.use_local_data)


if __name__ == "__main__":
    main()

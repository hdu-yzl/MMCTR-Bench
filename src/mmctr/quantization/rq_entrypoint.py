"""Explicit entry point for fitting canonical per-modality RQ artifacts."""

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from mmctr.data import get_data_loader
from mmctr.config import ConfigValidationError
from mmctr.quantization import ResidualQuantizer, rq_artifact_path

from .preparation import load_pretraining_config


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def train(
    dataset_name: str,
    use_local_data: bool = False,
    modalities: Optional[Sequence[str]] = None,
) -> None:
    dataset = dataset_name.lower()
    training, data_config, config = load_pretraining_config(
        PROJECT_ROOT, dataset, "rq", use_local_data
    )
    available = tuple(name for name in data_config["use_mm_features"] if name != "id")
    modalities = available if modalities is None else tuple(modalities)
    if not modalities or set(modalities) - set(available):
        raise ConfigValidationError(["RQ modalities must be nonempty dataset modalities"])
    loader = get_data_loader(dataset, data_config, training.batch_size)
    feature_table = loader.get_multi_modal()
    for modality in modalities:
        values = np.asarray(feature_table[modality], dtype=np.float32)
        denominator = np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)
        modality_config = dict(config)
        modality_config["dimension"] = int(values.shape[1])
        quantizer = ResidualQuantizer(modality_config).fit(values / denominator)
        destination = rq_artifact_path(training.quantization_artifact_dir, dataset, modality)
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
    parser.add_argument("--modalities", nargs="+", help="Fit only these modality artifacts")
    arguments = parser.parse_args(argv)
    train(arguments.dataset_name, arguments.use_local_data, arguments.modalities)


if __name__ == "__main__":
    main()

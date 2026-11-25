"""Shared configuration preparation for RQ and PSRQ fitting."""

from pathlib import Path
from typing import Any, Dict, Tuple

from mmctr.config import (
    ConfigValidationError,
    TrainingConfig,
    load_local_paths,
    load_model_config,
    load_yaml_mapping,
    resolve_dataset_config,
    resolve_training_config,
)
from mmctr.utils import helper


def load_pretraining_config(
    project_root: Path, dataset: str, quantizer: str, use_local_data: bool
) -> Tuple[TrainingConfig, Dict[str, Any], Dict[str, Any]]:
    """Resolve model-owned fitting parameters and dataset paths, then seed fitting."""
    parameters = load_model_config(project_root / "configs/models", dataset, quantizer)
    training = resolve_training_config(
        parameters["pretraining" if quantizer == "psrq" else "training"], project_root
    )
    catalog = load_yaml_mapping(project_root / "configs/datasets/catalog.yaml")
    local_paths = None
    if use_local_data:
        local_paths = load_local_paths(project_root / "configs/local/paths.yaml")
        if dataset not in local_paths.datasets:
            raise ConfigValidationError(["local path for dataset {!r} is missing".format(dataset)])
    data = resolve_dataset_config(
        dataset, catalog[dataset], project_root=project_root, local_paths=local_paths
    )
    helper.setup_seed(training.seed)
    return training, data, parameters["model"]

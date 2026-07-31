"""Validated configuration loading and layering."""

from .loading import (
    find_project_root,
    load_training_config,
    load_yaml_mapping,
    merge_config_layers,
)
from .schema import ConfigValidationError, TrainingConfig
from .paths import (
    LocalPaths,
    load_dataset_catalog,
    load_local_paths,
    resolve_dataset_config,
)


__all__ = [
    "ConfigValidationError",
    "TrainingConfig",
    "find_project_root",
    "load_training_config",
    "load_yaml_mapping",
    "merge_config_layers",
    "LocalPaths",
    "load_dataset_catalog",
    "load_local_paths",
    "resolve_dataset_config",
]

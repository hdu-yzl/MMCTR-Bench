"""Validated configuration loading and layering."""

from .loading import (
    find_project_root,
    load_training_config,
    load_yaml_mapping,
    merge_config_layers,
)
from .schema import ConfigValidationError, TrainingConfig
from .runtime import resolve_training_config, runtime_defaults
from .models import MODEL_TRAINING_KEYS, load_model_config, write_model_config
from .paths import (
    LocalPaths,
    load_dataset_catalog,
    load_local_paths,
    resolve_dataset_config,
)


__all__ = [
    "ConfigValidationError",
    "TrainingConfig",
    "resolve_training_config",
    "runtime_defaults",
    "find_project_root",
    "load_training_config",
    "load_yaml_mapping",
    "merge_config_layers",
    "LocalPaths",
    "load_dataset_catalog",
    "load_local_paths",
    "resolve_dataset_config",
    "MODEL_TRAINING_KEYS",
    "load_model_config",
    "write_model_config",
]

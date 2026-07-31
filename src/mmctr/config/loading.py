"""Strict YAML loading and deterministic configuration layering."""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import yaml

from .schema import ConfigValidationError, TrainingConfig


PathLike = Union[str, Path]


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConfigValidationError(
                ["duplicate YAML key {!r} at line {}".format(key, key_node.start_mark.line + 1)]
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_yaml_mapping(path: PathLike) -> Dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError("config file not found: {}".format(config_path))
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            value = yaml.load(handle, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise ConfigValidationError(
            ["invalid YAML syntax in {}: {}".format(config_path, error)]
        ) from error
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ConfigValidationError(
            ["top-level YAML value in {} must be a mapping".format(config_path)]
        )
    return dict(value)


def find_project_root(path: PathLike) -> Path:
    start = Path(path).expanduser().resolve()
    if start.is_file():
        start = start.parent
    for candidate in (start,) + tuple(start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise ConfigValidationError(
        ["could not find a project root containing pyproject.toml from {}".format(start)]
    )


def load_training_config(
    path: PathLike,
    project_root: Optional[PathLike] = None,
) -> TrainingConfig:
    config_path = Path(path).expanduser().resolve()
    root = (
        Path(project_root).expanduser().resolve()
        if project_root
        else find_project_root(config_path)
    )
    return TrainingConfig.from_mapping(load_yaml_mapping(config_path), root)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def merge_config_layers(*layers: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Merge low-to-high precedence layers without mutating any input mapping."""

    resolved: Dict[str, Any] = {}
    for index, layer in enumerate(layers):
        if layer is None:
            continue
        if not isinstance(layer, Mapping):
            raise ConfigValidationError(["config layer {} must be a mapping".format(index)])
        resolved = _deep_merge(resolved, layer)
    return resolved

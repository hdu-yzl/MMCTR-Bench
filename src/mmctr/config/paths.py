"""Machine-local dataset and output path injection."""

import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Union

from .loading import load_yaml_mapping
from .schema import ConfigValidationError


PathLike = Union[str, Path]
DATASETS = ("antm2c", "microlens", "tiktok")
DATASET_ENVIRONMENT_KEYS = {
    "antm2c": "MMCTR_ANTM2C_DATA_DIR",
    "microlens": "MMCTR_MICROLENS_DATA_DIR",
    "tiktok": "MMCTR_TIKTOK_DATA_DIR",
}
OUTPUT_ENVIRONMENT_KEY = "MMCTR_OUTPUT_ROOT"


def _absolute_path(value: Any, field: str, issues: list) -> Optional[Path]:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        issues.append("{} must be a non-empty absolute path".format(field))
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        issues.append("{} must be absolute".format(field))
        return None
    return path.resolve()


@dataclass(frozen=True)
class LocalPaths:
    """Validated paths that are intentionally excluded from version control."""

    datasets: Mapping[str, Path]
    output_root: Optional[Path] = None

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        require_existing_data: bool = True,
    ) -> "LocalPaths":
        if not isinstance(values, Mapping):
            raise ConfigValidationError(["local paths config must be a mapping"])
        issues = []
        unknown_top_level = sorted(set(values) - {"datasets", "output_root"})
        if unknown_top_level:
            issues.append("unknown local path keys: {}".format(", ".join(unknown_top_level)))

        raw_datasets = values.get("datasets", {})
        if not isinstance(raw_datasets, Mapping):
            issues.append("datasets must be a mapping")
            raw_datasets = {}
        unknown_datasets = sorted(set(raw_datasets) - set(DATASETS))
        if unknown_datasets:
            issues.append("unknown datasets: {}".format(", ".join(unknown_datasets)))

        datasets = {}
        for dataset, raw_path in raw_datasets.items():
            if dataset not in DATASETS:
                continue
            path = _absolute_path(raw_path, "datasets.{}".format(dataset), issues)
            if path is not None:
                if require_existing_data and not path.is_dir():
                    issues.append("datasets.{} directory does not exist".format(dataset))
                datasets[dataset] = path

        raw_output_root = values.get("output_root")
        output_root = None
        if raw_output_root is not None:
            output_root = _absolute_path(raw_output_root, "output_root", issues)

        if issues:
            raise ConfigValidationError(issues)
        return cls(datasets=MappingProxyType(datasets), output_root=output_root)


def load_local_paths(
    path: PathLike,
    environ: Optional[Mapping[str, str]] = None,
    require_existing_data: bool = True,
) -> LocalPaths:
    """Load ignored local paths, then apply explicit environment overrides."""

    config_path = Path(path).expanduser().resolve()
    values = load_yaml_mapping(config_path) if config_path.is_file() else {}
    merged: Dict[str, Any] = deepcopy(values)
    datasets = {
        key: value
        for key, value in dict(merged.get("datasets") or {}).items()
        if value is not None and str(value).strip()
    }
    environment = os.environ if environ is None else environ
    for dataset, environment_key in DATASET_ENVIRONMENT_KEYS.items():
        value = environment.get(environment_key)
        if value:
            datasets[dataset] = value
    if datasets:
        merged["datasets"] = datasets
    output_root = environment.get(OUTPUT_ENVIRONMENT_KEY)
    if output_root:
        merged["output_root"] = output_root

    if not merged:
        raise ConfigValidationError([
            "no local paths found; copy configs/local/paths.example.yaml to "
            "configs/local/paths.yaml or set an MMCTR_*_DATA_DIR environment variable"
        ])
    return LocalPaths.from_mapping(merged, require_existing_data=require_existing_data)


def resolve_dataset_config(
    dataset_name: str,
    dataset_config: Mapping[str, Any],
    project_root: PathLike,
    local_paths: Optional[LocalPaths] = None,
) -> Dict[str, Any]:
    """Return a copy with one unambiguous absolute dataset directory."""

    name = dataset_name.lower()
    if name not in DATASETS:
        raise ConfigValidationError(["unknown dataset: {}".format(name)])
    if not isinstance(dataset_config, Mapping):
        raise ConfigValidationError(["dataset config must be a mapping"])
    configured_name = dataset_config.get("name")
    if configured_name != name:
        raise ConfigValidationError([
            "dataset config name {!r} does not match {!r}".format(configured_name, name)
        ])

    resolved = deepcopy(dict(dataset_config))
    override = local_paths.datasets.get(name) if local_paths else None
    if override is not None:
        data_dir = override
    else:
        raw_data_dir = resolved.get("data_dir")
        if not isinstance(raw_data_dir, (str, Path)) or not str(raw_data_dir).strip():
            raise ConfigValidationError(["data_dir must be a non-empty path"])
        data_dir = Path(raw_data_dir).expanduser()
        if not data_dir.is_absolute():
            data_dir = Path(project_root).expanduser().resolve() / data_dir
        data_dir = data_dir.resolve()

    resolved["data_dir"] = str(data_dir)
    resolved["using_local_data"] = override is not None
    return resolved


def load_dataset_catalog(
    config_path: PathLike,
    dataset_name: str,
    project_root: PathLike,
    use_local_data: bool = False,
    local_paths_path: Optional[PathLike] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Load a dataset catalog and resolve the selected dataset's data directory."""

    catalog = load_yaml_mapping(config_path)
    name = dataset_name.lower()
    if name not in catalog:
        raise ConfigValidationError([
            "dataset {!r} is missing from {}".format(name, Path(config_path).resolve())
        ])
    local_paths = None
    if use_local_data:
        paths_path = local_paths_path or (
            Path(project_root).expanduser().resolve() / "configs/local/paths.yaml"
        )
        local_paths = load_local_paths(paths_path, environ=environ)
        if name not in local_paths.datasets:
            raise ConfigValidationError([
                "local path for dataset {!r} is missing".format(name)
            ])
    catalog[name] = resolve_dataset_config(
        name,
        catalog[name],
        project_root=project_root,
        local_paths=local_paths,
    )
    return catalog

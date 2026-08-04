"""Fusion sweeps over configurable components of production model implementations."""

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple, Union

from mmctr.config import load_yaml_mapping
from mmctr.core import ContractError
from mmctr.core.registry import RegistryError
from mmctr.experiments import ExperimentTask, load_task_matrix, save_task_matrix
from mmctr.models.common.components.fusion_registry import FUSION_REGISTRY
from mmctr.models.common.presets import default_pipeline_preset
from mmctr.models.common.registry import MODEL_REGISTRY


FUSION_STUDY_MODELS = frozenset({"dnn_mm", "dnn_mm_seq", "dmf", "make"})
FUSION_STUDY_MATRIX_SCHEMA = "fusion-study-matrix-v1"
_CONFIG_KEYS = frozenset(
    {
        "dataset",
        "data_fingerprint",
        "data",
        "model_configs",
        "models",
        "fusions",
        "seeds",
    }
)
PathLike = Union[str, Path]


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(name): _plain(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def build_fusion_study_tasks(
    dataset: str,
    data_fingerprint: str,
    data_config: Mapping[str, Any],
    model_configs: Mapping[str, Mapping[str, Any]],
    models: Sequence[str],
    fusions: Sequence[str],
    seeds: Sequence[int],
) -> Tuple[ExperimentTask, ...]:
    """Build ExperimentRunner tasks without copying or approximating model classes."""

    if not isinstance(data_config, Mapping) or not isinstance(model_configs, Mapping):
        raise ContractError("fusion study data/model configs must be mappings")
    try:
        model_names = tuple(MODEL_REGISTRY.canonical_name(name) for name in models)
        fusion_names = tuple(FUSION_REGISTRY.canonical_name(name) for name in fusions)
    except RegistryError as error:
        raise ContractError(str(error)) from error
    seed_values = tuple(int(seed) for seed in seeds)
    if (
        not model_names
        or len(set(model_names)) != len(model_names)
        or any(name not in FUSION_STUDY_MODELS for name in model_names)
    ):
        raise ContractError(
            "fusion study models must be unique supported production models: {}".format(
                sorted(FUSION_STUDY_MODELS)
            )
        )
    if not fusion_names or len(set(fusion_names)) != len(fusion_names):
        raise ContractError("fusion study components must be non-empty and unique")
    if (
        not seed_values
        or len(set(seed_values)) != len(seed_values)
        or any(isinstance(seed, bool) for seed in seeds)
    ):
        raise ContractError("fusion study seeds must be non-empty unique integers")
    if not isinstance(data_fingerprint, str) or not data_fingerprint:
        raise ContractError("fusion study data_fingerprint must be non-empty")

    plain_data = _plain(data_config)
    tasks = []
    for model_name in model_names:
        try:
            base_model_config = model_configs[model_name]
        except KeyError as error:
            raise ContractError(
                "fusion study is missing config for model {!r}".format(model_name)
            ) from error
        if not isinstance(base_model_config, Mapping):
            raise ContractError("fusion study model config must be a mapping")
        for fusion_name in fusion_names:
            configured_model: Dict[str, Any] = _plain(base_model_config)
            configured_model["modal_fusion_method"] = {
                "concatenate": "cat",
                "sum": "add",
            }.get(fusion_name, fusion_name)
            preset = default_pipeline_preset(model_name, configured_model, plain_data)
            if not preset.executable:
                raise ContractError(
                    "fusion study cannot approximate model-specific pipeline {!r}".format(
                        model_name
                    )
                )
            for seed in seed_values:
                task_id = "{}-{}-seed-{}".format(model_name, fusion_name, seed)
                tasks.append(
                    ExperimentTask(
                        task_id=task_id,
                        dataset=str(dataset),
                        model=model_name,
                        seed=seed,
                        resolved_config={
                            "data_fingerprint": data_fingerprint,
                            "data": plain_data,
                            "model": configured_model,
                            "analysis": {
                                "protocol": "fusion-component-sweep-v1",
                                "fusion": fusion_name,
                            },
                        },
                    )
                )
    return tuple(tasks)


def load_fusion_study_config(path: PathLike) -> Tuple[ExperimentTask, ...]:
    """Load one strict YAML study definition into canonical experiment tasks."""

    values = load_yaml_mapping(path)
    keys = set(values)
    missing = sorted(_CONFIG_KEYS - keys)
    unknown = sorted(keys - _CONFIG_KEYS)
    if missing or unknown:
        problems = []
        if missing:
            problems.append("missing keys: {}".format(", ".join(missing)))
        if unknown:
            problems.append("unknown keys: {}".format(", ".join(unknown)))
        raise ContractError("invalid fusion study config ({})".format("; ".join(problems)))
    return build_fusion_study_tasks(
        dataset=values["dataset"],
        data_fingerprint=values["data_fingerprint"],
        data_config=values["data"],
        model_configs=values["model_configs"],
        models=values["models"],
        fusions=values["fusions"],
        seeds=values["seeds"],
    )


def save_fusion_study_matrix(
    tasks: Sequence[ExperimentTask],
    path: PathLike,
) -> Path:
    """Atomically save a versioned matrix that can be loaded without legacy code."""

    return save_task_matrix(tasks, path, FUSION_STUDY_MATRIX_SCHEMA)


def load_fusion_study_matrix(path: PathLike) -> Tuple[ExperimentTask, ...]:
    """Load and verify a saved fusion matrix."""

    return load_task_matrix(path, FUSION_STUDY_MATRIX_SCHEMA)


__all__ = [
    "FUSION_STUDY_MATRIX_SCHEMA",
    "FUSION_STUDY_MODELS",
    "build_fusion_study_tasks",
    "load_fusion_study_config",
    "load_fusion_study_matrix",
    "save_fusion_study_matrix",
]

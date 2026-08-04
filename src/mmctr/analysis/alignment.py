"""Representation hooks and auxiliary losses for model-independent alignment studies."""

import math
from itertools import combinations
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as functional

from mmctr.config import load_yaml_mapping
from mmctr.core import ContractError, ModelOutput
from mmctr.core.registry import RegistryError
from mmctr.experiments import ExperimentTask, load_task_matrix, save_task_matrix
from mmctr.models.common.registry import MODEL_REGISTRY


ALIGNMENT_STUDY_MATRIX_SCHEMA = "alignment-study-matrix-v1"
_CONFIG_KEYS = frozenset(
    {
        "dataset",
        "data_fingerprint",
        "data",
        "model_configs",
        "models",
        "methods",
        "weights",
        "representations",
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


def build_alignment_study_tasks(
    dataset: str,
    data_fingerprint: str,
    data_config: Mapping[str, Any],
    model_configs: Mapping[str, Mapping[str, Any]],
    models: Sequence[str],
    methods: Sequence[str],
    weights: Sequence[float],
    representations: Mapping[str, str],
    seeds: Sequence[int],
) -> Tuple[ExperimentTask, ...]:
    """Build tasks that inject alignment through the canonical auxiliary-loss protocol."""

    if not isinstance(data_config, Mapping) or not isinstance(model_configs, Mapping):
        raise ContractError("alignment study data/model configs must be mappings")
    try:
        model_names = tuple(MODEL_REGISTRY.canonical_name(name) for name in models)
    except RegistryError as error:
        raise ContractError(str(error)) from error
    if not model_names or len(set(model_names)) != len(model_names):
        raise ContractError("alignment study models must be non-empty and unique")
    method_names = tuple(str(method).lower() for method in methods)
    if (
        not method_names
        or any(method not in {"cosine", "mse"} for method in method_names)
        or len(set(method_names)) != len(method_names)
    ):
        raise ContractError("alignment methods must be unique cosine or mse names")
    weight_values = tuple(float(weight) for weight in weights)
    if (
        not weight_values
        or any(isinstance(weight, bool) for weight in weights)
        or any(not math.isfinite(weight) or weight < 0.0 for weight in weight_values)
        or len(set(weight_values)) != len(weight_values)
    ):
        raise ContractError("alignment weights must be unique finite non-negative values")
    if not isinstance(representations, Mapping):
        raise ContractError("alignment representations must be a mapping")
    representation_paths = {
        str(name): str(module_path) for name, module_path in representations.items()
    }
    if (
        len(representation_paths) < 2
        or any(not name or not path for name, path in representation_paths.items())
        or len(set(representation_paths.values())) != len(representation_paths)
    ):
        raise ContractError("alignment requires unique names for at least two module paths")
    seed_values = tuple(int(seed) for seed in seeds)
    if (
        not seed_values
        or any(isinstance(seed, bool) for seed in seeds)
        or len(set(seed_values)) != len(seed_values)
    ):
        raise ContractError("alignment seeds must be non-empty unique integers")
    if not isinstance(data_fingerprint, str) or not data_fingerprint:
        raise ContractError("alignment data_fingerprint must be non-empty")

    plain_data = _plain(data_config)
    tasks = []
    for model_name in model_names:
        try:
            model_config = model_configs[model_name]
        except KeyError as error:
            raise ContractError(
                "alignment study is missing config for model {!r}".format(model_name)
            ) from error
        if not isinstance(model_config, Mapping):
            raise ContractError("alignment study model config must be a mapping")
        for method in method_names:
            for weight in weight_values:
                weight_name = ("{:.6f}".format(weight)).rstrip("0").rstrip(".") or "0"
                for seed in seed_values:
                    tasks.append(
                        ExperimentTask(
                            task_id="{}-{}-weight-{}-seed-{}".format(
                                model_name, method, weight_name.replace(".", "p"), seed
                            ),
                            dataset=str(dataset),
                            model=model_name,
                            seed=seed,
                            resolved_config={
                                "data_fingerprint": data_fingerprint,
                                "data": plain_data,
                                "model": _plain(model_config),
                                "analysis": {
                                    "protocol": "representation-alignment-v1",
                                    "method": method,
                                    "weight": weight,
                                    "representations": representation_paths,
                                },
                            },
                        )
                    )
    return tuple(tasks)


def load_alignment_study_config(path: PathLike) -> Tuple[ExperimentTask, ...]:
    """Load a strict YAML alignment definition into canonical tasks."""

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
        raise ContractError("invalid alignment study config ({})".format("; ".join(problems)))
    return build_alignment_study_tasks(
        dataset=values["dataset"],
        data_fingerprint=values["data_fingerprint"],
        data_config=values["data"],
        model_configs=values["model_configs"],
        models=values["models"],
        methods=values["methods"],
        weights=values["weights"],
        representations=values["representations"],
        seeds=values["seeds"],
    )


def save_alignment_study_matrix(tasks: Sequence[ExperimentTask], path: PathLike) -> Path:
    """Atomically save a canonical alignment task matrix."""

    return save_task_matrix(tasks, path, ALIGNMENT_STUDY_MATRIX_SCHEMA)


def load_alignment_study_matrix(path: PathLike) -> Tuple[ExperimentTask, ...]:
    """Load and verify a canonical alignment task matrix."""

    return load_task_matrix(path, ALIGNMENT_STUDY_MATRIX_SCHEMA)


class ActivationCapture:
    """Temporarily capture named module outputs without wrapping or copying a model."""

    def __init__(self, model: torch.nn.Module, modules: Mapping[str, str]) -> None:
        if not isinstance(model, torch.nn.Module):
            raise ContractError("activation capture requires a torch module")
        configured = {str(name): str(path) for name, path in modules.items()}
        if (
            len(configured) < 1
            or any(not name or not path for name, path in configured.items())
            or len(configured) != len(modules)
        ):
            raise ContractError("captured representation names and module paths must be unique")
        self.model = model
        self.modules = MappingProxyType(configured)
        self._values: Dict[str, torch.Tensor] = {}
        self._handles: Tuple[Any, ...] = ()

    @property
    def values(self) -> Mapping[str, torch.Tensor]:
        return MappingProxyType(dict(self._values))

    def _hook(self, name: str) -> Any:
        def capture(_module: torch.nn.Module, _inputs: Any, output: Any) -> None:
            if not isinstance(output, torch.Tensor):
                raise ContractError("captured module {!r} must return one tensor".format(name))
            self._values[name] = output

        return capture

    def __enter__(self) -> "ActivationCapture":
        if self._handles:
            raise ContractError("activation capture context is already active")
        self._values.clear()
        handles = []
        try:
            for name, path in self.modules.items():
                module = self.model.get_submodule(path)
                handles.append(module.register_forward_hook(self._hook(name)))
        except (AttributeError, KeyError) as error:
            for handle in handles:
                handle.remove()
            raise ContractError("captured module path does not exist") from error
        self._handles = tuple(handles)
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = ()


class AlignmentAuxiliary:
    """Attach a named pairwise representation objective to a canonical output."""

    def __init__(self, method: str, weight: float) -> None:
        canonical_method = str(method).lower()
        if canonical_method not in {"cosine", "mse"}:
            raise ContractError("alignment method must be cosine or mse")
        if isinstance(weight, bool) or not math.isfinite(float(weight)) or float(weight) < 0.0:
            raise ContractError("alignment weight must be a finite non-negative value")
        self.method = canonical_method
        self.weight = float(weight)

    def attach(
        self,
        output: ModelOutput,
        representations: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> ModelOutput:
        if not isinstance(output, ModelOutput):
            raise ContractError("alignment auxiliary requires a canonical ModelOutput")
        selected = output.representations if representations is None else representations
        if selected is None or len(selected) < 2:
            raise ContractError("alignment requires at least two named representations")
        values = {str(name): value for name, value in selected.items()}
        if len(values) != len(selected) or any(not name for name in values):
            raise ContractError("alignment representation names must be unique and non-empty")
        tensors = tuple(values.values())
        if any(not isinstance(value, torch.Tensor) for value in tensors):
            raise ContractError("alignment representations must be tensors")
        if any(value.shape != tensors[0].shape for value in tensors[1:]):
            raise ContractError("alignment representations must have the same shape")
        if tensors[0].ndim < 2 or tensors[0].shape[0] != output.batch_size:
            raise ContractError("alignment representations must start with the output batch size")
        if any(not value.is_floating_point() for value in tensors):
            raise ContractError("alignment representations must use floating dtypes")

        pair_losses = []
        for left, right in combinations(tensors, 2):
            if self.method == "cosine":
                flattened_left = left.flatten(start_dim=1)
                flattened_right = right.flatten(start_dim=1)
                pair_losses.append(
                    1.0
                    - functional.cosine_similarity(flattened_left, flattened_right, dim=-1).mean()
                )
            else:
                pair_losses.append(functional.mse_loss(left, right))
        loss = torch.stack(pair_losses).mean() * self.weight
        name = "alignment_{}".format(self.method)
        auxiliary_losses = dict(output.auxiliary_losses)
        if name in auxiliary_losses:
            raise ContractError("alignment auxiliary loss already exists")
        auxiliary_losses[name] = loss
        return ModelOutput(
            logits=output.logits,
            auxiliary_losses=auxiliary_losses,
            representations=output.representations,
        )


__all__ = [
    "ALIGNMENT_STUDY_MATRIX_SCHEMA",
    "ActivationCapture",
    "AlignmentAuxiliary",
    "build_alignment_study_tasks",
    "load_alignment_study_config",
    "load_alignment_study_matrix",
    "save_alignment_study_matrix",
]

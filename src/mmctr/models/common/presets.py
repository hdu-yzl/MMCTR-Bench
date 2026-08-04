"""Executable default modal-pipeline presets for canonical model behavior."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple

from mmctr.core import ContractError
from mmctr.models.common.components.pipeline import (
    ComponentConfig,
    ModalPipeline,
    ModalPipelineConfig,
    ModalPipelineSet,
)
from mmctr.models.common.registry import MODEL_REGISTRY


EXECUTABLE_PIPELINE_MODELS = frozenset(
    {
        "autoint",
        "dcn",
        "deepfm",
        "dmf",
        "dnn",
        "dnn_mm",
        "dnn_mm_seq",
        "lmf",
        "make",
        "mtfn",
        "naml",
        "qarm",
        "simcen",
    }
)

_MODEL_SPECIFIC_REASONS = MappingProxyType(
    {
        "din": "Dice normalization observes padded projection bias in the current model",
        "marn": "private/shared experts, gradient reversal, and gates are model-specific",
        "em3": "FQ-Former queries and CIC require registered paper components",
        "diff_msin": "SRC, expert gates, and label-aware objectives are model-specific",
        "gmmf": "DSN/CGAN auto-difference and user gating require registered paper fusion",
        "mb": "attention scoring and PGD modality balancing are model-specific",
        "pamd": "pairwise common/specific decomposition and reconstruction are model-specific",
        "mmmlp": "modality and fusion MLP-Mixer stacks are model-specific",
        "m3srec": "shared attention and specific/cross MoE stages are model-specific",
        "psrq": "frozen PSRQ encoding and code attention are owned by the quantized boundary",
    }
)


@dataclass(frozen=True)
class ModelPipelinePreset:
    """Resolved branch configs plus intentional parameter-sharing relationships."""

    model_name: str
    branches: Mapping[str, ModalPipelineConfig]
    shared_modules: Tuple[Tuple[str, str, str], ...] = ()
    model_specific_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ContractError("pipeline preset model name must be non-empty")
        branches = dict(self.branches)
        for branch, config in branches.items():
            if branch != config.branch:
                raise ContractError("pipeline preset branch key does not match configuration")
        if self.model_specific_reason is None and not branches:
            raise ContractError("executable pipeline preset requires at least one branch")
        if self.model_specific_reason is not None and branches:
            raise ContractError("model-specific pipeline preset cannot contain executable branches")
        object.__setattr__(self, "branches", MappingProxyType(branches))

    @property
    def executable(self) -> bool:
        return self.model_specific_reason is None

    def build(self) -> ModalPipelineSet:
        if not self.executable:
            raise ContractError(
                "model {!r} retains a model-specific pipeline: {}".format(
                    self.model_name, self.model_specific_reason
                )
            )
        pipelines = ModalPipelineSet(
            {branch: ModalPipeline(config) for branch, config in self.branches.items()}
        )
        for source_branch, target_branch, attribute in self.shared_modules:
            source = pipelines[source_branch]
            target = pipelines[target_branch]
            if attribute not in {"projector", "fusion"}:
                raise ContractError("unsupported shared pipeline module {!r}".format(attribute))
            setattr(target, attribute, getattr(source, attribute))
        return pipelines


def _names(config: Mapping[str, Any], key: str, fallback: Tuple[str, ...]) -> Tuple[str, ...]:
    raw = config.get(key, fallback)
    if isinstance(raw, str):
        raise ContractError("{} must be a sequence of modality names".format(key))
    names = tuple(str(name) for name in raw)
    if not names or any(not name for name in names) or len(set(names)) != len(names):
        raise ContractError("{} must contain unique non-empty modality names".format(key))
    return names


def _dimensions(
    names: Tuple[str, ...],
    configured: Mapping[str, Any],
    id_dimension: int,
) -> Dict[str, int]:
    dimensions = dict(configured)
    dimensions["id"] = id_dimension
    missing = [name for name in names if name not in dimensions]
    if missing:
        raise ContractError("missing pipeline dimensions: {}".format(missing))
    return {name: int(dimensions[name]) for name in names}


def _dnn_mm_preset(
    model_config: Mapping[str, Any], data_config: Mapping[str, Any]
) -> ModelPipelinePreset:
    latent_dim = int(model_config.get("latent_dim", 128))
    projection_dim = int(model_config.get("projection_dim", 128))
    target_names = _names(data_config, "use_mm_features", ("id",))
    history_names = _names(data_config, "use_mm_seq_features", target_names)
    target_dimensions = _dimensions(target_names, data_config.get("mm_dims", {}), latent_dim * 2)
    history_source = data_config.get("mm_seq_dims", data_config.get("mm_dims", {}))
    history_dimensions = _dimensions(history_names, history_source, latent_dim)
    fusion = _configured_fusion(model_config, "cat")
    branches = {
        "target": ModalPipelineConfig(
            branch="target",
            topology="feature_fusion",
            modalities=target_names,
            input_dimensions=target_dimensions,
            projection_dim=projection_dim,
            pooling=None,
            fusion=fusion,
        ),
        "history": ModalPipelineConfig(
            branch="history",
            topology="pool_then_fuse",
            modalities=history_names,
            input_dimensions=history_dimensions,
            projection_dim=projection_dim,
            pooling={name: ComponentConfig("mean") for name in history_names},
            fusion=fusion,
        ),
    }
    return ModelPipelinePreset("dnn_mm", branches)


def _pooled_id_preset(
    model_name: str,
    model_config: Mapping[str, Any],
) -> ModelPipelinePreset:
    latent_dim = int(model_config.get("latent_dim", 128))
    projection_dim = int(model_config.get("projection_dim", 128))
    fusion = ComponentConfig("sum")
    branches = {
        "target": ModalPipelineConfig(
            branch="target",
            topology="feature_fusion",
            modalities=("id",),
            input_dimensions={"id": latent_dim * 2},
            projection_dim=projection_dim,
            pooling=None,
            fusion=fusion,
        ),
        "history": ModalPipelineConfig(
            branch="history",
            topology="pool_then_fuse",
            modalities=("id",),
            input_dimensions={"id": latent_dim},
            projection_dim=projection_dim,
            pooling={"id": ComponentConfig("mean")},
            fusion=fusion,
        ),
    }
    return ModelPipelinePreset(model_name, branches)


def _simcen_preset(
    model_config: Mapping[str, Any], data_config: Mapping[str, Any]
) -> ModelPipelinePreset:
    configured = dict(model_config)
    configured["modal_fusion_method"] = "concatenate"
    dnn_preset = _dnn_mm_preset(configured, data_config)
    return ModelPipelinePreset("simcen", dnn_preset.branches)


def _pooled_fuse_then_pool_preset(
    model_name: str,
    model_config: Mapping[str, Any],
    data_config: Mapping[str, Any],
) -> ModelPipelinePreset:
    latent_dim = int(model_config.get("latent_dim", 128))
    projection_dim = int(model_config.get("projection_dim", 128))
    target_names = _names(data_config, "use_mm_features", ("id",))
    history_names = _names(data_config, "use_mm_seq_features", target_names)
    target_dimensions = _dimensions(target_names, data_config.get("mm_dims", {}), latent_dim * 2)
    history_source = data_config.get("mm_seq_dims", data_config.get("mm_dims", {}))
    history_dimensions = _dimensions(history_names, history_source, latent_dim)
    rank = int(model_config.get("rank", 5 if model_name == "lmf" else 20))
    options: Dict[str, Any] = {"rank": rank}
    if model_name == "lmf":
        options["output_dim"] = int(model_config.get("fusion_dim", 16))
    fusion = ComponentConfig(model_name, options)
    branches = {
        "target": ModalPipelineConfig(
            branch="target",
            topology="feature_fusion",
            modalities=target_names,
            input_dimensions=target_dimensions,
            projection_dim=projection_dim,
            pooling=None,
            fusion=fusion,
        ),
        "history": ModalPipelineConfig(
            branch="history",
            topology="fuse_then_pool",
            modalities=history_names,
            input_dimensions=history_dimensions,
            projection_dim=projection_dim,
            pooling=ComponentConfig("mean"),
            fusion=fusion,
        ),
    }
    return ModelPipelinePreset(model_name, branches)


def _sequence_branch_dimensions(
    model_config: Mapping[str, Any], data_config: Mapping[str, Any]
) -> Tuple[
    int,
    Tuple[str, ...],
    Dict[str, int],
    Tuple[str, ...],
    Dict[str, int],
]:
    latent_dim = int(model_config.get("latent_dim", 128))
    projection_dim = int(model_config.get("projection_dim", 128))
    item_names = _names(data_config, "use_mm_features", ("id",))
    item_source = data_config.get("mm_seq_dims", data_config.get("mm_dims", {}))
    item_dimensions = _dimensions(item_names, item_source, latent_dim)
    user_names = _names(data_config, "user_features", ("id",))
    user_dimensions = _dimensions(user_names, data_config.get("user_features_dim", {}), latent_dim)
    return projection_dim, item_names, item_dimensions, user_names, user_dimensions


def _qarm_preset(
    model_config: Mapping[str, Any], data_config: Mapping[str, Any]
) -> ModelPipelinePreset:
    selected = model_config.get(str(data_config.get("name", "")).lower(), model_config)
    if not isinstance(selected, Mapping):
        raise ContractError("QARM dataset-specific preset config must be a mapping")
    latent_dim = int(selected.get("latent_dim", 128))
    projection_dim = int(selected.get("projection_dim", 128))
    level_count = int(selected.get("n_levels", 3))
    item_names = _names(data_config, "use_mm_features", ("id",))
    item_dimensions = {
        name: latent_dim if name == "id" else level_count * latent_dim for name in item_names
    }
    user_names = _names(data_config, "user_features", ("id",))
    user_dimensions = _dimensions(user_names, data_config.get("user_features_dim", {}), latent_dim)
    concatenation = ComponentConfig("concatenate")
    branches = {
        "target": ModalPipelineConfig(
            branch="target",
            topology="feature_fusion",
            modalities=item_names,
            input_dimensions=item_dimensions,
            projection_dim=projection_dim,
            pooling=None,
            fusion=concatenation,
        ),
        "history": ModalPipelineConfig(
            branch="history",
            topology="pool_then_fuse",
            modalities=item_names,
            input_dimensions=item_dimensions,
            projection_dim=projection_dim,
            pooling={name: ComponentConfig("mean") for name in item_names},
            fusion=concatenation,
        ),
        "user": ModalPipelineConfig(
            branch="user",
            topology="feature_fusion",
            modalities=user_names,
            input_dimensions=user_dimensions,
            projection_dim=projection_dim,
            pooling=None,
            fusion=concatenation,
        ),
    }
    return ModelPipelinePreset(
        "qarm",
        branches,
        shared_modules=(("history", "target", "projector"),),
    )


def _dmf_preset(
    model_config: Mapping[str, Any], data_config: Mapping[str, Any]
) -> ModelPipelinePreset:
    (
        projection_dim,
        item_names,
        item_dimensions,
        user_names,
        user_dimensions,
    ) = _sequence_branch_dimensions(model_config, data_config)
    modal_names = tuple(name for name in item_names if name != "id")
    if not modal_names:
        raise ContractError("DMF pipeline requires at least one non-ID modality")
    modal_dimensions = {name: item_dimensions[name] for name in modal_names}
    fusion = _configured_fusion(model_config, "cat")
    branches = {
        "target": ModalPipelineConfig(
            branch="target",
            topology="feature_fusion",
            modalities=modal_names,
            input_dimensions=modal_dimensions,
            projection_dim=projection_dim,
            pooling=None,
            fusion=fusion,
        ),
        "history": ModalPipelineConfig(
            branch="history",
            topology="sequence_fusion",
            modalities=modal_names,
            input_dimensions=modal_dimensions,
            projection_dim=projection_dim,
            pooling=None,
            fusion=fusion,
        ),
        "user": ModalPipelineConfig(
            branch="user",
            topology="feature_fusion",
            modalities=user_names,
            input_dimensions=user_dimensions,
            projection_dim=projection_dim,
            pooling=None,
            fusion=ComponentConfig("concatenate"),
        ),
    }
    return ModelPipelinePreset(
        "dmf",
        branches,
        shared_modules=(
            ("history", "target", "projector"),
            ("target", "history", "fusion"),
        ),
    )


def _dnn_mm_seq_preset(
    model_config: Mapping[str, Any], data_config: Mapping[str, Any]
) -> ModelPipelinePreset:
    (
        projection_dim,
        item_names,
        item_dimensions,
        user_names,
        user_dimensions,
    ) = _sequence_branch_dimensions(model_config, data_config)
    fusion = _configured_fusion(model_config, "add")
    branches = {
        "target": ModalPipelineConfig(
            branch="target",
            topology="feature_fusion",
            modalities=item_names,
            input_dimensions=item_dimensions,
            projection_dim=projection_dim,
            pooling=None,
            fusion=fusion,
        ),
        "history": ModalPipelineConfig(
            branch="history",
            topology="pool_then_fuse",
            modalities=item_names,
            input_dimensions=item_dimensions,
            projection_dim=projection_dim,
            pooling={name: ComponentConfig("mean") for name in item_names},
            fusion=fusion,
        ),
        "user": ModalPipelineConfig(
            branch="user",
            topology="feature_fusion",
            modalities=user_names,
            input_dimensions=user_dimensions,
            projection_dim=projection_dim,
            pooling=None,
            fusion=ComponentConfig("concatenate"),
        ),
    }
    return ModelPipelinePreset(
        "dnn_mm_seq",
        branches,
        shared_modules=(("history", "target", "projector"),),
    )


def _naml_preset(
    model_config: Mapping[str, Any], data_config: Mapping[str, Any]
) -> ModelPipelinePreset:
    (
        projection_dim,
        item_names,
        item_dimensions,
        user_names,
        user_dimensions,
    ) = _sequence_branch_dimensions(model_config, data_config)
    fusion = ComponentConfig("maf")
    branches = {
        "target": ModalPipelineConfig(
            branch="target",
            topology="feature_fusion",
            modalities=item_names,
            input_dimensions=item_dimensions,
            projection_dim=projection_dim,
            pooling=None,
            fusion=fusion,
        ),
        "history": ModalPipelineConfig(
            branch="history",
            topology="fuse_then_pool",
            modalities=item_names,
            input_dimensions=item_dimensions,
            projection_dim=projection_dim,
            pooling=ComponentConfig("attention"),
            fusion=fusion,
        ),
        "user": ModalPipelineConfig(
            branch="user",
            topology="feature_fusion",
            modalities=user_names,
            input_dimensions=user_dimensions,
            projection_dim=projection_dim,
            pooling=None,
            fusion=ComponentConfig("concatenate"),
        ),
    }
    return ModelPipelinePreset(
        "naml",
        branches,
        shared_modules=(
            ("history", "target", "projector"),
            ("target", "history", "fusion"),
        ),
    )


def _configured_fusion(model_config: Mapping[str, Any], default: str) -> ComponentConfig:
    name = str(model_config.get("modal_fusion_method", default)).strip().lower()
    options: Dict[str, Any] = {}
    if name in {"lmf", "mtfn"}:
        options["rank"] = int(model_config.get("rank", 5 if name == "lmf" else 20))
    if name == "lmf":
        options["output_dim"] = int(model_config.get("fusion_dim", 16))
    return ComponentConfig(name, options)


def _make_preset(
    model_config: Mapping[str, Any], data_config: Mapping[str, Any]
) -> ModelPipelinePreset:
    (
        projection_dim,
        item_names,
        item_dimensions,
        user_names,
        user_dimensions,
    ) = _sequence_branch_dimensions(model_config, data_config)
    fusion = _configured_fusion(model_config, "cat")
    hidden_dims = tuple(int(value) for value in model_config.get("mlp_dims", (1024, 512, 256)))
    pooling = ComponentConfig(
        "din",
        {
            "hidden_dims": hidden_dims,
            "dropout": float(model_config.get("dropout", 0.5)),
        },
    )
    branches = {
        "target": ModalPipelineConfig(
            branch="target",
            topology="feature_fusion",
            modalities=item_names,
            input_dimensions=item_dimensions,
            projection_dim=projection_dim,
            pooling=None,
            fusion=fusion,
        ),
        "history": ModalPipelineConfig(
            branch="history",
            topology="fuse_then_pool",
            modalities=item_names,
            input_dimensions=item_dimensions,
            projection_dim=projection_dim,
            pooling=pooling,
            fusion=fusion,
        ),
        "user": ModalPipelineConfig(
            branch="user",
            topology="feature_fusion",
            modalities=user_names,
            input_dimensions=user_dimensions,
            projection_dim=projection_dim,
            pooling=None,
            fusion=ComponentConfig("concatenate"),
        ),
    }
    return ModelPipelinePreset(
        "make",
        branches,
        shared_modules=(
            ("history", "target", "projector"),
            ("target", "history", "fusion"),
        ),
    )


def default_pipeline_preset(
    model_name: str,
    model_config: Mapping[str, Any],
    data_config: Mapping[str, Any],
) -> ModelPipelinePreset:
    """Resolve one model's compatibility preset from validated model/data config."""

    name = MODEL_REGISTRY.canonical_name(model_name)
    if name in {"dnn", "dcn", "deepfm", "autoint"}:
        return _pooled_id_preset(name, model_config)
    if name == "dnn_mm":
        return _dnn_mm_preset(model_config, data_config)
    if name == "simcen":
        return _simcen_preset(model_config, data_config)
    if name == "qarm":
        return _qarm_preset(model_config, data_config)
    if name == "dmf":
        return _dmf_preset(model_config, data_config)
    if name == "dnn_mm_seq":
        return _dnn_mm_seq_preset(model_config, data_config)
    if name in {"lmf", "mtfn"}:
        return _pooled_fuse_then_pool_preset(name, model_config, data_config)
    if name == "naml":
        return _naml_preset(model_config, data_config)
    if name == "make":
        return _make_preset(model_config, data_config)
    try:
        reason = _MODEL_SPECIFIC_REASONS[name]
    except KeyError as error:
        raise ContractError(
            "registered model {!r} has no pipeline coverage decision".format(name)
        ) from error
    return ModelPipelinePreset(name, {}, model_specific_reason=reason)


def model_pipeline_coverage() -> Mapping[str, str]:
    """Return the complete explicit executable/model-specific coverage decision."""

    coverage = {name: "executable" for name in EXECUTABLE_PIPELINE_MODELS}
    coverage.update(_MODEL_SPECIFIC_REASONS)
    return MappingProxyType(coverage)


__all__ = [
    "EXECUTABLE_PIPELINE_MODELS",
    "ModelPipelinePreset",
    "default_pipeline_preset",
    "model_pipeline_coverage",
]

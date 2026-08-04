"""Typed composition of projection, pooling, fusion, and dimension adaptation."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch

from mmctr.core import ContractError
from mmctr.models.components.fusion import FusionOutput, ModalityFusion
from mmctr.models.components.fusion_registry import create_fusion
from mmctr.models.components.masking import apply_feature_mask, validate_sequence_mask
from mmctr.models.components.pooling import SequencePooling
from mmctr.models.components.pooling_registry import create_pooling
from mmctr.models.components.projection import DimensionAdapter, NamedFeatureProjector


@dataclass(frozen=True)
class ComponentConfig:
    """One registry name and constructor options from a resolved configuration."""

    name: str
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.name).strip().lower()
        if not name:
            raise ContractError("component name must be non-empty")
        if not isinstance(self.options, Mapping):
            raise ContractError("component options must be a mapping")
        options = dict(self.options)
        if any(not isinstance(key, str) or not key for key in options):
            raise ContractError("component option names must be non-empty strings")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "options", MappingProxyType(options))

    @classmethod
    def from_value(cls, value: Any, field_name: str) -> "ComponentConfig":
        """Parse one shorthand name or strict ``name/options`` mapping."""

        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(value)
        if not isinstance(value, Mapping):
            raise ContractError("{} must be a component name or mapping".format(field_name))
        unknown = set(value).difference({"name", "options"})
        if unknown:
            raise ContractError("unknown {} keys: {}".format(field_name, sorted(unknown)))
        if "name" not in value:
            raise ContractError("{}.name is required".format(field_name))
        return cls(value["name"], value.get("options", {}))


@dataclass(frozen=True)
class ModalPipelineConfig:
    """Resolved configuration for one target, history, or user modal branch."""

    branch: str
    topology: str
    modalities: Tuple[str, ...]
    input_dimensions: Mapping[str, int]
    projection_dim: int
    pooling: Optional[Union[ComponentConfig, Mapping[str, ComponentConfig]]]
    fusion: ComponentConfig
    output_dim: Optional[int] = None

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ModalPipelineConfig":
        """Parse a strict resolved YAML-like branch mapping without mutating it."""

        if not isinstance(values, Mapping):
            raise ContractError("modal pipeline config must be a mapping")
        allowed = {
            "branch",
            "topology",
            "modalities",
            "input_dimensions",
            "projection_dim",
            "pooling",
            "fusion",
            "output_dim",
        }
        unknown = set(values).difference(allowed)
        if unknown:
            raise ContractError("unknown modal pipeline config keys: {}".format(sorted(unknown)))
        required = {"branch", "modalities", "input_dimensions", "projection_dim", "fusion"}
        missing = required.difference(values)
        if missing:
            raise ContractError("missing modal pipeline config keys: {}".format(sorted(missing)))
        branch = str(values["branch"]).strip().lower()
        topology = values.get(
            "topology", "feature_fusion" if branch in {"target", "user"} else None
        )
        if topology is None:
            raise ContractError("history modal pipeline requires topology")
        modalities = values["modalities"]
        if isinstance(modalities, str) or not isinstance(modalities, Sequence):
            raise ContractError("modalities must be a sequence of names")
        raw_pooling = values.get("pooling")
        if str(topology).strip().lower() == "pool_then_fuse":
            if not isinstance(raw_pooling, Mapping):
                raise ContractError("pool_then_fuse pooling must be a modality mapping")
            pooling: Optional[Union[ComponentConfig, Mapping[str, ComponentConfig]]] = {
                str(name): ComponentConfig.from_value(component, "pooling.{}".format(name))
                for name, component in raw_pooling.items()
            }
        elif raw_pooling is None:
            pooling = None
        else:
            pooling = ComponentConfig.from_value(raw_pooling, "pooling")
        return cls(
            branch=branch,
            topology=str(topology),
            modalities=tuple(str(name) for name in modalities),
            input_dimensions=values["input_dimensions"],
            projection_dim=values["projection_dim"],
            pooling=pooling,
            fusion=ComponentConfig.from_value(values["fusion"], "fusion"),
            output_dim=values.get("output_dim"),
        )

    def __post_init__(self) -> None:
        branch = str(self.branch).strip().lower()
        topology = str(self.topology).strip().lower()
        modalities = tuple(str(name) for name in self.modalities)
        if branch not in {"target", "history", "user"}:
            raise ContractError("pipeline branch must be target, history, or user")
        history_topologies = {"pool_then_fuse", "fuse_then_pool", "sequence_fusion"}
        if branch == "history" and topology not in history_topologies:
            raise ContractError("unsupported history pipeline topology: {!r}".format(topology))
        if branch != "history" and topology != "feature_fusion":
            raise ContractError("target/user branches require feature_fusion topology")
        if not modalities or any(not name for name in modalities):
            raise ContractError("pipeline modalities must be non-empty names")
        if len(set(modalities)) != len(modalities):
            raise ContractError("pipeline modalities must be unique")
        dimensions = {str(name): int(value) for name, value in self.input_dimensions.items()}
        if set(dimensions) != set(modalities):
            raise ContractError("pipeline dimensions must match configured modalities")
        if any(value <= 0 for value in dimensions.values()):
            raise ContractError("pipeline input dimensions must be positive")
        projection_dim = int(self.projection_dim)
        if projection_dim <= 0:
            raise ContractError("pipeline projection dimension must be positive")
        if topology == "pool_then_fuse":
            if not isinstance(self.pooling, Mapping) or set(self.pooling) != set(modalities):
                raise ContractError("pool_then_fuse requires one pooling config per modality")
            modal_pooling = dict(self.pooling)
            if any(not isinstance(config, ComponentConfig) for config in modal_pooling.values()):
                raise ContractError("pipeline pooling values must be ComponentConfig objects")
            pooling: Optional[Union[ComponentConfig, Mapping[str, ComponentConfig]]] = (
                MappingProxyType(modal_pooling)
            )
        elif topology == "fuse_then_pool":
            if not isinstance(self.pooling, ComponentConfig):
                raise ContractError("fuse_then_pool requires one pooling ComponentConfig")
            pooling = self.pooling
        else:
            if self.pooling is not None:
                raise ContractError("fusion-only topology does not accept pooling configuration")
            pooling = None
        if not isinstance(self.fusion, ComponentConfig):
            raise ContractError("pipeline fusion must be a ComponentConfig")
        output_dim = None if self.output_dim is None else int(self.output_dim)
        if output_dim is not None and output_dim <= 0:
            raise ContractError("pipeline output dimension must be positive")
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "topology", topology)
        object.__setattr__(self, "modalities", modalities)
        object.__setattr__(self, "input_dimensions", MappingProxyType(dimensions))
        object.__setattr__(self, "projection_dim", projection_dim)
        object.__setattr__(self, "pooling", pooling)
        object.__setattr__(self, "output_dim", output_dim)


@dataclass(frozen=True)
class ModalPipelineOutput:
    """Final branch representation and named losses emitted while composing it."""

    representation: torch.Tensor
    presence: torch.Tensor
    auxiliary_losses: Mapping[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validated = FusionOutput(self.representation, self.auxiliary_losses)
        if not isinstance(self.presence, torch.Tensor) or self.presence.dtype != torch.bool:
            raise ContractError("pipeline output presence must be a boolean torch.Tensor")
        if tuple(self.presence.shape) != tuple(validated.fused.shape[:-1]):
            raise ContractError("pipeline output presence must match representation prefix")
        if self.presence.device != validated.fused.device:
            raise ContractError("pipeline output presence and representation must share a device")
        object.__setattr__(self, "representation", validated.fused)
        object.__setattr__(self, "presence", self.presence)
        object.__setattr__(self, "auxiliary_losses", validated.auxiliary_losses)


class ModalPipeline(torch.nn.Module):
    """Execute one validated modal branch without inspecting a canonical Batch."""

    def __init__(self, config: ModalPipelineConfig) -> None:
        super().__init__()
        if not isinstance(config, ModalPipelineConfig):
            raise ContractError("modal pipeline requires ModalPipelineConfig")
        self.config = config
        self.modalities = config.modalities
        self.projector = NamedFeatureProjector(
            config.input_dimensions,
            config.projection_dim,
            allowed_ranks=(2,) if config.topology == "feature_fusion" else (2, 3),
        )
        self.fusion: ModalityFusion = create_fusion(
            config.fusion.name,
            config.modalities,
            config.projection_dim,
            **dict(config.fusion.options),
        )
        self.pooling = torch.nn.ModuleDict()
        self.sequence_pooling: Optional[SequencePooling] = None
        if config.topology == "pool_then_fuse":
            if not isinstance(config.pooling, Mapping):
                raise ContractError("pool_then_fuse requires modal pooling configuration")
            self.pooling.update(
                {
                    name: create_pooling(
                        component.name, config.projection_dim, **dict(component.options)
                    )
                    for name, component in config.pooling.items()
                }
            )
        elif config.topology == "fuse_then_pool":
            if not isinstance(config.pooling, ComponentConfig):
                raise ContractError("fuse_then_pool requires sequence pooling configuration")
            self.sequence_pooling = create_pooling(
                config.pooling.name, self.fusion.output_dim, **dict(config.pooling.options)
            )
        self.output_dim = self.fusion.output_dim if config.output_dim is None else config.output_dim
        self.adapter = DimensionAdapter(self.fusion.output_dim, self.output_dim)
        if self.sequence_pooling is not None and self.sequence_pooling.capability.target_required:
            self.target_modalities = self.modalities
        elif config.topology == "pool_then_fuse":
            self.target_modalities = tuple(
                name
                for name in self.modalities
                if isinstance(self.pooling[name], SequencePooling)
                and self.pooling[name].capability.target_required
            )
        else:
            self.target_modalities = ()

    def _effective_presence(
        self,
        values: Mapping[str, torch.Tensor],
        presence: Optional[Mapping[str, torch.Tensor]],
        sequence_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if not isinstance(values, Mapping) or set(values) != set(self.modalities):
            raise ContractError("pipeline values must contain exactly {}".format(self.modalities))
        first = values[self.modalities[0]]
        validate_sequence_mask(sequence_mask, first, "sequence_mask")
        if presence is not None and set(presence) != set(self.modalities):
            raise ContractError("pipeline presence must contain exactly {}".format(self.modalities))
        masks: Dict[str, torch.Tensor] = {}
        for name in self.modalities:
            value = values[name]
            validate_sequence_mask(sequence_mask, value, "sequence_mask")
            modality_mask = sequence_mask if presence is None else presence[name]
            validate_sequence_mask(modality_mask, value, "presence.{}".format(name))
            masks[name] = sequence_mask & modality_mask
        return masks

    def forward(
        self,
        values: Mapping[str, torch.Tensor],
        presence: Optional[Mapping[str, torch.Tensor]] = None,
        sequence_mask: Optional[torch.Tensor] = None,
        targets: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> ModalPipelineOutput:
        if self.config.topology == "feature_fusion":
            if sequence_mask is not None or targets is not None:
                raise ContractError("feature_fusion does not accept sequence_mask or targets")
            projected = self.projector(values, presence)
            fused = self.fusion(projected, presence)
            if presence is None:
                valid_rows = torch.ones(
                    fused.fused.shape[0], dtype=torch.bool, device=fused.fused.device
                )
            else:
                valid_rows = torch.stack(tuple(presence[name] for name in self.modalities)).any(
                    dim=0
                )
            representation = apply_feature_mask(self.adapter(fused.fused), valid_rows)
            return ModalPipelineOutput(representation, valid_rows, fused.auxiliary_losses)
        if sequence_mask is None:
            raise ContractError("history modal pipeline requires sequence_mask")
        masks = self._effective_presence(values, presence, sequence_mask)
        projected = self.projector(values, masks)
        if not self.target_modalities and targets is not None:
            raise ContractError("configured pipeline does not use targets")
        if self.target_modalities and targets is None:
            raise ContractError(
                "pipeline targets are required for modalities {}".format(self.target_modalities)
            )
        projected_targets: Dict[str, torch.Tensor] = {}
        if targets is not None:
            if not isinstance(targets, Mapping) or set(targets) != set(self.modalities):
                raise ContractError(
                    "pipeline targets must contain exactly {}".format(self.modalities)
                )
            projected_targets = self.projector(targets)
        if self.config.topology == "pool_then_fuse":
            pooled: Dict[str, torch.Tensor] = {}
            for name in self.modalities:
                component = self.pooling[name]
                if not isinstance(component, SequencePooling):
                    raise ContractError("registered pooling must implement SequencePooling")
                pooled[name] = component(projected[name], masks[name], projected_targets.get(name))
            row_presence = {name: mask.any(dim=1) for name, mask in masks.items()}
            fused = self.fusion(pooled, row_presence)
            valid_rows = torch.stack(tuple(row_presence.values()), dim=0).any(dim=0)
        elif self.config.topology == "fuse_then_pool":
            fused = self.fusion(projected, masks)
            valid_tokens = torch.stack(tuple(masks.values()), dim=0).any(dim=0)
            target = None
            if projected_targets:
                target = self.fusion(projected_targets).fused
            if self.sequence_pooling is None:
                raise ContractError("fuse_then_pool has no sequence pooling component")
            pooled_fused = self.sequence_pooling(fused.fused, valid_tokens, target)
            fused = FusionOutput(pooled_fused, fused.auxiliary_losses)
            valid_rows = valid_tokens.any(dim=1)
            representation = apply_feature_mask(self.adapter(fused.fused), valid_rows)
            return ModalPipelineOutput(representation, valid_rows, fused.auxiliary_losses)
        else:
            fused = self.fusion(projected, masks)
            valid_tokens = torch.stack(tuple(masks.values()), dim=0).any(dim=0)
            representation = apply_feature_mask(self.adapter(fused.fused), valid_tokens)
            return ModalPipelineOutput(representation, valid_tokens, fused.auxiliary_losses)
        representation = apply_feature_mask(self.adapter(fused.fused), valid_rows)
        return ModalPipelineOutput(representation, valid_rows, fused.auxiliary_losses)


class ModalPipelineSet(torch.nn.ModuleDict):
    """A strict named collection for separately configured model branches."""

    def __init__(self, pipelines: Mapping[str, ModalPipeline]) -> None:
        if not isinstance(pipelines, Mapping) or not pipelines:
            raise ContractError("modal pipeline set must contain at least one branch")
        invalid = set(pipelines).difference({"target", "history", "user"})
        if invalid:
            raise ContractError("unknown modal pipeline branches: {}".format(sorted(invalid)))
        for branch, pipeline in pipelines.items():
            if not isinstance(pipeline, ModalPipeline):
                raise ContractError("modal pipeline set values must be ModalPipeline objects")
            if pipeline.config.branch != branch:
                raise ContractError("modal pipeline branch key does not match its configuration")
        super().__init__(dict(pipelines))
        self.branches = tuple(pipelines)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ModalPipelineSet":
        """Build separate branches from the resolved ``modal_pipeline`` mapping."""

        if not isinstance(values, Mapping) or not values:
            raise ContractError("modal_pipeline must be a non-empty branch mapping")
        pipelines: Dict[str, ModalPipeline] = {}
        for branch, raw_config in values.items():
            if branch not in {"target", "history", "user"}:
                raise ContractError("unknown modal pipeline branch {!r}".format(branch))
            if not isinstance(raw_config, Mapping):
                raise ContractError("modal pipeline branch config must be a mapping")
            configured = dict(raw_config)
            explicit_branch = configured.get("branch", branch)
            if explicit_branch != branch:
                raise ContractError("modal pipeline branch key does not match explicit branch")
            configured["branch"] = branch
            pipelines[branch] = ModalPipeline(ModalPipelineConfig.from_mapping(configured))
        return cls(pipelines)

    def __getitem__(self, branch: str) -> ModalPipeline:
        try:
            pipeline = super().__getitem__(branch)
        except KeyError as error:
            raise ContractError("unknown modal pipeline branch {!r}".format(branch)) from error
        if not isinstance(pipeline, ModalPipeline):
            raise ContractError("registered branch is not a ModalPipeline")
        return pipeline


__all__ = [
    "ComponentConfig",
    "ModalPipeline",
    "ModalPipelineConfig",
    "ModalPipelineOutput",
    "ModalPipelineSet",
]

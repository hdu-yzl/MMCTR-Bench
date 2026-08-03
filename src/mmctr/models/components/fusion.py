"""Named multimodal fusion components with explicit output contracts."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch

from mmctr.core import ContractError
from mmctr.models.components.masking import apply_feature_mask


@dataclass(frozen=True)
class FusionCapability:
    """Declarative constraints exposed to configuration and pipeline validation."""

    allowed_ranks: Tuple[int, ...] = (2, 3)
    minimum_modalities: int = 1
    maximum_modalities: Optional[int] = None
    presence_supported: bool = True
    output_dim_rule: str = "preserves"
    auxiliary_loss_names: Tuple[str, ...] = ()


@dataclass(frozen=True)
class FusionOutput:
    """One fused representation and stable, named scalar auxiliary losses."""

    fused: torch.Tensor
    auxiliary_losses: Mapping[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.fused, torch.Tensor) or not self.fused.is_floating_point():
            raise ContractError("fused representation must be a floating torch.Tensor")
        if self.fused.ndim not in (2, 3):
            raise ContractError("fused representation must have rank 2 or 3")
        copied: Dict[str, torch.Tensor] = {}
        for name, loss in self.auxiliary_losses.items():
            if not isinstance(name, str) or not name:
                raise ContractError("fusion auxiliary-loss names must be non-empty strings")
            if not isinstance(loss, torch.Tensor) or not loss.is_floating_point():
                raise ContractError("fusion auxiliary losses must be floating tensors")
            if loss.ndim != 0:
                raise ContractError("fusion auxiliary loss {!r} must be scalar".format(name))
            if loss.device != self.fused.device:
                raise ContractError("fusion output and auxiliary losses must share a device")
            copied[name] = loss
        object.__setattr__(self, "auxiliary_losses", MappingProxyType(copied))

    def total_auxiliary_loss(self) -> torch.Tensor:
        if not self.auxiliary_losses:
            return self.fused.new_zeros(())
        return torch.stack(tuple(self.auxiliary_losses.values())).sum()


class ModalityFusion(torch.nn.Module):
    """Base class validating an exact named mapping of equally sized modalities."""

    capability = FusionCapability()

    def __init__(
        self,
        features: Sequence[str],
        input_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.features = tuple(str(name) for name in features)
        if not self.features or any(not name for name in self.features):
            raise ContractError("fusion features must be non-empty names")
        if len(set(self.features)) != len(self.features):
            raise ContractError("fusion feature names must be unique")
        count = len(self.features)
        if count < self.capability.minimum_modalities:
            raise ContractError("fusion requires more modalities")
        maximum = self.capability.maximum_modalities
        if maximum is not None and count > maximum:
            raise ContractError("fusion received too many modalities")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        if self.input_dim <= 0 or self.output_dim <= 0:
            raise ContractError("fusion input/output dimensions must be positive")

    def _validate_inputs(
        self,
        values: Mapping[str, torch.Tensor],
        presence: Optional[Mapping[str, torch.Tensor]],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if not isinstance(values, Mapping) or set(values) != set(self.features):
            raise ContractError("fusion values must contain exactly {}".format(self.features))
        if presence is not None and set(presence) != set(self.features):
            raise ContractError("fusion presence must contain exactly {}".format(self.features))
        checked: Dict[str, torch.Tensor] = {}
        checked_presence: Dict[str, torch.Tensor] = {}
        prefix = None
        dtype = None
        device = None
        for name in self.features:
            value = values[name]
            if not isinstance(value, torch.Tensor) or not value.is_floating_point():
                raise ContractError("fusion inputs must be floating torch.Tensors")
            if value.ndim not in self.capability.allowed_ranks:
                raise ContractError("fusion input rank is not supported")
            if value.shape[-1] != self.input_dim:
                raise ContractError(
                    "fusion expected final dimension {}, got {}".format(
                        self.input_dim, value.shape[-1]
                    )
                )
            current_prefix = tuple(value.shape[:-1])
            if prefix is None:
                prefix, dtype, device = current_prefix, value.dtype, value.device
            elif current_prefix != prefix or value.dtype != dtype or value.device != device:
                raise ContractError("fusion modalities must share prefix shape, dtype, and device")
            mask = (
                torch.ones(current_prefix, dtype=torch.bool, device=value.device)
                if presence is None
                else presence[name]
            )
            if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
                raise ContractError("fusion presence values must be boolean tensors")
            if tuple(mask.shape) != current_prefix or mask.device != value.device:
                raise ContractError("fusion presence must match the modality prefix and device")
            checked[name] = apply_feature_mask(value, mask)
            checked_presence[name] = mask
        return checked, checked_presence


class ConcatenateFusion(ModalityFusion):
    capability = FusionCapability(output_dim_rule="modalities_times_input")

    def __init__(self, features: Sequence[str], dimension: int) -> None:
        names = tuple(features)
        super().__init__(names, dimension, dimension * len(names))

    def forward(
        self,
        values: Mapping[str, torch.Tensor],
        presence: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> FusionOutput:
        checked, _ = self._validate_inputs(values, presence)
        return FusionOutput(torch.cat([checked[name] for name in self.features], dim=-1))


class SumFusion(ModalityFusion):
    def __init__(self, features: Sequence[str], dimension: int) -> None:
        super().__init__(features, dimension, dimension)

    def forward(
        self,
        values: Mapping[str, torch.Tensor],
        presence: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> FusionOutput:
        checked, _ = self._validate_inputs(values, presence)
        return FusionOutput(torch.stack([checked[name] for name in self.features], dim=0).sum(0))


class MeanFusion(ModalityFusion):
    def __init__(self, features: Sequence[str], dimension: int) -> None:
        super().__init__(features, dimension, dimension)

    def forward(
        self,
        values: Mapping[str, torch.Tensor],
        presence: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> FusionOutput:
        checked, _ = self._validate_inputs(values, presence)
        return FusionOutput(torch.stack([checked[name] for name in self.features], dim=0).mean(0))


class MAFFusion(ModalityFusion):
    def __init__(self, features: Sequence[str], dimension: int) -> None:
        super().__init__(features, dimension, dimension)
        self.weights = torch.nn.ParameterDict(
            {name: torch.nn.Parameter(torch.empty(dimension, dimension)) for name in self.features}
        )
        self.biases = torch.nn.ParameterDict(
            {name: torch.nn.Parameter(torch.zeros(dimension)) for name in self.features}
        )
        for weight in self.weights.values():
            torch.nn.init.xavier_uniform_(weight)

    def forward(
        self,
        values: Mapping[str, torch.Tensor],
        presence: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> FusionOutput:
        checked, masks = self._validate_inputs(values, presence)
        transformed = [
            apply_feature_mask(
                torch.tanh(checked[name] @ self.weights[name] + self.biases[name]),
                masks[name],
            )
            for name in self.features
        ]
        return FusionOutput(torch.stack(transformed, dim=0).sum(0))


class LowRankFusion(ModalityFusion):
    capability = FusionCapability(output_dim_rule="configured")

    def __init__(
        self,
        features: Sequence[str],
        dimension: int,
        rank: int = 5,
        output_dim: int = 16,
    ) -> None:
        super().__init__(features, dimension, output_dim)
        self.rank = int(rank)
        if self.rank <= 0:
            raise ContractError("low-rank fusion rank must be positive")
        self.factors = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.empty(self.rank, dimension + 1, output_dim))
                for _ in self.features
            ]
        )
        self.fusion_weights = torch.nn.Parameter(torch.empty(1, self.rank))
        self.fusion_bias = torch.nn.Parameter(torch.zeros(1, output_dim))
        for factor in self.factors:
            torch.nn.init.xavier_normal_(factor)
        torch.nn.init.xavier_normal_(self.fusion_weights)

    def forward(
        self,
        values: Mapping[str, torch.Tensor],
        presence: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> FusionOutput:
        checked, _ = self._validate_inputs(values, presence)
        terms = []
        for name, factor in zip(self.features, self.factors):
            value = checked[name]
            ones = torch.ones(*value.shape[:-1], 1, dtype=value.dtype, device=value.device)
            augmented = torch.cat([ones, value], dim=-1)
            terms.append(torch.einsum("rdo,...d->...ro", factor, augmented))
        product = torch.stack(terms, dim=0).prod(dim=0)
        fused = (product * self.fusion_weights.unsqueeze(-1)).sum(dim=-2) + self.fusion_bias
        return FusionOutput(fused)


class MTFNFusion(ModalityFusion):
    def __init__(
        self,
        features: Sequence[str],
        dimension: int,
        rank: int = 20,
    ) -> None:
        super().__init__(features, dimension, dimension)
        self.rank = int(rank)
        if self.rank <= 0:
            raise ContractError("MTFN fusion rank must be positive")
        self.heads = torch.nn.ModuleDict(
            {
                name: torch.nn.ModuleList(
                    [torch.nn.Linear(dimension, dimension, bias=False) for _ in range(self.rank)]
                )
                for name in self.features
            }
        )
        self.compress = torch.nn.Linear(dimension, dimension)

    def forward(
        self,
        values: Mapping[str, torch.Tensor],
        presence: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> FusionOutput:
        checked, _ = self._validate_inputs(values, presence)
        rank_outputs = []
        for rank_index in range(self.rank):
            modal_values = [self.heads[name][rank_index](checked[name]) for name in self.features]
            fused = modal_values[0]
            for value in modal_values[1:]:
                fused = fused * value
            rank_outputs.append(fused)
        return FusionOutput(self.compress(torch.stack(rank_outputs, dim=0).sum(dim=0)))


__all__ = [
    "ConcatenateFusion",
    "FusionCapability",
    "FusionOutput",
    "LowRankFusion",
    "MAFFusion",
    "MTFNFusion",
    "MeanFusion",
    "ModalityFusion",
    "SumFusion",
]

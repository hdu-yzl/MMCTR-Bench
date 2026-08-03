"""Named feature projection and explicit dimension adapters."""

from types import MappingProxyType
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch

from mmctr.core import ContractError
from mmctr.models.components.masking import apply_feature_mask


def _normalise_ranks(allowed_ranks: Sequence[int]) -> Tuple[int, ...]:
    ranks = tuple(sorted(set(int(rank) for rank in allowed_ranks)))
    if not ranks or any(rank < 2 for rank in ranks):
        raise ContractError("allowed_ranks must contain ranks greater than or equal to two")
    return ranks


def _validate_dimensions(dimensions: Mapping[str, int]) -> Dict[str, int]:
    if not isinstance(dimensions, Mapping) or not dimensions:
        raise ContractError("feature dimensions must be a non-empty mapping")
    checked: Dict[str, int] = {}
    for name, dimension in dimensions.items():
        if not isinstance(name, str) or not name:
            raise ContractError("feature names must be non-empty strings")
        value = int(dimension)
        if value <= 0:
            raise ContractError("feature dimension for {!r} must be positive".format(name))
        checked[name] = value
    return checked


class DimensionAdapter(torch.nn.Module):
    """Validate tensor rank/dtype/dimension and adapt only the final dimension."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bias: bool = True,
        allowed_ranks: Sequence[int] = (2, 3),
        identity_if_same: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        if self.input_dim <= 0 or self.output_dim <= 0:
            raise ContractError("adapter input/output dimensions must be positive")
        self.allowed_ranks = _normalise_ranks(allowed_ranks)
        self.is_identity = bool(identity_if_same and self.input_dim == self.output_dim)
        self.adapter = (
            torch.nn.Identity()
            if self.is_identity
            else torch.nn.Linear(self.input_dim, self.output_dim, bias=bias)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if not isinstance(values, torch.Tensor):
            raise ContractError("adapter input must be a torch.Tensor")
        if not values.is_floating_point():
            raise ContractError("adapter input must use a floating dtype")
        if values.ndim not in self.allowed_ranks:
            raise ContractError(
                "adapter input rank {} is not in {}".format(values.ndim, self.allowed_ranks)
            )
        if values.shape[-1] != self.input_dim:
            raise ContractError(
                "adapter expected final dimension {}, got {}".format(
                    self.input_dim, values.shape[-1]
                )
            )
        return self.adapter(values)


class _FeatureProjection(torch.nn.Linear):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bias: bool,
        allowed_ranks: Tuple[int, ...],
    ) -> None:
        super().__init__(input_dim, output_dim, bias=bias)
        self.allowed_ranks = allowed_ranks

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if not isinstance(values, torch.Tensor):
            raise ContractError("projection input must be a torch.Tensor")
        if not values.is_floating_point():
            raise ContractError("projection input must use a floating dtype")
        if values.ndim not in self.allowed_ranks:
            raise ContractError(
                "projection input rank {} is not in {}".format(
                    values.ndim, self.allowed_ranks
                )
            )
        if values.shape[-1] != self.in_features:
            raise ContractError(
                "projection expected final dimension {}, got {}".format(
                    self.in_features, values.shape[-1]
                )
            )
        return super().forward(values)


class NamedFeatureProjector(torch.nn.ModuleDict):
    """Project an exact named tensor mapping to one common final dimension."""

    def __init__(
        self,
        dimensions: Mapping[str, int],
        output_dim: int,
        bias: bool = True,
        allowed_ranks: Sequence[int] = (2, 3),
    ) -> None:
        super().__init__()
        checked = _validate_dimensions(dimensions)
        self.feature_names = tuple(checked)
        self._input_dimensions = dict(checked)
        self.input_dimensions = MappingProxyType(self._input_dimensions)
        self.output_dim = int(output_dim)
        if self.output_dim <= 0:
            raise ContractError("projection output dimension must be positive")
        self.allowed_ranks = _normalise_ranks(allowed_ranks)
        for name in self.feature_names:
            self[name] = _FeatureProjection(
                checked[name], self.output_dim, bias, self.allowed_ranks
            )

    def replace(self, name: str, input_dim: int, bias: bool = True) -> None:
        """Replace one learned projection while retaining its registered state key."""

        if name not in self._input_dimensions:
            raise ContractError("unknown projection feature {!r}".format(name))
        checked = int(input_dim)
        if checked <= 0:
            raise ContractError("replacement projection dimension must be positive")
        self._input_dimensions[name] = checked
        self[name] = _FeatureProjection(
            checked, self.output_dim, bias, self.allowed_ranks
        )

    def __getitem__(self, name: str) -> _FeatureProjection:
        try:
            return super().__getitem__(name)
        except KeyError as error:
            raise ContractError("unknown projection feature {!r}".format(name)) from error

    def forward(
        self,
        values: Mapping[str, torch.Tensor],
        presence: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        if not isinstance(values, Mapping):
            raise ContractError("projection values must be a mapping")
        if any(not isinstance(name, str) or not name for name in values):
            raise ContractError("projection value keys must be non-empty strings")
        actual = set(values)
        expected = set(self.feature_names)
        if actual != expected:
            raise ContractError(
                "projection features must match exactly; missing={}, unknown={}".format(
                    sorted(expected.difference(actual)), sorted(actual.difference(expected))
                )
            )
        if presence is not None and not isinstance(presence, Mapping):
            raise ContractError("projection presence must be a mapping")
        masks = {} if presence is None else dict(presence)
        if any(not isinstance(name, str) or not name for name in masks):
            raise ContractError("projection presence keys must be non-empty strings")
        unknown_masks = set(masks).difference(expected)
        if unknown_masks:
            raise ContractError(
                "unknown projection presence masks: {}".format(sorted(unknown_masks))
            )
        projected: Dict[str, torch.Tensor] = {}
        for name in self.feature_names:
            output = self[name](values[name])
            if name in masks:
                output = apply_feature_mask(output, masks[name])
            projected[name] = output
        return projected


__all__ = ["DimensionAdapter", "NamedFeatureProjector"]

"""Strict mask utilities shared by canonical model components."""

from typing import Optional

import torch

from mmctr.core import ContractError


def _require_float_tensor(values: torch.Tensor, name: str, minimum_rank: int) -> None:
    if not isinstance(values, torch.Tensor):
        raise ContractError("{} must be a torch.Tensor".format(name))
    if not values.is_floating_point():
        raise ContractError("{} must use a floating dtype".format(name))
    if values.ndim < minimum_rank:
        raise ContractError("{} must have rank at least {}".format(name, minimum_rank))
    if values.shape[-1] <= 0:
        raise ContractError("{} must have a non-empty feature dimension".format(name))


def validate_sequence_mask(
    mask: torch.Tensor,
    sequence: Optional[torch.Tensor] = None,
    name: str = "mask",
) -> torch.Tensor:
    """Validate a boolean ``[B, L]`` mask and optional ``[B, L, ...]`` tensor."""

    if not isinstance(mask, torch.Tensor):
        raise ContractError("{} must be a torch.Tensor".format(name))
    if mask.dtype != torch.bool or mask.ndim != 2:
        raise ContractError("{} must be a bool tensor with shape [B, L]".format(name))
    if sequence is not None:
        if not isinstance(sequence, torch.Tensor) or sequence.ndim < 3:
            raise ContractError("sequence must have shape [B, L, ...]")
        if tuple(sequence.shape[:2]) != tuple(mask.shape):
            raise ContractError("sequence and {} dimensions do not match".format(name))
        if sequence.device != mask.device:
            raise ContractError("sequence and {} must use the same device".format(name))
    return mask


def apply_sequence_mask(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero every padded position without modifying ``sequence``."""

    _require_float_tensor(sequence, "sequence", 3)
    validate_sequence_mask(mask, sequence)
    expanded = mask
    while expanded.ndim < sequence.ndim:
        expanded = expanded.unsqueeze(-1)
    return sequence * expanded.to(dtype=sequence.dtype)


def feature_presence(values: torch.Tensor) -> torch.Tensor:
    """Return a boolean mask for non-zero vectors along the final dimension."""

    _require_float_tensor(values, "feature values", 2)
    return values.abs().sum(dim=-1).ne(0)


def apply_feature_mask(values: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
    """Zero projected values where a boolean prefix-shaped presence mask is false."""

    _require_float_tensor(values, "feature values", 2)
    if not isinstance(presence, torch.Tensor):
        raise ContractError("feature presence must be a torch.Tensor")
    if presence.dtype != torch.bool:
        raise ContractError("feature presence must use torch.bool")
    if tuple(presence.shape) != tuple(values.shape[:-1]):
        raise ContractError(
            "feature presence shape {} does not match value prefix {}".format(
                tuple(presence.shape), tuple(values.shape[:-1])
            )
        )
    if presence.device != values.device:
        raise ContractError("feature values and presence must use the same device")
    return values * presence.unsqueeze(-1).to(dtype=values.dtype)


def masked_softmax(
    scores: torch.Tensor,
    mask: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """Apply softmax with masked entries and all-masked slices mapped to zero."""

    if not isinstance(scores, torch.Tensor) or not scores.is_floating_point():
        raise ContractError("scores must be a floating torch.Tensor")
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise ContractError("mask must be a boolean torch.Tensor")
    if scores.shape != mask.shape:
        raise ContractError("scores and mask must have identical shapes")
    if scores.ndim == 0:
        raise ContractError("scores and mask must have rank at least one")
    if scores.device != mask.device:
        raise ContractError("scores and mask must use the same device")
    if dim < -scores.ndim or dim >= scores.ndim:
        raise ContractError("softmax dimension is out of range")
    masked_scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(masked_scores, dim=dim)
    weights = weights * mask.to(dtype=weights.dtype)
    denominator = weights.sum(dim=dim, keepdim=True)
    epsilon = torch.finfo(weights.dtype).eps
    return weights / denominator.clamp_min(epsilon)


__all__ = [
    "apply_feature_mask",
    "apply_sequence_mask",
    "feature_presence",
    "masked_softmax",
    "validate_sequence_mask",
]

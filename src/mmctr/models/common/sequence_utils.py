"""Shared mask-aware sequence utilities for model implementations."""

import torch


def last_valid_token(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
    positions = positions.expand(mask.shape[0], -1).masked_fill(~mask, -1)
    indices = positions.max(dim=1).values
    gathered = values.gather(
        1,
        indices.clamp_min(0).view(-1, 1, 1).expand(-1, 1, values.shape[-1]),
    ).squeeze(1)
    return gathered.masked_fill(indices.eq(-1).unsqueeze(-1), 0.0)


__all__ = ["last_valid_token"]

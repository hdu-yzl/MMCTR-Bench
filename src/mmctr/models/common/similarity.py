"""Shared similarity aggregation helpers for sequence models."""

import torch

from mmctr.core import ContractError


class SimilarityTiers(torch.nn.Module):
    def __init__(self, tier_count: int, minimum: float = -1.0, maximum: float = 1.0) -> None:
        super().__init__()
        if tier_count <= 0 or minimum >= maximum:
            raise ContractError("similarity tiers require a positive count and valid range")
        self.tier_count = int(tier_count)
        self.register_buffer(
            "boundaries", torch.linspace(minimum, maximum, steps=self.tier_count + 1)
        )

    def forward(self, scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        indices = torch.bucketize(scores, self.boundaries, right=False)
        indices = torch.clamp(indices - 1, 0, self.tier_count - 1)
        counts = scores.new_zeros((scores.shape[0], self.tier_count))
        counts.scatter_add_(1, indices, mask.to(dtype=scores.dtype))
        return counts


__all__ = ["SimilarityTiers"]

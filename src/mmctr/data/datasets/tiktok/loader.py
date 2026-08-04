"""Canonical TikTok named-array loader."""

from mmctr.data.datasets.arrays import NamedArrayDatasetLoader


class TikTokLoader(NamedArrayDatasetLoader):
    """Load official-split interactions and their precomputed causal prefixes."""

    dataset_name = "tiktok"


__all__ = ["TikTokLoader"]

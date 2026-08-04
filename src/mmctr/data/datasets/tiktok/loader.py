"""Canonical TikTok named-array loader."""

from mmctr.data.datasets.arrays import NamedArrayDatasetLoader


class TikTokLoader(NamedArrayDatasetLoader):
    dataset_name = "tiktok"


__all__ = ["TikTokLoader"]

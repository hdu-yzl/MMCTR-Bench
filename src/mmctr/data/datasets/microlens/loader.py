"""Canonical MicroLens named-array loader."""

from mmctr.data.datasets.arrays import NamedArrayDatasetLoader


class MicroLensLoader(NamedArrayDatasetLoader):
    dataset_name = "microlens"


__all__ = ["MicroLensLoader"]

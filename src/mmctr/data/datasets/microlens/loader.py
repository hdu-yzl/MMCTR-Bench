"""Canonical MicroLens named-array loader."""

from mmctr.data.datasets.arrays import NamedArrayDatasetLoader


class MicroLensLoader(NamedArrayDatasetLoader):
    """Load the versioned row-level split without recomputing upstream histories."""

    dataset_name = "microlens"


__all__ = ["MicroLensLoader"]

"""Canonical lazy registry for dataset reader implementations."""

from typing import Any

from mmctr.core.registry import ComponentRegistry, ComponentSpec


DATASET_REGISTRY = ComponentRegistry("dataset")
DATASET_REGISTRY.register_many(
    [
        ComponentSpec(
            "antm2c",
            "mmctr.data.datasets.antm2c.canonical",
            "AntM2CCanonicalLoader",
        ),
        ComponentSpec("microlens", "mmctr.data.datasets.microlens.loader", "MicroLensLoader"),
        ComponentSpec("tiktok", "mmctr.data.datasets.tiktok.loader", "TikTokLoader"),
    ]
)


def available_datasets():
    return DATASET_REGISTRY.names()


def dataset_spec(name: str) -> ComponentSpec:
    return DATASET_REGISTRY.spec(name)


def resolve_data_loader_class(name: str):
    return DATASET_REGISTRY.resolve(name)


def create_data_loader(name: str, *args: Any, **kwargs: Any):
    return DATASET_REGISTRY.create(name, *args, **kwargs)


__all__ = [
    "DATASET_REGISTRY",
    "available_datasets",
    "create_data_loader",
    "dataset_spec",
    "resolve_data_loader_class",
]

"""Dependency-light registry for public sequence pooling components."""

from typing import Any, Mapping, Tuple

from mmctr.core.registry import ComponentRegistry, ComponentSpec


def _metadata(target_required: bool) -> Mapping[str, Any]:
    return {
        "input_rank": 3,
        "output_rank": 2,
        "mask_required": True,
        "target_required": target_required,
        "output_dim_rule": "preserves",
    }


POOLING_REGISTRY = ComponentRegistry("pooling")
POOLING_REGISTRY.register_many(
    [
        ComponentSpec(
            "mean",
            "mmctr.models.common.components.pooling",
            "MeanPooling",
            aliases=("average",),
            metadata=_metadata(False),
        ),
        ComponentSpec(
            "sum",
            "mmctr.models.common.components.pooling",
            "SumPooling",
            metadata=_metadata(False),
        ),
        ComponentSpec(
            "max",
            "mmctr.models.common.components.pooling",
            "MaxPooling",
            metadata=_metadata(False),
        ),
        ComponentSpec(
            "attention",
            "mmctr.models.common.components.pooling",
            "AttentionPooling",
            metadata=_metadata(False),
        ),
        ComponentSpec(
            "din",
            "mmctr.models.common.components.pooling",
            "DinPooling",
            metadata=_metadata(True),
        ),
        ComponentSpec(
            "cross_attention",
            "mmctr.models.common.components.pooling",
            "CrossAttentionPooling",
            metadata=_metadata(True),
        ),
    ]
)


def available_pooling() -> Tuple[str, ...]:
    return POOLING_REGISTRY.names()


def pooling_capabilities(name: str) -> Mapping[str, Any]:
    return POOLING_REGISTRY.spec(name).metadata


def resolve_pooling_class(name: str):
    return POOLING_REGISTRY.resolve(name)


def create_pooling(name: str, *args: Any, **kwargs: Any):
    return POOLING_REGISTRY.create(name, *args, **kwargs)


__all__ = [
    "POOLING_REGISTRY",
    "available_pooling",
    "create_pooling",
    "pooling_capabilities",
    "resolve_pooling_class",
]

"""Dependency-light registry for public multimodal fusion components."""

from typing import Any, Mapping, Tuple

from mmctr.core.registry import ComponentRegistry, ComponentSpec


def _metadata(output_dim_rule: str) -> Mapping[str, Any]:
    return {
        "allowed_ranks": (2, 3),
        "minimum_modalities": 1,
        "maximum_modalities": None,
        "presence_supported": True,
        "output_dim_rule": output_dim_rule,
        "auxiliary_loss_names": (),
    }


FUSION_REGISTRY = ComponentRegistry("fusion")
FUSION_REGISTRY.register_many(
    [
        ComponentSpec(
            "concatenate",
            "mmctr.models.components.fusion",
            "ConcatenateFusion",
            aliases=("cat",),
            metadata=_metadata("modalities_times_input"),
        ),
        ComponentSpec(
            "sum",
            "mmctr.models.components.fusion",
            "SumFusion",
            aliases=("add",),
            metadata=_metadata("preserves"),
        ),
        ComponentSpec(
            "mean",
            "mmctr.models.components.fusion",
            "MeanFusion",
            aliases=("average",),
            metadata=_metadata("preserves"),
        ),
        ComponentSpec(
            "maf",
            "mmctr.models.components.fusion",
            "MAFFusion",
            metadata=_metadata("preserves"),
        ),
        ComponentSpec(
            "lmf",
            "mmctr.models.components.fusion",
            "LowRankFusion",
            metadata=_metadata("configured"),
        ),
        ComponentSpec(
            "mtfn",
            "mmctr.models.components.fusion",
            "MTFNFusion",
            metadata=_metadata("preserves"),
        ),
    ]
)


def available_fusion() -> Tuple[str, ...]:
    return FUSION_REGISTRY.names()


def fusion_capabilities(name: str) -> Mapping[str, Any]:
    return FUSION_REGISTRY.spec(name).metadata


def resolve_fusion_class(name: str):
    return FUSION_REGISTRY.resolve(name)


def create_fusion(name: str, *args: Any, **kwargs: Any):
    return FUSION_REGISTRY.create(name, *args, **kwargs)


__all__ = [
    "FUSION_REGISTRY",
    "available_fusion",
    "create_fusion",
    "fusion_capabilities",
    "resolve_fusion_class",
]

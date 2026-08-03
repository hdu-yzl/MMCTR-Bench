"""Lazy registry for pretraining components that produce quantization artifacts."""

from typing import Any

from mmctr.core.registry import ComponentRegistry, ComponentSpec


QUANTIZER_REGISTRY = ComponentRegistry("quantizer")
QUANTIZER_REGISTRY.register_many(
    [
        ComponentSpec(
            "rq",
            "mmctr.quantization.residual",
            "ResidualQuantizer",
            metadata={
                "artifact_kind": "residual-quantizer",
                "legacy_module": "models.pre_models",
                "legacy_symbol": "RQ",
            },
        ),
        ComponentSpec(
            "psrq",
            "mmctr.quantization.psrq",
            "PSRQPretrainer",
            metadata={
                "artifact_kind": "progressive-semantic-residual-quantizer",
                "legacy_module": "models.pre_models",
                "legacy_symbol": "PSRQ",
            },
        ),
    ]
)


def available_quantizers():
    return QUANTIZER_REGISTRY.names()


def quantizer_spec(name: str) -> ComponentSpec:
    return QUANTIZER_REGISTRY.spec(name)


def resolve_quantizer_class(name: str):
    return QUANTIZER_REGISTRY.resolve(name)


def create_quantizer(name: str, *args: Any, **kwargs: Any):
    return QUANTIZER_REGISTRY.create(name, *args, **kwargs)


__all__ = [
    "QUANTIZER_REGISTRY",
    "available_quantizers",
    "create_quantizer",
    "quantizer_spec",
    "resolve_quantizer_class",
]

"""Lazy public quantization pretraining and artifact contracts."""

import importlib

from .registry import (
    QUANTIZER_REGISTRY,
    available_quantizers,
    create_quantizer,
    quantizer_spec,
    resolve_quantizer_class,
)


_LAZY_EXPORTS = {
    "ARTIFACT_FORMAT": ("mmctr.quantization.artifacts", "ARTIFACT_FORMAT"),
    "ARTIFACT_VERSION": ("mmctr.quantization.artifacts", "ARTIFACT_VERSION"),
    "QuantizationArtifactError": (
        "mmctr.quantization.artifacts",
        "QuantizationArtifactError",
    ),
    "load_quantization_artifact": (
        "mmctr.quantization.artifacts",
        "load_quantization_artifact",
    ),
    "psrq_artifact_path": ("mmctr.quantization.artifacts", "psrq_artifact_path"),
    "rq_artifact_path": ("mmctr.quantization.artifacts", "rq_artifact_path"),
    "save_quantization_artifact": (
        "mmctr.quantization.artifacts",
        "save_quantization_artifact",
    ),
    "load_model_quantization_dependencies": (
        "mmctr.quantization.loading",
        "load_model_quantization_dependencies",
    ),
    "PSRQOutput": ("mmctr.quantization.psrq", "PSRQOutput"),
    "PSRQPretrainer": ("mmctr.quantization.psrq", "PSRQPretrainer"),
    "ResidualQuantizer": ("mmctr.quantization.residual", "ResidualQuantizer"),
    "fit_psrq": ("mmctr.quantization.training", "fit_psrq"),
}


def __getattr__(name):
    try:
        module_name, symbol_name = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name)) from error
    value = getattr(importlib.import_module(module_name), symbol_name)
    globals()[name] = value
    return value


__all__ = [
    "ARTIFACT_FORMAT",
    "ARTIFACT_VERSION",
    "PSRQOutput",
    "PSRQPretrainer",
    "QUANTIZER_REGISTRY",
    "QuantizationArtifactError",
    "ResidualQuantizer",
    "available_quantizers",
    "create_quantizer",
    "fit_psrq",
    "load_model_quantization_dependencies",
    "load_quantization_artifact",
    "psrq_artifact_path",
    "quantizer_spec",
    "resolve_quantizer_class",
    "rq_artifact_path",
    "save_quantization_artifact",
]

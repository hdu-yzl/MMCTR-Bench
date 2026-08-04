"""Canonical lazy registry for production model implementations."""

from typing import Any

from mmctr.core.registry import ComponentRegistry, ComponentSpec


MODEL_REGISTRY = ComponentRegistry("model")
MODEL_REGISTRY.register_many(
    [
        ComponentSpec(
            "dnn",
            "mmctr.models.baseline.dnn",
            "DNN",
            metadata={
                "history": "pooled",
            },
        ),
        ComponentSpec(
            "dnn_mm",
            "mmctr.models.mm_models.dnn_mm",
            "DNN_mm",
            metadata={
                "history": "pooled",
            },
        ),
        ComponentSpec(
            "dnn_mm_seq",
            "mmctr.models.mm_models.dnn_mm_seq",
            "DNN_mm_seq",
            aliases=("dnn_seq",),
            metadata={
                "history": "sequence_tokens",
            },
        ),
        ComponentSpec(
            "dcn",
            "mmctr.models.baseline.dcn",
            "DCN",
            metadata={
                "history": "pooled",
            },
        ),
        ComponentSpec(
            "deepfm",
            "mmctr.models.baseline.deepfm",
            "DeepFM",
            metadata={
                "history": "pooled",
            },
        ),
        ComponentSpec(
            "din",
            "mmctr.models.baseline.din",
            "DIN",
            metadata={
                "history": "sequence_tokens",
            },
        ),
        ComponentSpec(
            "autoint",
            "mmctr.models.baseline.autoint",
            "AutoInt",
            metadata={
                "history": "pooled",
            },
        ),
        ComponentSpec(
            "lmf",
            "mmctr.models.mm_models.lmf",
            "LMF",
            metadata={
                "history": "pooled",
            },
        ),
        ComponentSpec(
            "diff_msin",
            "mmctr.models.mm_models.diff_msin",
            "Diff_MSIN",
            metadata={
                "history": "sequence_tokens",
                "forward_uses_labels": True,
            },
        ),
        ComponentSpec(
            "marn",
            "mmctr.models.mm_models.marn",
            "MARN",
            metadata={
                "history": "sequence_tokens",
            },
        ),
        ComponentSpec(
            "mtfn",
            "mmctr.models.mm_models.mtfn",
            "MTFN",
            metadata={
                "history": "pooled",
            },
        ),
        ComponentSpec(
            "dmf",
            "mmctr.models.mm_models.dmf",
            "DMF",
            metadata={
                "history": "sequence_tokens",
            },
        ),
        ComponentSpec(
            "simcen",
            "mmctr.models.mm_models.simcen",
            "SimCEN",
            metadata={
                "history": "pooled",
            },
        ),
        ComponentSpec(
            "naml",
            "mmctr.models.mm_models.naml",
            "NAML",
            metadata={
                "history": "sequence_tokens",
            },
        ),
        ComponentSpec(
            "make",
            "mmctr.models.mm_models.make",
            "MAKE",
            metadata={
                "history": "sequence_tokens",
            },
        ),
        ComponentSpec(
            "em3",
            "mmctr.models.mm_models.em3",
            "EM3",
            metadata={
                "history": "sequence_tokens",
            },
        ),
        ComponentSpec(
            "gmmf",
            "mmctr.models.mm_models.gmmf",
            "GMMF",
            metadata={
                "history": "sequence_tokens",
                "alternating_optimization": "gmmf_gan",
            },
        ),
        ComponentSpec(
            "qarm",
            "mmctr.models.mm_models.qarm",
            "QARM",
            metadata={
                "history": "sequence_tokens",
                "quantization_artifacts": "rq_per_modality",
            },
        ),
        ComponentSpec(
            "psrq",
            "mmctr.models.mm_models.psrq",
            "MCCA",
            aliases=("mcca",),
            metadata={
                "history": "sequence_tokens",
                "quantization_artifacts": "psrq",
            },
        ),
        ComponentSpec(
            "mb",
            "mmctr.models.mm_models.mb",
            "MB",
            metadata={
                "history": "pooled",
                "forward_uses_labels": True,
            },
        ),
        ComponentSpec(
            "pamd",
            "mmctr.models.mm_models.pamd",
            "PAMD",
            metadata={
                "history": "pooled",
            },
        ),
        ComponentSpec(
            "mmmlp",
            "mmctr.models.mm_models.mmmlp",
            "MMMLP",
            metadata={
                "history": "sequence_tokens",
            },
        ),
        ComponentSpec(
            "m3srec",
            "mmctr.models.mm_models.m3srec",
            "M3SRec",
            metadata={
                "history": "sequence_tokens",
            },
        ),
    ]
)


def available_models():
    return MODEL_REGISTRY.names()


def model_spec(name: str) -> ComponentSpec:
    return MODEL_REGISTRY.spec(name)


def resolve_model_class(name: str):
    return MODEL_REGISTRY.resolve(name)


def create_model(name: str, *args: Any, **kwargs: Any):
    return MODEL_REGISTRY.create(name, *args, **kwargs)


def create_model_from_artifacts(
    name: str,
    model_config,
    data_config,
    artifact_root,
):
    """Compose a pure CTR model with validated pretrained quantization artifacts."""

    spec = model_spec(name)
    if "quantization_artifacts" not in spec.metadata:
        return create_model(name, model_config, data_config)
    from mmctr.quantization import load_model_quantization_dependencies

    dependencies = load_model_quantization_dependencies(
        spec.name, model_config, data_config, artifact_root
    )
    return create_model(name, model_config, data_config, **dependencies)


__all__ = [
    "MODEL_REGISTRY",
    "available_models",
    "create_model",
    "create_model_from_artifacts",
    "model_spec",
    "resolve_model_class",
]

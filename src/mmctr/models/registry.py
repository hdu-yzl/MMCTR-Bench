"""Canonical lazy registry for production and compatibility model implementations."""

from typing import Any

from mmctr.core.registry import ComponentRegistry, ComponentSpec


MODEL_REGISTRY = ComponentRegistry("model")
MODEL_REGISTRY.register_many(
    [
        ComponentSpec(
            "dnn",
            "mmctr.models.baselines",
            "DNN",
            metadata={
                "history": "pooled",
                "legacy_module": "models.ctr_models",
                "legacy_symbol": "DNN",
            },
        ),
        ComponentSpec(
            "dnn_mm",
            "mmctr.models.multimodal",
            "DNN_mm",
            metadata={
                "history": "pooled",
                "legacy_module": "models.ctr_models",
                "legacy_symbol": "DNN_mm",
            },
        ),
        ComponentSpec(
            "dnn_mm_seq",
            "mmctr.models.sequence",
            "DNN_mm_seq",
            aliases=("dnn_seq",),
            metadata={
                "history": "sequence_tokens",
                "legacy_module": "models.ctr_models",
                "legacy_symbol": "DNN_mm_seq",
            },
        ),
        ComponentSpec(
            "dcn",
            "mmctr.models.baselines",
            "DCN",
            metadata={
                "history": "pooled",
                "legacy_module": "models.ctr_models",
                "legacy_symbol": "DCN",
            },
        ),
        ComponentSpec(
            "deepfm",
            "mmctr.models.baselines",
            "DeepFM",
            metadata={
                "history": "pooled",
                "legacy_module": "models.ctr_models",
                "legacy_symbol": "DeepFM",
            },
        ),
        ComponentSpec(
            "din",
            "mmctr.models.baselines",
            "DIN",
            metadata={
                "history": "sequence_tokens",
                "legacy_module": "models.ctr_models",
                "legacy_symbol": "DIN",
            },
        ),
        ComponentSpec(
            "autoint",
            "mmctr.models.baselines",
            "AutoInt",
            metadata={
                "history": "pooled",
                "legacy_module": "models.ctr_models",
                "legacy_symbol": "AutoInt",
            },
        ),
        ComponentSpec(
            "lmf",
            "mmctr.models.multimodal",
            "LMF",
            metadata={
                "history": "pooled",
                "legacy_module": "models.mm_ctr_models",
                "legacy_symbol": "LMF",
            },
        ),
        ComponentSpec(
            "diff_msin",
            "models.mm_ctr_models",
            "Diff_MSIN",
            metadata={"history": "sequence_tokens", "forward_uses_labels": True},
        ),
        ComponentSpec(
            "marn",
            "mmctr.models.sequence",
            "MARN",
            metadata={
                "history": "sequence_tokens",
                "legacy_module": "models.mm_ctr_models",
                "legacy_symbol": "MARN",
            },
        ),
        ComponentSpec(
            "mtfn",
            "mmctr.models.multimodal",
            "MTFN",
            metadata={
                "history": "pooled",
                "legacy_module": "models.mm_ctr_models",
                "legacy_symbol": "MTFN",
            },
        ),
        ComponentSpec(
            "dmf",
            "mmctr.models.sequence",
            "DMF",
            metadata={
                "history": "sequence_tokens",
                "legacy_module": "models.mm_ctr_models",
                "legacy_symbol": "DMF",
            },
        ),
        ComponentSpec(
            "simcen",
            "mmctr.models.multimodal",
            "SimCEN",
            metadata={
                "history": "pooled",
                "legacy_module": "models.mm_ctr_models",
                "legacy_symbol": "SimCEN",
            },
        ),
        ComponentSpec(
            "naml",
            "mmctr.models.sequence",
            "NAML",
            metadata={
                "history": "sequence_tokens",
                "legacy_module": "models.mm_ctr_models",
                "legacy_symbol": "NAML",
            },
        ),
        ComponentSpec(
            "make",
            "mmctr.models.sequence",
            "MAKE",
            metadata={
                "history": "sequence_tokens",
                "legacy_module": "models.mm_ctr_models",
                "legacy_symbol": "MAKE",
            },
        ),
        ComponentSpec(
            "em3", "models.mm_ctr_models", "EM3", metadata={"history": "sequence_tokens"}
        ),
        ComponentSpec(
            "gmmf", "models.mm_ctr_models", "GMMF", metadata={"history": "sequence_tokens"}
        ),
        ComponentSpec(
            "qarm", "models.mm_ctr_models", "QARM", metadata={"history": "sequence_tokens"}
        ),
        ComponentSpec(
            "mcca", "models.mm_ctr_models", "MCCA", metadata={"history": "sequence_tokens"}
        ),
        ComponentSpec("mb", "models.mm_ctr_models", "MB", metadata={"history": "pooled"}),
        ComponentSpec("pamd", "models.mm_ctr_models", "PAMD", metadata={"history": "pooled"}),
        ComponentSpec(
            "mmmlp", "models.mm_ctr_models", "MMMLP", metadata={"history": "sequence_tokens"}
        ),
        ComponentSpec(
            "m3srec",
            "models.mm_ctr_models",
            "M3SRec",
            metadata={"history": "sequence_tokens"},
        ),
        ComponentSpec(
            "rq",
            "models.pre_models",
            "RQ",
            metadata={"history": "none", "takes_logger": False, "pretraining": True},
        ),
        ComponentSpec(
            "psrq",
            "models.pre_models",
            "PSRQ",
            metadata={"history": "none", "takes_logger": False, "pretraining": True},
        ),
    ]
)


def available_models():
    return MODEL_REGISTRY.names()


def model_spec(name: str) -> ComponentSpec:
    return MODEL_REGISTRY.spec(name)


def resolve_model_class(name: str):
    return MODEL_REGISTRY.resolve(name)


def resolve_legacy_model_class(name: str):
    """Resolve the compatibility implementation retained for historical entry points."""

    import importlib

    spec = model_spec(name)
    module_name = spec.metadata.get("legacy_module", spec.module)
    symbol_name = spec.metadata.get("legacy_symbol", spec.symbol)
    return getattr(importlib.import_module(module_name), symbol_name)


def create_model(name: str, *args: Any, **kwargs: Any):
    return MODEL_REGISTRY.create(name, *args, **kwargs)


def create_canonical_model(name: str, *args: Any, **kwargs: Any):
    """Construct a registered model and adapt legacy forward signatures when required."""

    from mmctr.models.base import BaseSeqModel, HistoryCapability
    from mmctr.models.compat import LegacyModelAdapter

    spec = model_spec(name)
    model = create_model(name, *args, **kwargs)
    if isinstance(model, BaseSeqModel):
        return model
    history = spec.metadata.get("history", "sequence_tokens")
    capability = (
        HistoryCapability.POOLED_HISTORY
        if history == "pooled"
        else HistoryCapability.SEQUENCE_TOKENS
    )
    return LegacyModelAdapter(
        model,
        capability,
        forward_uses_labels=bool(spec.metadata.get("forward_uses_labels", False)),
    )


__all__ = [
    "MODEL_REGISTRY",
    "available_models",
    "create_canonical_model",
    "create_model",
    "model_spec",
    "resolve_legacy_model_class",
    "resolve_model_class",
]

"""Lazy public model exports backed by the current legacy implementations."""

import importlib


_MODEL_EXPORTS = {
    "DNN": "dnn",
    "DNN_mm": "dnn_mm",
    "DNN_mm_seq": "dnn_mm_seq",
    "DCN": "dcn",
    "DeepFM": "deepfm",
    "DIN": "din",
    "AutoInt": "autoint",
    "LMF": "lmf",
    "Diff_MSIN": "diff_msin",
    "MARN": "marn",
    "MTFN": "mtfn",
    "DMF": "dmf",
    "SimCEN": "simcen",
    "NAML": "naml",
    "MAKE": "make",
    "EM3": "em3",
    "GMMF": "gmmf",
    "QARM": "qarm",
    "MCCA": "mcca",
    "MB": "mb",
    "PAMD": "pamd",
    "MMMLP": "mmmlp",
    "M3SRec": "m3srec",
    "RQ": "rq",
    "PSRQ": "psrq",
}

_BASE_EXPORTS = {
    "BaseModel": ("models.base_model", "BaseModel"),
    "BaseSeqModel": ("models.base_seq_model", "BaseSeqModel"),
}


def available_models():
    return tuple(sorted(set(_MODEL_EXPORTS.values())))


def get_model(*args, **kwargs):
    from mmctr.utils.helper import get_model as legacy_get_model

    return legacy_get_model(*args, **kwargs)


def __getattr__(name):
    if name in _BASE_EXPORTS:
        module_name, class_name = _BASE_EXPORTS[name]
        model_class = getattr(importlib.import_module(module_name), class_name)
        globals()[name] = model_class
        return model_class
    try:
        model_name = _MODEL_EXPORTS[name]
    except KeyError as error:
        raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name)) from error
    from mmctr.utils.helper import resolve_model_class

    model_class = resolve_model_class(model_name)
    globals()[name] = model_class
    return model_class


__all__ = [
    "BaseModel",
    "BaseSeqModel",
    "available_models",
    "get_model",
] + list(_MODEL_EXPORTS)

"""Lazy exports for ID-only recommendation models."""

import importlib


_EXPORTS = {
    "AutoInt": ("mmctr.models.baseline.autoint", "AutoInt"),
    "DCN": ("mmctr.models.baseline.dcn", "DCN"),
    "DIN": ("mmctr.models.baseline.din", "DIN"),
    "DNN": ("mmctr.models.baseline.dnn", "DNN"),
    "DeepFM": ("mmctr.models.baseline.deepfm", "DeepFM"),
}


def __getattr__(name):
    try:
        module_name, symbol_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name)) from error
    value = getattr(importlib.import_module(module_name), symbol_name)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORTS)

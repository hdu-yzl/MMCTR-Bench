"""Public package namespace for MMCTR-Bench."""

import importlib

__version__ = "0.1.0"

_CORE_EXPORTS = {"Batch", "ContractError", "ModelOutput", "RunResult"}


def __getattr__(name):
    if name not in _CORE_EXPORTS:
        raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
    value = getattr(importlib.import_module("mmctr.core"), name)
    globals()[name] = value
    return value


__all__ = ["__version__"] + sorted(_CORE_EXPORTS)

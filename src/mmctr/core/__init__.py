"""Lazy public core contracts for MMCTR-Bench."""

import importlib


_EXPORTS = {
    "Batch": ("mmctr.core.schemas", "Batch"),
    "ContractError": ("mmctr.core.schemas", "ContractError"),
    "ModelOutput": ("mmctr.core.schemas", "ModelOutput"),
    "RunResult": ("mmctr.core.schemas", "RunResult"),
    "ensure_model_output": ("mmctr.core.schemas", "ensure_model_output"),
    "ComponentRegistry": ("mmctr.core.registry", "ComponentRegistry"),
    "ComponentSpec": ("mmctr.core.registry", "ComponentSpec"),
    "RegistryError": ("mmctr.core.registry", "RegistryError"),
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

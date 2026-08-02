"""Dependency-light data catalog with lazy runtime contract exports."""

import importlib

from .registry import available_datasets, create_data_loader, resolve_data_loader_class


_LOADER_EXPORTS = {
    "Antm2cLoader": "antm2c",
    "MicrolensLoader": "microlens",
    "TiktokLoader": "tiktok",
}
_PUBLIC_EXPORTS = {
    "CanonicalDataLoader": ("mmctr.data.loader", "CanonicalDataLoader"),
    "DataLoaderProtocol": ("mmctr.data.loader", "DataLoaderProtocol"),
    "HistoryMode": ("mmctr.data.loader", "HistoryMode"),
    "adapt_legacy_loader": ("mmctr.data.loader", "adapt_legacy_loader"),
    "DatasetManifest": ("mmctr.data.manifest", "DatasetManifest"),
    "SplitStatistics": ("mmctr.data.manifest", "SplitStatistics"),
}


def get_data_loader(*args, **kwargs):
    return create_data_loader(*args, **kwargs)


def __getattr__(name):
    if name in _PUBLIC_EXPORTS:
        module_name, symbol_name = _PUBLIC_EXPORTS[name]
        value = getattr(importlib.import_module(module_name), symbol_name)
        globals()[name] = value
        return value
    try:
        dataset_name = _LOADER_EXPORTS[name]
    except KeyError as error:
        raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name)) from error
    loader_class = resolve_data_loader_class(dataset_name)
    globals()[name] = loader_class
    return loader_class


__all__ = ["available_datasets", "get_data_loader"] + sorted(
    list(_LOADER_EXPORTS) + list(_PUBLIC_EXPORTS)
)

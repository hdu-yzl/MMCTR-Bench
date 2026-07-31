"""Lazy public dataset-loader exports backed by legacy adapters."""

from mmctr.utils.helper import get_data_loader, resolve_data_loader_class


_LOADER_EXPORTS = {
    "Antm2cLoader": "antm2c",
    "MicrolensLoader": "microlens",
    "TiktokLoader": "tiktok",
}


def __getattr__(name):
    try:
        dataset_name = _LOADER_EXPORTS[name]
    except KeyError as error:
        raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name)) from error
    loader_class = resolve_data_loader_class(dataset_name)
    globals()[name] = loader_class
    return loader_class


__all__ = ["get_data_loader"] + list(_LOADER_EXPORTS)

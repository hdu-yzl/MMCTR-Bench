"""Lazy public dataset-loader exports backed by legacy adapters."""

_LOADER_EXPORTS = {
    "Antm2cLoader": "antm2c",
    "MicrolensLoader": "microlens",
    "TiktokLoader": "tiktok",
}


def available_datasets():
    return tuple(sorted(set(_LOADER_EXPORTS.values())))


def get_data_loader(*args, **kwargs):
    from mmctr.utils.helper import get_data_loader as legacy_get_data_loader

    return legacy_get_data_loader(*args, **kwargs)


def __getattr__(name):
    try:
        dataset_name = _LOADER_EXPORTS[name]
    except KeyError as error:
        raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name)) from error
    from mmctr.utils.helper import resolve_data_loader_class

    loader_class = resolve_data_loader_class(dataset_name)
    globals()[name] = loader_class
    return loader_class


__all__ = ["available_datasets", "get_data_loader"] + list(_LOADER_EXPORTS)

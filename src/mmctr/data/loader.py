"""Native data-loader contract consumed by training."""

from typing import Any, Iterable, Protocol

from mmctr.core import Batch, ContractError
from .manifest import DatasetManifest


class DataLoaderProtocol(Protocol):
    dataset_name: str
    manifest: DatasetManifest

    def iter_batches(self, split: str) -> Iterable[Batch]: ...


def ensure_canonical_loader(dataset_name: str, source: Any) -> DataLoaderProtocol:
    """Validate a native loader and preserve its audited manifest and identity."""
    manifest = getattr(source, "manifest", None)
    if not isinstance(manifest, DatasetManifest) or not callable(
        getattr(source, "iter_batches", None)
    ):
        raise ContractError(
            "dataset loader must expose DatasetManifest and iter_batches(split) yielding Batch"
        )
    if (
        getattr(source, "dataset_name", None) != dataset_name.lower()
        or manifest.name != dataset_name.lower()
    ):
        raise ContractError("canonical loader dataset identity does not match request")
    return source


__all__ = ["DataLoaderProtocol", "ensure_canonical_loader"]

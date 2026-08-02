"""Canonical data-loader API and adapters for existing loaders."""

from enum import Enum
from typing import Any, Iterable, Iterator, Mapping, Optional, Protocol, Sequence

import torch

from mmctr.core import Batch, ContractError

from .manifest import DatasetManifest, KNOWN_SPLITS


class HistoryMode(str, Enum):
    """History representation requested from a dataset adapter."""

    POOLED_COMPAT = "pooled_compat"
    SEQUENCE_TOKENS = "sequence_tokens"


class DataLoaderProtocol(Protocol):
    """Interface consumed by the training engine."""

    dataset_name: str
    manifest: DatasetManifest

    def iter_batches(self, split: str) -> Iterable[Batch]:
        ...


def _normalise_split(split: str) -> str:
    value = str(split).lower()
    if value == "validation":
        value = "val"
    if value not in KNOWN_SPLITS:
        raise ContractError("split must be one of train, val, or test")
    return value


def _split_legacy_ids(values: Sequence[Any]) -> Sequence[Any]:
    """Separate legacy combined ``[user_id, item_id, ...]`` target IDs."""

    if len(values) != 3:
        return values
    item_features, history_features, labels = values
    if not isinstance(item_features, Mapping):
        return values
    combined_ids = item_features.get("id")
    if not isinstance(combined_ids, torch.Tensor):
        return values
    if combined_ids.ndim != 2 or combined_ids.shape[1] < 2:
        return values
    canonical_items = dict(item_features)
    canonical_items["id"] = combined_ids[:, 1:2]
    user_features = {"id": combined_ids[:, 0:1]}
    return user_features, canonical_items, history_features, labels


class CanonicalDataLoader:
    """Expose a legacy dataset loader through the canonical `Batch` contract."""

    def __init__(
        self,
        dataset_name: str,
        source: Any,
        manifest: DatasetManifest,
        history_mode: HistoryMode = HistoryMode.SEQUENCE_TOKENS,
    ) -> None:
        if manifest.name != dataset_name.lower():
            raise ContractError("manifest name must match dataset_name")
        self.dataset_name = dataset_name.lower()
        self.source = source
        self.manifest = manifest
        self.history_mode = HistoryMode(history_mode)

    def iter_batches(self, split: str) -> Iterator[Batch]:
        canonical_split = _normalise_split(split)
        method_name = (
            "get_data_seq"
            if self.history_mode == HistoryMode.SEQUENCE_TOKENS
            else "get_data"
        )
        try:
            load_split = getattr(self.source, method_name)
        except AttributeError as error:
            raise ContractError(
                "dataset {!r} does not support history mode {!r}".format(
                    self.dataset_name, self.history_mode.value
                )
            ) from error
        for batch_index, legacy_batch in enumerate(load_split(canonical_split)):
            if isinstance(legacy_batch, Batch):
                batch = legacy_batch
            else:
                values = _split_legacy_ids(legacy_batch)
                batch = Batch.from_legacy(
                    values,
                    padding_id=self.manifest.padding_id,
                    metadata={
                        "dataset": self.dataset_name,
                        "dataset_version": self.manifest.version,
                        "dataset_fingerprint": self.manifest.fingerprint,
                        "split": canonical_split,
                        "batch_index": batch_index,
                    },
                )
            yield batch

    def get_multi_modal(self) -> Mapping[str, Any]:
        """Temporary feature-store bridge for quantization models."""

        try:
            values = self.source.get_multi_modal()
        except AttributeError as error:
            raise ContractError("dataset does not expose a multimodal feature store") from error
        if not isinstance(values, Mapping):
            raise ContractError("multimodal feature store must be a mapping")
        return values


def adapt_legacy_loader(
    dataset_name: str,
    source: Any,
    data_config: Mapping[str, Any],
    history_mode: Optional[HistoryMode] = None,
) -> CanonicalDataLoader:
    """Create a canonical adapter using manifest information available today."""

    manifest = DatasetManifest.from_config(dataset_name, data_config)
    mode = history_mode or HistoryMode.SEQUENCE_TOKENS
    return CanonicalDataLoader(dataset_name, source, manifest, mode)


__all__ = [
    "CanonicalDataLoader",
    "DataLoaderProtocol",
    "HistoryMode",
    "adapt_legacy_loader",
]

"""Canonical EB-NeRD loader retaining impression and missing-modality metadata."""

from typing import Any, Iterator, Mapping

import numpy as np
import torch

from mmctr.core import Batch, ContractError
from mmctr.data.datasets.arrays import NamedArrayDatasetLoader


class EBNeRDLoader(NamedArrayDatasetLoader):
    """Share the existing model interface and expose auditable exposure metadata."""

    dataset_name = "ebnerd"

    def __init__(self, data_config: Mapping[str, Any], batch_size: int) -> None:
        super().__init__(data_config, batch_size)
        if self.manifest.version not in (
            "ebnerd-small-canonical-v1",
            "ebnerd-small-canonical-v2",
            "ebnerd-small-canonical-v3",
            "ebnerd-small-canonical-v4",
        ):
            raise ContractError("unsupported EB-NeRD dataset version")
        for key, expected in (
            ("fingerprint", self.manifest.fingerprint),
            ("seq_len", self.manifest.sequence_length),
            ("id_feature_num", self.manifest.metadata["max_encoded_id"]),
        ):
            if key in data_config and data_config[key] != expected:
                raise ContractError("EB-NeRD config/manifest mismatch: {}".format(key))
        self._ids = self.manifest.metadata.get("id_features")
        self._structured = self.manifest.metadata.get("structured_features")
        if self._structured:
            if (
                not self._ids
                and data_config.get("structured_feature_dim") != self._structured["dimension"]
            ):
                raise ContractError("EB-NeRD structured feature dimension missing or mismatched")
            self._structured_items = np.load(self.root / "structured/articles.npy", mmap_mode="r")
            self._published = np.load(self.root / "structured/published.npy", mmap_mode="r")
        if self._ids:
            if data_config.get("extra_id_fields") != self._ids["extra_fields"]:
                raise ContractError("additional ID field count mismatch")
            self._article_ids = np.load(self.root / "categorical/articles.npy", mmap_mode="r")
        self._presence = {
            name: np.load(self.root / "item_features" / (name + "_present.npy"), mmap_mode="r")
            for name in self._modalities
        }
        for values in self._presence.values():
            if values.dtype != np.bool_ or values.shape != self._item_ids.shape or values[0]:
                raise ContractError("invalid EB-NeRD modality presence store")

    def iter_batches(self, split: str) -> Iterator[Batch]:
        name = self._split_name(split)
        root = self.root / "splits" / name
        audit = {
            field: np.load(root / (field + ".npy"), mmap_mode="r")
            for field in ("impression_ids", "timestamps", "candidate_positions")
        }
        count = self.manifest.splits[name].samples
        if any(values.shape != (count,) for values in audit.values()):
            raise ContractError("EB-NeRD audit arrays have inconsistent lengths")
        if self._structured:
            store = self.root / "structured"
            official = "validation" if name == "test" else "train"
            profiles = np.load(store / (official + "_profiles.npy"), mmap_mode="r")
            structured = {
                key: np.load(store / name / (key + ".npy"), mmap_mode="r")
                for key in ("context", "profile_index", "impression_index", "history_frequency")
            }
        if self._ids:
            categories = self.root / "categorical"
            profile_ids = np.load(categories / (official + "_profiles.npy"), mmap_mode="r")
            context_ids = np.load(categories / (name + "_context.npy"), mmap_mode="r")
        start = 0
        for batch in super().iter_batches(name):
            stop = start + batch.batch_size
            item_rows = self._feature_rows(batch.item_features["id"]).numpy().reshape(-1)
            history_rows = self._feature_rows(batch.history_features["id"]).numpy()
            metadata = dict(batch.metadata)
            metadata.update(
                {key: self._tensor(values[start:stop], torch.long) for key, values in audit.items()}
            )
            metadata["event_ids"] = tuple(
                "{}:{}".format(int(i), int(j))
                for i, j in zip(
                    audit["impression_ids"][start:stop], audit["candidate_positions"][start:stop]
                )
            )
            metadata["modality_presence"] = {
                modality: {
                    "item": self._tensor(values[item_rows], torch.bool),
                    "history": self._tensor(values[history_rows], torch.bool),
                }
                for modality, values in self._presence.items()
            }
            context = dict(batch.context_features)
            user = dict(batch.user_features)
            if self._structured:
                from .enrichment import assemble_features

                indices = structured["impression_index"][start:stop]
                values = assemble_features(
                    self._structured_items[item_rows],
                    profiles[structured["profile_index"][indices]],
                    structured["context"][indices],
                    self._published[item_rows],
                    audit["timestamps"][start:stop],
                    structured["history_frequency"][start:stop],
                )
                if values.shape != (batch.batch_size, self._structured["dimension"]):
                    raise ContractError("invalid assembled structured features")
                if self._ids:
                    from .id_features import numeric_ids, numeric_values, BAG_WIDTH

                    numeric = numeric_ids(numeric_values(values), self._ids["numeric_offsets"])
                    padded = np.zeros((len(numeric), numeric.shape[1], BAG_WIDTH), dtype=np.int64)
                    padded[:, :, 0] = numeric
                    ids = np.concatenate(
                        [
                            self._article_ids[item_rows],
                            profile_ids[structured["profile_index"][indices]],
                            context_ids[indices],
                            padded,
                        ],
                        axis=1,
                    )
                    user["attribute_ids"] = self._tensor(ids, torch.long)
                else:
                    context["structured"] = self._tensor(values, torch.float32)
            yield Batch(
                user_features=user,
                item_features=batch.item_features,
                history_features=batch.history_features,
                history_mask=batch.history_mask,
                labels=batch.labels,
                context_features=context,
                metadata=metadata,
            )
            start = stop
        if start != count:
            raise ContractError("EB-NeRD split count differs from manifest")


__all__ = ["EBNeRDLoader"]

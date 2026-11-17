"""Diagnostic data transforms and AUC comparisons."""

import math
from dataclasses import replace
from typing import Any, Dict, Iterator, Mapping, Optional

import numpy as np
import torch

from mmctr.core import Batch, ContractError


def compare_auc(test_auc: Optional[float], baseline_auc: float) -> Dict[str, Any]:
    """Positive signed deltas mean improvement; relative values are percentages."""
    if not math.isfinite(baseline_auc) or not 0.0 <= baseline_auc <= 1.0:
        raise ContractError("baseline AUC must be finite and in [0, 1]")
    if test_auc is None:
        return dict(
            test_auc=None,
            baseline_test_auc=baseline_auc,
            auc_delta=None,
            auc_absolute_difference=None,
            relative_change_percent=None,
            direction="not_evaluable",
        )
    if not math.isfinite(test_auc) or not 0.0 <= test_auc <= 1.0:
        raise ContractError("test AUC must be finite and in [0, 1]")
    delta = test_auc - baseline_auc
    return dict(
        test_auc=test_auc,
        baseline_test_auc=baseline_auc,
        auc_delta=delta,
        auc_absolute_difference=abs(delta),
        relative_change_percent=100.0 * delta / baseline_auc if baseline_auc else None,
        direction="increase" if delta > 0 else ("decrease" if delta < 0 else "equal"),
    )


def cold_selection(
    user_support: np.ndarray, item_support: np.ndarray, target: str, regime: str
) -> np.ndarray:
    """Count all train candidate rows; legacy low-frequency includes zero support."""
    if target not in {"union", "user", "item", "intersection"}:
        raise ContractError("unknown cold-start target")
    if regime not in {"zero", "low_lt5"}:
        raise ContractError("unknown cold-start regime")
    if user_support.shape != item_support.shape or user_support.ndim != 1:
        raise ContractError("support vectors must be aligned one-dimensional arrays")
    if np.any(user_support < 0) or np.any(item_support < 0):
        raise ContractError("support counts cannot be negative")
    user = user_support == 0 if regime == "zero" else user_support < 5
    item = item_support == 0 if regime == "zero" else item_support < 5
    return {"union": user | item, "user": user, "item": item, "intersection": user & item}[target]


def population_summary(labels: np.ndarray) -> Dict[str, Any]:
    """Empty and single-class populations have no valid ROC-AUC."""
    values = np.asarray(labels).reshape(-1)
    if not np.isin(values, [0, 1]).all():
        raise ContractError("labels must be binary")
    positives = int(values.sum())
    count = len(values)
    reason = "empty" if count == 0 else ("single_class" if positives in (0, count) else "ok")
    return dict(
        samples=count,
        positives=positives,
        negatives=count - positives,
        positive_rate=positives / count if count else None,
        auc_evaluable=reason == "ok",
        reason=reason,
    )


class ColdSubsetLoader:
    """Filter validation/test only; keep original training and entity histories."""

    def __init__(self, source: Any, selections: Mapping[str, np.ndarray]) -> None:
        if set(selections) != {"val", "test"}:
            raise ContractError("cold-start requires val and test masks; train stays unchanged")
        self.source = source
        self.selections = {name: np.asarray(mask, dtype=bool) for name, mask in selections.items()}
        if any(mask.ndim != 1 for mask in self.selections.values()):
            raise ContractError("cold-start masks must be one-dimensional")

    def iter_batches(self, split: str) -> Iterator[Batch]:
        split = "val" if split == "validation" else split
        selection = self.selections.get(split)
        offset = 0
        for batch in self.source.iter_batches(split):
            if selection is None:
                yield batch
                continue
            end = offset + batch.batch_size
            if end > len(selection):
                raise ContractError("cold-start mask is shorter than source split")
            mask = torch.as_tensor(selection[offset:end], device=batch.labels.device)
            offset = end
            if not bool(mask.any()):
                continue

            def select(values):
                return {name: value[mask] for name, value in values.items()}

            yield replace(
                batch,
                user_features=select(batch.user_features),
                item_features=select(batch.item_features),
                history_features=select(batch.history_features),
                context_features=select(batch.context_features),
                history_mask=batch.history_mask[mask],
                labels=batch.labels[mask],
                metadata={**dict(batch.metadata), "analysis_subset": True},
            )
        if selection is not None and offset != len(selection):
            raise ContractError("cold-start mask length differs from source split")


def reserve_oov_embeddings(model):
    """Append separate neutral user/item OOV rows only AFTER freezing best weights."""
    if hasattr(model, "_paper_oov_ids"):
        return model._paper_oov_ids
    embedding = model.embedding
    weights = embedding.embedding
    size, width = weights.shape
    embedding.embedding = torch.nn.Parameter(
        torch.cat([weights.detach(), weights.new_zeros(2, width)], dim=0),
        requires_grad=False,
    )
    model._paper_oov_ids = (size, size + 1)
    return model._paper_oov_ids


class PaperColdStartLoader:
    """Full train/val; predefined test subset with train-graph shared OOV IDs.

    Features have already been gathered under original IDs. Remapping changes only
    identity tensors, never context, modality vectors, labels or history validity.
    ``selection=None`` applies the same OOV policy to the full-test comparator.
    """

    def __init__(self, source, selection, seen, item_offset, oov_ids):
        self.source = source
        self.selection = None if selection is None else np.asarray(selection, dtype=bool)
        if self.selection is not None and self.selection.ndim != 1:
            raise ContractError("cold test mask must be one-dimensional")
        self.seen = {name: torch.as_tensor(value) for name, value in seen.items()}
        self.item_offset = item_offset
        self.oov_ids = oov_ids

    def _map(self, ids, field, oov, valid=None):
        index = ids - (self.item_offset if field == "item_ids" else 0)
        seen = self.seen[field].to(ids.device)
        known = torch.zeros_like(ids, dtype=torch.bool)
        in_range = (index >= 0) & (index < len(seen))
        known[in_range] = seen[index[in_range]]
        replace_mask = ~known
        if valid is not None:
            replace_mask &= valid
        return torch.where(replace_mask, torch.full_like(ids, oov), ids)

    def iter_batches(self, split):
        if split != "test":
            yield from self.source.iter_batches(split)
            return
        # Reuse the checked slicing boundary, but never filter validation.
        source = (
            self.source
            if self.selection is None
            else ColdSubsetLoader(
                self.source, {"val": np.array([], dtype=bool), "test": self.selection}
            )
        )
        for batch in source.iter_batches("test"):
            yield replace(
                batch,
                user_features={
                    **batch.user_features,
                    "id": self._map(batch.user_features["id"], "user_ids", self.oov_ids[0]),
                },
                item_features={
                    **batch.item_features,
                    "id": self._map(batch.item_features["id"], "item_ids", self.oov_ids[1]),
                },
                history_features={
                    **batch.history_features,
                    "id": self._map(
                        batch.history_features["id"],
                        "item_ids",
                        self.oov_ids[1],
                        batch.history_mask,
                    ),
                },
                metadata={**batch.metadata, "cold_protocol": "paper-mixed-graph-oov-v2"},
            )


class LegacyRobustnessLoader:
    """Original sample-wide non-ID masking: train/test masked, validation clean."""

    def __init__(self, source: Any, probability: float, mask_seed: int = 42) -> None:
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ContractError("drop probability must be in [0, 1]")
        self.source = source
        self.probability = probability
        self.mask_seed = mask_seed
        self.epoch = -1

    def iter_batches(self, split: str) -> Iterator[Batch]:
        split = "val" if split == "validation" else split
        if split == "train":
            self.epoch += 1
        for index, batch in enumerate(self.source.iter_batches(split)):
            if split == "val" or self.probability == 0:
                yield batch
                continue
            seed = self.mask_seed * 1_000_003 + index
            if split == "train":
                seed += self.epoch * 100_003
            rng = np.random.RandomState(seed)
            mask = torch.as_tensor(
                rng.random(batch.batch_size) < self.probability, device=batch.labels.device
            )

            def masked(values):
                result = {}
                for name, value in values.items():
                    if name == "id":
                        result[name] = value
                    else:
                        result[name] = value.clone()
                        result[name][mask] = 0
                return result

            yield replace(
                batch,
                user_features=masked(batch.user_features),
                item_features=masked(batch.item_features),
                history_features=masked(batch.history_features),
                metadata={**dict(batch.metadata), "robustness_mask": mask.detach().cpu().tolist()},
            )

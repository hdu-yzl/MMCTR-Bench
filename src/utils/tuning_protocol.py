"""Shared model-selection rules for legacy tuning scripts.

The test split is intentionally absent from this module. Hyperparameter search
must select configurations from validation metrics only; test evaluation is a
separate step after the configuration has been frozen.
"""

from dataclasses import dataclass
from typing import Any, Optional


SELECTION_SPLIT = "val"


@dataclass(frozen=True)
class SelectionMetrics:
    """Validation metrics used to compare tuning trials."""

    auc: float
    loss: float


def evaluate_for_selection(model: Any, data_loader: Any) -> SelectionMetrics:
    """Evaluate a trained trial on the only split allowed for selection."""

    auc, loss = model.evalate(data_loader, SELECTION_SPLIT)
    return SelectionMetrics(auc=float(auc), loss=float(loss))


def is_better(
    candidate: SelectionMetrics,
    incumbent: Optional[SelectionMetrics],
) -> bool:
    """Preserve the legacy comparison rule: a strictly larger AUC wins."""

    return incumbent is None or candidate.auc > incumbent.auc

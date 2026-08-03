"""Validated binary CTR metrics."""

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn import metrics  # type: ignore[import-untyped]

from mmctr.core import ContractError


@dataclass(frozen=True)
class BinaryMetrics:
    """ROC-AUC and LogLoss computed from one complete split."""

    auc: float
    log_loss: float
    samples: int

    def to_dict(self, prefix: str = "") -> Dict[str, float]:
        return {
            "{}auc".format(prefix): float(self.auc),
            "{}log_loss".format(prefix): float(self.log_loss),
            "{}samples".format(prefix): float(self.samples),
        }


def binary_classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> BinaryMetrics:
    """Compute CTR metrics after strict shape, class, and finiteness checks."""

    labels = np.asarray(labels).reshape(-1)
    probabilities = np.asarray(probabilities).reshape(-1)
    if labels.shape != probabilities.shape:
        raise ContractError("labels and probabilities must have identical shape")
    if labels.size == 0:
        raise ContractError("cannot evaluate an empty split")
    if not np.isfinite(labels).all() or not np.isfinite(probabilities).all():
        raise ContractError("metric inputs must be finite")
    unique_labels = np.unique(labels)
    if not np.array_equal(unique_labels, np.array([0.0, 1.0])):
        raise ContractError("CTR evaluation requires both binary label classes")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ContractError("probabilities must be within [0, 1]")
    epsilon: float = float(np.finfo(np.float64).eps)
    clipped = np.clip(probabilities.astype(np.float64), epsilon, 1.0 - epsilon)
    return BinaryMetrics(
        auc=float(metrics.roc_auc_score(labels, clipped)),
        log_loss=float(metrics.log_loss(labels, clipped, labels=[0.0, 1.0])),
        samples=int(labels.size),
    )


__all__ = ["BinaryMetrics", "binary_classification_metrics"]

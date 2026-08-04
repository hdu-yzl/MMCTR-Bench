"""Model evaluation over canonical batches."""

from typing import Iterable, List

import numpy as np
import torch

from mmctr.core import Batch, ensure_model_output

from .metrics import BinaryMetrics, binary_classification_metrics


class BinaryClassificationEvaluator:
    """Collect a complete split and compute validated CTR metrics."""

    def __init__(self, device: torch.device) -> None:
        self.device = device

    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module, batches: Iterable[Batch]) -> BinaryMetrics:
        """Collect sigmoid probabilities for one full split under ``no_grad``."""

        model.eval()
        probabilities: List[np.ndarray] = []
        labels: List[np.ndarray] = []
        for batch in batches:
            device_batch = batch.to(self.device)
            output = ensure_model_output(model(device_batch))
            probabilities.append(output.logits.sigmoid().detach().cpu().numpy())
            labels.append(device_batch.labels.detach().cpu().numpy())
        if not probabilities:
            return binary_classification_metrics(np.array([]), np.array([]))
        return binary_classification_metrics(
            np.concatenate(labels).astype(np.float64),
            np.concatenate(probabilities).astype(np.float64),
        )


__all__ = ["BinaryClassificationEvaluator"]

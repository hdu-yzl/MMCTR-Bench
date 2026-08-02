"""Explicit compatibility boundary for legacy model forward signatures."""

from typing import Dict

import torch

from mmctr.core import Batch, ContractError, ModelOutput

from .base import BaseSeqModel, HistoryCapability


def _copy_features(values) -> Dict[str, torch.Tensor]:
    return {name: tensor for name, tensor in values.items()}


class LegacyModelAdapter(BaseSeqModel):
    """Wrap a legacy model without exposing tuple signatures to new training code."""

    def __init__(
        self,
        model: torch.nn.Module,
        history_capability: HistoryCapability,
        forward_uses_labels: bool = False,
    ) -> None:
        super().__init__(history_capability)
        self.model = model
        self.forward_uses_labels = bool(forward_uses_labels)

    def forward_batch(self, batch: Batch) -> ModelOutput:
        user_features = _copy_features(batch.user_features)
        item_features = _copy_features(batch.item_features)
        history_features = _copy_features(batch.history_features)
        collisions = set(item_features).intersection(batch.context_features)
        if collisions:
            raise ContractError(
                "legacy target/context feature names collide: {}".format(sorted(collisions))
            )
        item_features.update(_copy_features(batch.context_features))

        if self.history_capability == HistoryCapability.POOLED_HISTORY:
            user_ids = user_features.get("id")
            item_ids = item_features.get("id")
            if user_ids is not None and item_ids is not None:
                item_features["id"] = torch.cat([user_ids, item_ids], dim=1)
            if self.forward_uses_labels:
                legacy_output = self.model(item_features, history_features, batch.labels)
            else:
                legacy_output = self.model(item_features, history_features)
        else:
            if self.forward_uses_labels:
                legacy_output = self.model(
                    user_features, item_features, history_features, batch.labels
                )
            else:
                legacy_output = self.model(user_features, item_features, history_features)
        return ModelOutput.from_legacy(legacy_output)


__all__ = ["LegacyModelAdapter"]

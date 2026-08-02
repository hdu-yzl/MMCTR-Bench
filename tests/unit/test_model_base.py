import unittest

import torch

from mmctr.core import Batch, ModelOutput
from mmctr.models import BaseSeqModel, HistoryCapability, LegacyModelAdapter


def make_batch():
    return Batch(
        user_features={"id": torch.tensor([[1], [2]])},
        item_features={"id": torch.tensor([[3], [4]])},
        history_features={"id": torch.tensor([[5, 0], [6, 7]])},
        history_mask=torch.tensor([[True, False], [True, True]]),
        labels=torch.tensor([0.0, 1.0]),
    )


class PooledModel(BaseSeqModel):
    def __init__(self):
        super().__init__(HistoryCapability.POOLED_HISTORY)

    def forward_batch(self, batch):
        sequence = batch.history_features["id"].float().unsqueeze(-1)
        return ModelOutput(self.masked_pool(sequence, batch.history_mask).squeeze(-1))


class MutatingLegacyModel(torch.nn.Module):
    def forward(self, features, history):
        features["id"] = torch.zeros_like(features["id"])
        history["id"] = torch.zeros_like(history["id"])
        return {"pred": torch.ones(features["id"].shape[0], 1)}


class ContextLegacyModel(torch.nn.Module):
    def forward(self, features, history):
        return {"pred": features["query_text"].sum(dim=-1, keepdim=True)}


class ModelBaseTests(unittest.TestCase):
    def test_masked_pool_handles_padding_and_all_padding(self):
        output = PooledModel()(make_batch())
        self.assertTrue(torch.equal(output.logits, torch.tensor([5.0, 6.5])))

    def test_legacy_adapter_does_not_mutate_batch_mappings(self):
        batch = make_batch()
        original_items = batch.item_features["id"].clone()
        original_history = batch.history_features["id"].clone()
        adapter = LegacyModelAdapter(
            MutatingLegacyModel(), HistoryCapability.POOLED_HISTORY
        )
        output = adapter(batch)
        self.assertEqual((2,), tuple(output.logits.shape))
        self.assertTrue(torch.equal(batch.item_features["id"], original_items))
        self.assertTrue(torch.equal(batch.history_features["id"], original_history))

    def test_legacy_adapter_exposes_context_at_compatibility_boundary(self):
        source = make_batch()
        batch = Batch(
            user_features=source.user_features,
            item_features=source.item_features,
            history_features=source.history_features,
            history_mask=source.history_mask,
            labels=source.labels,
            context_features={"query_text": torch.ones(2, 3)},
        )
        adapter = LegacyModelAdapter(
            ContextLegacyModel(), HistoryCapability.POOLED_HISTORY
        )
        self.assertTrue(torch.equal(adapter(batch).logits, torch.tensor([3.0, 3.0])))


if __name__ == "__main__":
    unittest.main()

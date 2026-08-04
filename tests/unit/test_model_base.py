import unittest

import torch

from mmctr.core import Batch, ModelOutput
from mmctr.models import BaseSeqModel, HistoryCapability


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


class ModelBaseTests(unittest.TestCase):
    def test_masked_pool_handles_padding_and_all_padding(self):
        output = PooledModel()(make_batch())
        self.assertTrue(torch.equal(output.logits, torch.tensor([5.0, 6.5])))


if __name__ == "__main__":
    unittest.main()

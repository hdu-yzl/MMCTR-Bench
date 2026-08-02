import unittest

import torch

from mmctr.core import Batch, ContractError, ModelOutput, RunResult


class BatchContractTests(unittest.TestCase):
    def make_batch(self):
        return Batch(
            user_features={"age": torch.ones(2, 1)},
            item_features={"id": torch.tensor([[1], [2]], dtype=torch.long)},
            history_features={"id": torch.tensor([[1, 0, 0], [2, 1, 0]])},
            history_mask=torch.tensor([[True, False, False], [True, True, False]]),
            labels=torch.tensor([[0.0], [1.0]]),
            metadata={"split": "train"},
        )

    def test_normalises_labels_and_preserves_shapes(self):
        batch = self.make_batch()
        self.assertEqual((2,), tuple(batch.labels.shape))
        self.assertEqual(2, batch.batch_size)
        self.assertEqual(3, batch.sequence_length)

    def test_context_features_are_validated_and_moved(self):
        batch = self.make_batch()
        contextual = Batch(
            user_features=batch.user_features,
            item_features=batch.item_features,
            history_features=batch.history_features,
            history_mask=batch.history_mask,
            labels=batch.labels,
            context_features={"query_text": torch.ones(2, 3)},
        )
        moved = contextual.to("cpu")
        self.assertEqual((2, 3), tuple(moved.context_features["query_text"].shape))

    def test_legacy_adapter_derives_mask_from_padding_id(self):
        legacy = (
            {"id": torch.tensor([[1], [2]])},
            {"id": torch.tensor([[1, 0], [2, 1]])},
            torch.tensor([[0.0], [1.0]]),
        )
        batch = Batch.from_legacy(legacy, padding_id=0)
        self.assertTrue(
            torch.equal(batch.history_mask, torch.tensor([[True, False], [True, True]]))
        )

    def test_rejects_wrong_mask_dtype(self):
        with self.assertRaisesRegex(ContractError, "torch.bool"):
            Batch(
                user_features={},
                item_features={"id": torch.ones(2, 1, dtype=torch.long)},
                history_features={"id": torch.ones(2, 3, dtype=torch.long)},
                history_mask=torch.ones(2, 3),
                labels=torch.ones(2),
            )


class ModelOutputContractTests(unittest.TestCase):
    def test_adapts_legacy_output_and_sums_auxiliary_loss(self):
        logits = torch.tensor([[0.1], [0.2]])
        output = ModelOutput.from_legacy({"pred": logits, "au_loss": torch.tensor(0.5)})
        self.assertEqual((2,), tuple(output.logits.shape))
        self.assertAlmostEqual(0.5, output.auxiliary_loss().item())

    def test_rejects_non_scalar_auxiliary_loss(self):
        with self.assertRaisesRegex(ContractError, "must be scalar"):
            ModelOutput(torch.ones(2), {"alignment": torch.ones(2)})


class RunResultContractTests(unittest.TestCase):
    def test_serialises_completed_result(self):
        result = RunResult("run-1", "completed", {"auc": 0.75})
        self.assertTrue(result.succeeded)
        self.assertEqual(0.75, result.to_dict()["metrics"]["auc"])

    def test_rejects_non_finite_metric(self):
        with self.assertRaisesRegex(ContractError, "finite"):
            RunResult("run-1", "failed", {"loss": float("nan")})


if __name__ == "__main__":
    unittest.main()

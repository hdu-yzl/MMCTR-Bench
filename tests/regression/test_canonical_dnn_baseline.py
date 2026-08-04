import json
import unittest
from pathlib import Path

import torch

from mmctr.core import Batch
from mmctr.models.registry import create_model


BASELINE_PATH = Path(__file__).parents[1] / "baselines" / "legacy_dnn_cpu_v1.json"


class CanonicalDnnRegressionTest(unittest.TestCase):
    def test_migration_preserves_frozen_logits_loss_and_parameter_count(self) -> None:
        """Recreate legacy weights from the seed; this is not a paper-metric baseline."""
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        torch.manual_seed(int(baseline["train_config"]["seed"]))
        model = create_model("dnn", baseline["model_config"], baseline["data_config"])
        model.eval()

        target_ids = torch.tensor(baseline["inputs"]["features"]["id"], dtype=torch.long)
        history_ids = torch.tensor(baseline["inputs"]["history_features"]["id"], dtype=torch.long)
        labels = torch.tensor(baseline["inputs"]["labels"], dtype=torch.float32).reshape(-1)
        batch = Batch(
            user_features={"id": target_ids[:, :1]},
            item_features={"id": target_ids[:, 1:]},
            history_features={"id": history_ids},
            history_mask=torch.ones_like(history_ids, dtype=torch.bool),
            labels=labels,
        )

        with torch.no_grad():
            logits = model(batch).logits
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)

        expected = baseline["expected"]
        tolerance = baseline["tolerance"]
        expected_logits = torch.tensor(expected["logits"], dtype=logits.dtype)
        self.assertTrue(
            torch.allclose(
                logits,
                expected_logits,
                atol=tolerance["absolute"],
                rtol=tolerance["relative"],
            )
        )
        self.assertAlmostEqual(loss.item(), expected["loss"], places=6)
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            expected["parameter_count"],
        )


if __name__ == "__main__":
    unittest.main()

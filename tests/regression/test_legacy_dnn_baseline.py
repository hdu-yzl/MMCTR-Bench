import json
import logging
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import torch

from mmctr.utils import helper


BASELINE_PATH = Path(__file__).parents[1] / "baselines" / "legacy_dnn_cpu_v1.json"


class LegacyDnnRegressionTest(unittest.TestCase):
    def test_logits_loss_and_parameter_count_match_frozen_fixture(self) -> None:
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        logger = logging.getLogger("tests.regression.legacy_dnn")
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())

        model_config = deepcopy(baseline["model_config"])
        train_config = deepcopy(baseline["train_config"])
        data_config = deepcopy(baseline["data_config"])

        with tempfile.TemporaryDirectory() as temp_dir:
            train_config["checkpoint_dir"] = temp_dir
            model = helper.getModel(
                baseline["model_name"],
                model_config,
                train_config,
                data_config,
                logger,
            )
            model.eval()

            features = {"id": torch.tensor(baseline["inputs"]["features"]["id"], dtype=torch.long)}
            history_features = {
                "id": torch.tensor(
                    baseline["inputs"]["history_features"]["id"],
                    dtype=torch.long,
                )
            }
            labels = torch.tensor(baseline["inputs"]["labels"], dtype=torch.float32)

            with torch.no_grad():
                logits = model(features, history_features)["pred"]
                loss = model.compute_loss(logits, labels)

        expected = baseline["expected"]
        tolerance = baseline["tolerance"]
        expected_logits = torch.tensor(expected["logits"], dtype=logits.dtype).reshape_as(logits)
        self.assertTrue(
            torch.allclose(
                logits,
                expected_logits,
                atol=tolerance["absolute"],
                rtol=tolerance["relative"],
            )
        )
        self.assertAlmostEqual(loss.item(), expected["loss"], places=6)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 205)


if __name__ == "__main__":
    unittest.main()

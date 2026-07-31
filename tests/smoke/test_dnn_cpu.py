import logging
import tempfile
import unittest
from pathlib import Path

import torch

from tests.fixtures.synthetic import legacy_dnn_configs, make_legacy_dnn_batch
from utils import helper


class LegacyDnnCpuSmokeTest(unittest.TestCase):
    def test_forward_loss_and_backward_on_synthetic_batch(self) -> None:
        logger = logging.getLogger("tests.smoke.legacy_dnn")
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())

        with tempfile.TemporaryDirectory() as temp_dir:
            model_config, train_config, data_config = legacy_dnn_configs(Path(temp_dir))
            model = helper.getModel("dnn", model_config, train_config, data_config, logger)
            self.assertEqual(model.device.type, "cpu")

            features, history_features, labels = make_legacy_dnn_batch()
            output = model(features, history_features)

            self.assertEqual(set(output), {"pred"})
            self.assertEqual(output["pred"].shape, labels.shape)
            self.assertTrue(torch.isfinite(output["pred"]).all())

            loss = model.compute_loss(output["pred"], labels)
            self.assertEqual(loss.ndim, 0)
            self.assertTrue(torch.isfinite(loss))

            model.optim.zero_grad()
            loss.backward()
            gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
            model.optim.step()


if __name__ == "__main__":
    unittest.main()

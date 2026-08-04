import unittest

import torch

from mmctr.core import Batch
from mmctr.models.common.registry import create_model


class CanonicalDnnCpuSmokeTest(unittest.TestCase):
    def test_forward_loss_backward_and_optimizer_step(self) -> None:
        torch.manual_seed(2025)
        model = create_model(
            "dnn",
            {
                "latent_dim": 4,
                "projection_dim": 4,
                "mlp_dims": [8],
                "dropout": 0.0,
                "batch_norm": False,
            },
            {"id_feature_num": 16},
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        batch = Batch(
            user_features={"id": torch.tensor([[1], [3], [5], [7]])},
            item_features={"id": torch.tensor([[2], [4], [6], [8]])},
            history_features={"id": torch.tensor([[2, 3, 4], [4, 3, 2], [6, 5, 4], [8, 7, 6]])},
            history_mask=torch.ones((4, 3), dtype=torch.bool),
            labels=torch.tensor([0.0, 1.0, 0.0, 1.0]),
        )

        output = model(batch)
        self.assertEqual((4,), tuple(output.logits.shape))
        self.assertTrue(torch.isfinite(output.logits).all())

        loss = torch.nn.functional.binary_cross_entropy_with_logits(output.logits, batch.labels)
        optimizer.zero_grad()
        loss.backward()
        gradients = [
            parameter.grad for parameter in model.parameters() if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        optimizer.step()


if __name__ == "__main__":
    unittest.main()

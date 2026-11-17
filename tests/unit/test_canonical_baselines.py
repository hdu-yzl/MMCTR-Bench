import unittest

import torch

from mmctr.core import Batch
from mmctr.models.common.registry import create_model


def make_batch():
    return Batch(
        user_features={"id": torch.tensor([[1], [2], [3], [4]])},
        item_features={"id": torch.tensor([[5], [6], [7], [8]])},
        history_features={"id": torch.tensor([[5, 0, 0], [6, 5, 0], [7, 6, 5], [8, 7, 6]])},
        history_mask=torch.tensor(
            [
                [True, False, False],
                [True, True, False],
                [True, True, True],
                [True, True, True],
            ]
        ),
        labels=torch.tensor([0.0, 1.0, 0.0, 1.0]),
    )


class CanonicalBaselineTests(unittest.TestCase):
    def configs(self, name):
        model = {
            "latent_dim": 4,
            "projection_dim": 4,
            "mlp_dims": [8],
            "dropout": 0.0,
            "batch_norm": False,
        }
        if name == "autoint":
            model.update({"attn_layers": 1, "attn_heads": 2, "attn_size": 2})
        if name == "din":
            model["attention_mlp_dims"] = [4]
        return model, {"id_feature_num": 32}

    def test_all_migrated_baselines_forward_and_backward(self):
        from mmctr.models import DNN

        for name in ("dnn", "dcn", "deepfm", "autoint", "din"):
            with self.subTest(name=name):
                model_config, data_config = self.configs(name)
                model = create_model(name, model_config, data_config)
                if name == "dnn":
                    self.assertIs(type(model), DNN)
                output = model(make_batch())
                self.assertEqual((4,), tuple(output.logits.shape))
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    output.logits, make_batch().labels
                )
                loss.backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_batch_size_one_keeps_logit_rank(self):
        model_config, data_config = self.configs("dnn")
        model = create_model("dnn", model_config, data_config)
        batch = Batch(
            user_features={"id": torch.tensor([[1]])},
            item_features={"id": torch.tensor([[2]])},
            history_features={"id": torch.tensor([[2, 0]])},
            history_mask=torch.tensor([[True, False]]),
            labels=torch.tensor([1.0]),
        )
        self.assertEqual((1,), tuple(model(batch).logits.shape))


if __name__ == "__main__":
    unittest.main()

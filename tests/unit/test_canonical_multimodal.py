import unittest

import torch

from mmctr.core import Batch
from mmctr.models.common.registry import create_model


def make_batch():
    return Batch(
        user_features={"id": torch.tensor([[1], [2]])},
        item_features={
            "id": torch.tensor([[3], [4]]),
            "text": torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]),
            "image": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        },
        history_features={
            "id": torch.tensor([[3, 0], [4, 3]]),
            "text": torch.tensor(
                [
                    [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]],
                    [[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]],
                ]
            ),
            "image": torch.tensor([[[1.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [1.0, 0.0]]]),
        },
        history_mask=torch.tensor([[True, False], [True, True]]),
        labels=torch.tensor([1.0, 0.0]),
    )


class CanonicalMultimodalTests(unittest.TestCase):
    def configs(self):
        model_config = {
            "latent_dim": 4,
            "projection_dim": 4,
            "mlp_dims": [8],
            "dropout": 0.0,
            "batch_norm": False,
            "rank": 2,
            "fusion_dim": 4,
        }
        data_config = {
            "id_feature_num": 32,
            "use_mm_features": ["id", "text", "image"],
            "use_mm_seq_features": ["id", "text", "image"],
            "mm_dims": {"text": 3, "image": 2},
            "mm_seq_dims": {"text": 3, "image": 2},
            "user_features": ["id"],
            "user_features_dim": {},
        }
        return model_config, data_config

    def test_simple_multimodal_models_forward_and_backward(self):
        for name in ("dnn_mm", "dnn_mm_seq", "lmf", "mtfn", "simcen"):
            with self.subTest(name=name):
                model_config, data_config = self.configs()
                if name == "dnn_mm":
                    model_config["modal_fusion_method"] = "cat"
                if name == "dnn_mm_seq":
                    model_config["modal_fusion_method"] = "add"
                if name == "simcen":
                    model_config.update(
                        {
                            "hidden_unit": [12, 12],
                            "ego_batch_norm": False,
                            "v1_batch_norm": False,
                            "v2_batch_norm": False,
                        }
                    )
                model = create_model(name, model_config, data_config)
                output = model(make_batch())
                self.assertEqual((2,), tuple(output.logits.shape))
                if name == "simcen":
                    self.assertIn("simcen_contrastive", output.auxiliary_losses)
                output.logits.sum().backward()
                self.assertTrue(any(value.grad is not None for value in model.parameters()))


if __name__ == "__main__":
    unittest.main()

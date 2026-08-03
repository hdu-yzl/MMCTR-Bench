import unittest

import torch

from mmctr.core import Batch, ContractError
from mmctr.models.base import BaseSeqModel, HistoryCapability
from mmctr.models.registry import create_model, model_spec


def make_batch(all_padding=False, batch_size=2):
    user_ids = torch.arange(1, batch_size + 1, dtype=torch.long).unsqueeze(1)
    item_ids = torch.arange(5, 5 + batch_size, dtype=torch.long).unsqueeze(1)
    history_ids = torch.tensor([[5, 0], [6, 5]], dtype=torch.long)[:batch_size]
    history_mask = history_ids.ne(0)
    if all_padding:
        history_ids = torch.zeros_like(history_ids)
        history_mask = torch.zeros_like(history_mask)
    text = torch.arange(batch_size * 3, dtype=torch.float32).view(batch_size, 3)
    image = torch.arange(batch_size * 2, dtype=torch.float32).view(batch_size, 2)
    history_text = torch.stack([text, torch.zeros_like(text)], dim=1)
    history_image = torch.stack([image, torch.zeros_like(image)], dim=1)
    if all_padding:
        history_text = torch.zeros_like(history_text)
        history_image = torch.zeros_like(history_image)
    return Batch(
        user_features={"id": user_ids},
        item_features={"id": item_ids, "text": text, "image": image},
        history_features={
            "id": history_ids,
            "text": history_text,
            "image": history_image,
        },
        history_mask=history_mask,
        labels=torch.arange(batch_size, dtype=torch.float32).remainder(2),
    )


def configs():
    model_config = {
        "latent_dim": 8,
        "projection_dim": 8,
        "mlp_dims": [16],
        "dropout": 0.0,
        "batch_norm": False,
        "query_num": 2,
        "cic_tau": 0.2,
        "cic_weight": 0.1,
        "fq_heads": 2,
        "fq_layer_num": 2,
        "lambda1": 0.01,
        "lambda2": 0.02,
        "T": 2,
        "num_cross_layers": 2,
        "heads": 2,
    }
    data_config = {
        "id_feature_num": 64,
        "use_mm_features": ["id", "text", "image"],
        "mm_seq_dims": {"text": 3, "image": 2},
        "user_features": ["id"],
        "user_features_dim": {},
    }
    return model_config, data_config


class CanonicalAdvancedSequenceTests(unittest.TestCase):
    def test_em3_and_diff_msin_use_canonical_contract_and_backward(self):
        for name in ("em3", "diff_msin"):
            with self.subTest(name=name):
                model_config, data_config = configs()
                model = create_model(name, model_config, data_config)
                self.assertIsInstance(model, BaseSeqModel)
                self.assertEqual(
                    HistoryCapability.SEQUENCE_TOKENS, model.history_capability
                )
                output = model(make_batch())
                self.assertEqual((2,), tuple(output.logits.shape))
                objective = output.logits.sum() + output.auxiliary_loss()
                objective.backward()
                self.assertTrue(
                    any(parameter.grad is not None for parameter in model.parameters())
                )

    def test_auxiliary_losses_are_named_finite_scalars(self):
        expected = {
            "em3": {"em3_content_item_contrastive"},
            "diff_msin": {"diff_msin_synthesis", "diff_msin_contrastive"},
        }
        for name, expected_names in expected.items():
            with self.subTest(name=name):
                model_config, data_config = configs()
                output = create_model(name, model_config, data_config)(make_batch())
                self.assertEqual(expected_names, set(output.auxiliary_losses))
                for loss in output.auxiliary_losses.values():
                    self.assertEqual(0, loss.ndim)
                    self.assertTrue(torch.isfinite(loss))

    def test_all_padding_history_is_finite(self):
        for name in ("em3", "diff_msin"):
            with self.subTest(name=name):
                model_config, data_config = configs()
                output = create_model(name, model_config, data_config)(
                    make_batch(all_padding=True)
                )
                self.assertTrue(torch.isfinite(output.logits).all())
                self.assertTrue(torch.isfinite(output.auxiliary_loss()))
                if name == "em3":
                    interest = output.representations["history_interest"]
                    self.assertTrue(torch.equal(interest, torch.zeros_like(interest)))

    def test_batch_size_one_keeps_logit_and_loss_rank(self):
        for name in ("em3", "diff_msin"):
            with self.subTest(name=name):
                model_config, data_config = configs()
                output = create_model(name, model_config, data_config)(
                    make_batch(batch_size=1)
                )
                self.assertEqual((1,), tuple(output.logits.shape))
                self.assertEqual(0, output.auxiliary_loss().ndim)

    def test_context_fallback_does_not_mutate_batch(self):
        batch = make_batch()
        original_ids = batch.item_features["id"].clone()
        contextual = Batch(
            user_features=batch.user_features,
            item_features={
                "id": batch.item_features["id"],
                "text": batch.item_features["text"],
            },
            history_features=batch.history_features,
            history_mask=batch.history_mask,
            labels=batch.labels,
            context_features={"image": batch.item_features["image"]},
        )
        for name in ("em3", "diff_msin"):
            with self.subTest(name=name):
                model_config, data_config = configs()
                output = create_model(name, model_config, data_config)(contextual)
                self.assertEqual((2,), tuple(output.logits.shape))
        self.assertTrue(torch.equal(contextual.item_features["id"], original_ids))

    def test_invalid_advanced_configuration_is_rejected(self):
        model_config, data_config = configs()
        model_config["cic_tau"] = 0.0
        with self.assertRaises(ContractError):
            create_model("em3", model_config, data_config)
        model_config, data_config = configs()
        model_config["lambda1"] = -0.1
        with self.assertRaises(ContractError):
            create_model("diff_msin", model_config, data_config)
        model_config, data_config = configs()
        model_config["T"] = 0
        with self.assertRaises(ContractError):
            create_model("diff_msin", model_config, data_config)
        model_config, data_config = configs()
        model_config["num_cross_layers"] = 0
        with self.assertRaises(ContractError):
            create_model("diff_msin", model_config, data_config)

    def test_registry_keeps_legacy_regression_metadata(self):
        for name, symbol in (("em3", "EM3"), ("diff_msin", "Diff_MSIN")):
            specification = model_spec(name)
            self.assertEqual("mmctr.models.advanced_sequence", specification.module)
            self.assertEqual(
                "models.mm_ctr_models", specification.metadata["legacy_module"]
            )
            self.assertEqual(symbol, specification.metadata["legacy_symbol"])


if __name__ == "__main__":
    unittest.main()

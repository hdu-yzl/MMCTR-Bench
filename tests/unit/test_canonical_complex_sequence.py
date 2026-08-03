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
        "latent_dim": 4,
        "projection_dim": 4,
        "mlp_dims": [8],
        "dropout": 0.0,
        "batch_norm": False,
        "modal_fusion_method": "cat",
        "tier_num": 4,
        "attention_dim": 6,
        "num_buckets": 5,
        "alpha": 0.25,
        "lambda0": 0.05,
    }
    data_config = {
        "id_feature_num": 64,
        "use_mm_features": ["id", "text", "image"],
        "mm_seq_dims": {"text": 3, "image": 2},
        "user_features": ["id"],
        "user_features_dim": {},
    }
    return model_config, data_config


class CanonicalComplexSequenceTests(unittest.TestCase):
    def test_dmf_and_marn_use_canonical_contract_and_backward(self):
        for name in ("dmf", "marn"):
            with self.subTest(name=name):
                model_config, data_config = configs()
                model = create_model(name, model_config, data_config)
                self.assertIsInstance(model, BaseSeqModel)
                self.assertEqual(HistoryCapability.SEQUENCE_TOKENS, model.history_capability)
                output = model(make_batch())
                self.assertEqual((2,), tuple(output.logits.shape))
                objective = output.logits.sum() + output.auxiliary_loss()
                objective.backward()
                self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_marn_exposes_weighted_scalar_auxiliary_losses(self):
        model_config, data_config = configs()
        output = create_model("marn", model_config, data_config)(make_batch())
        self.assertEqual(
            {
                "marn_domain_classifier",
                "marn_adversarial_invariance",
                "marn_specific_classifier",
            },
            set(output.auxiliary_losses),
        )
        for loss in output.auxiliary_losses.values():
            self.assertEqual(0, loss.ndim)
            self.assertTrue(torch.isfinite(loss))

    def test_all_padding_history_is_finite_and_zeroed(self):
        batch = make_batch(all_padding=True)
        for name, representation in (
            ("dmf", "modality_enhanced"),
            ("marn", "history_interest"),
        ):
            with self.subTest(name=name):
                model_config, data_config = configs()
                output = create_model(name, model_config, data_config)(batch)
                self.assertTrue(torch.isfinite(output.logits).all())
                values = output.representations[representation]
                self.assertTrue(torch.equal(values, torch.zeros_like(values)))
        model_config, data_config = configs()
        dmf_output = create_model("dmf", model_config, data_config)(batch)
        tiers = dmf_output.representations["similarity_tiers"]
        self.assertTrue(torch.equal(tiers, torch.zeros_like(tiers)))

    def test_batch_size_one_keeps_logit_rank(self):
        for name in ("dmf", "marn"):
            with self.subTest(name=name):
                model_config, data_config = configs()
                output = create_model(name, model_config, data_config)(make_batch(batch_size=1))
                self.assertEqual((1,), tuple(output.logits.shape))

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
        for name in ("dmf", "marn"):
            with self.subTest(name=name):
                model_config, data_config = configs()
                output = create_model(name, model_config, data_config)(contextual)
                self.assertEqual((2,), tuple(output.logits.shape))
        self.assertTrue(torch.equal(contextual.item_features["id"], original_ids))

    def test_invalid_model_specific_configuration_is_rejected(self):
        model_config, data_config = configs()
        model_config["alpha"] = 1.5
        with self.assertRaises(ContractError):
            create_model("dmf", model_config, data_config)
        model_config, data_config = configs()
        model_config["lambda0"] = -0.1
        with self.assertRaises(ContractError):
            create_model("marn", model_config, data_config)

    def test_marn_construction_preserves_global_anomaly_mode(self):
        anomaly_mode = torch.is_anomaly_enabled()
        model_config, data_config = configs()
        create_model("marn", model_config, data_config)
        self.assertEqual(anomaly_mode, torch.is_anomaly_enabled())

    def test_registry_keeps_legacy_regression_metadata(self):
        for name in ("dmf", "marn"):
            specification = model_spec(name)
            self.assertEqual("mmctr.models.sequence", specification.module)
            self.assertEqual("models.mm_ctr_models", specification.metadata["legacy_module"])
            self.assertEqual(name.upper(), specification.metadata["legacy_symbol"])


if __name__ == "__main__":
    unittest.main()

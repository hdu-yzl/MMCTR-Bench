import unittest

import torch

from mmctr.core import Batch
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
        "tier_num": 4,
    }
    data_config = {
        "id_feature_num": 64,
        "use_mm_features": ["id", "text", "image"],
        "mm_seq_dims": {"text": 3, "image": 2},
        "user_features": ["id"],
        "user_features_dim": {},
    }
    return model_config, data_config


class CanonicalSequenceTests(unittest.TestCase):
    def test_naml_and_make_use_canonical_sequence_contract(self):
        for name in ("naml", "make"):
            with self.subTest(name=name):
                model_config, data_config = configs()
                model = create_model(name, model_config, data_config)
                self.assertIsInstance(model, BaseSeqModel)
                self.assertEqual(HistoryCapability.SEQUENCE_TOKENS, model.history_capability)
                output = model(make_batch())
                self.assertEqual((2,), tuple(output.logits.shape))
                output.logits.sum().backward()
                self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_all_padding_history_is_finite_and_masked(self):
        for name in ("naml", "make"):
            with self.subTest(name=name):
                model_config, data_config = configs()
                model = create_model(name, model_config, data_config)
                output = model(make_batch(all_padding=True))
                self.assertTrue(torch.isfinite(output.logits).all())
                if name == "make":
                    tiers = output.representations["similarity_tiers"]
                    self.assertTrue(torch.equal(tiers, torch.zeros_like(tiers)))

    def test_batch_size_one_keeps_logit_rank(self):
        for name in ("naml", "make"):
            with self.subTest(name=name):
                model_config, data_config = configs()
                model = create_model(name, model_config, data_config)
                self.assertEqual((1,), tuple(model(make_batch(batch_size=1)).logits.shape))

    def test_target_context_fallback_does_not_mutate_batch(self):
        batch = make_batch()
        original_ids = batch.item_features["id"].clone()
        contextual = Batch(
            user_features=batch.user_features,
            item_features={"id": batch.item_features["id"], "text": batch.item_features["text"]},
            history_features=batch.history_features,
            history_mask=batch.history_mask,
            labels=batch.labels,
            context_features={"image": batch.item_features["image"]},
        )
        for name in ("dnn_mm_seq", "naml", "make"):
            with self.subTest(name=name):
                model_config, data_config = configs()
                output = create_model(name, model_config, data_config)(contextual)
                self.assertEqual((2,), tuple(output.logits.shape))
        self.assertTrue(torch.equal(contextual.item_features["id"], original_ids))

    def test_registry_keeps_legacy_regression_metadata(self):
        for name in ("dnn_mm_seq", "naml", "make"):
            specification = model_spec(name)
            self.assertEqual("mmctr.models.sequence", specification.module)
            self.assertIn("legacy_module", specification.metadata)
            self.assertIn("legacy_symbol", specification.metadata)


if __name__ == "__main__":
    unittest.main()

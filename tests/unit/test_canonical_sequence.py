import unittest

import torch

from mmctr.core import Batch
from mmctr.models.common.base import BaseSeqModel, HistoryCapability
from mmctr.models.common.registry import create_model, model_spec


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

    def test_naml_masks_missing_views_using_raw_presence_not_projection_values(self):
        model_config, data_config = configs()
        model = create_model("naml", model_config, data_config)
        batch = make_batch()
        # A present text view projects to zero; a missing image would gain bias
        # without masking. Both cases must retain their raw presence semantics.
        with torch.no_grad():
            model.projectors["text"].weight.zero_()
            model.projectors["text"].bias.zero_()
            model.projectors["image"].bias.fill_(10.0)
        batch = Batch(
            user_features=batch.user_features,
            item_features={**batch.item_features, "image": torch.zeros(2, 2)},
            history_features={
                **batch.history_features,
                "image": torch.zeros_like(batch.history_features["image"]),
            },
            history_mask=batch.history_mask,
            labels=batch.labels,
        )
        output = model(batch)
        target_weights = output.representations["target_view_weights"]
        history_weights = output.representations["history_view_weights"]
        self.assertTrue((target_weights[:, 1] > 0).all())
        self.assertTrue(torch.equal(target_weights[:, 2], torch.zeros(2)))
        torch.testing.assert_close(target_weights.sum(-1), torch.ones(2))
        self.assertTrue(torch.equal(history_weights[..., 2], torch.zeros(2, 2)))
        torch.testing.assert_close(history_weights.sum(-1), batch.history_mask.float())
        output.logits.sum().backward()
        for parameter in model.modal_fusion.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_naml_id_only_and_all_padding_attention(self):
        model_config, data_config = configs()
        data_config["use_mm_features"] = ["id"]
        model = create_model("naml", model_config, data_config)
        output = model(make_batch(all_padding=True))
        torch.testing.assert_close(output.representations["target_view_weights"], torch.ones(2, 1))
        self.assertTrue(
            torch.equal(output.representations["history_view_weights"], torch.zeros(2, 2, 1))
        )
        self.assertTrue(torch.equal(output.representations["user_interest"], torch.zeros(2, 4)))
        self.assertTrue(torch.isfinite(output.logits).all())

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

    def test_registry_uses_individual_model_modules(self):
        for name in ("dnn_mm_seq", "naml", "make"):
            specification = model_spec(name)
            self.assertEqual("mmctr.models.mm_models.{}".format(name), specification.module)


if __name__ == "__main__":
    unittest.main()

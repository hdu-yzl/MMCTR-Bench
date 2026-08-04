import unittest

import torch

from mmctr.core import Batch, ContractError
from mmctr.models.base import BaseSeqModel, HistoryCapability
from mmctr.models.registry import create_model, model_spec


MODEL_NAMES = ("mb", "pamd", "mmmlp", "m3srec")


def configs():
    model_config = {
        "latent_dim": 8,
        "projection_dim": 8,
        "mlp_dims": [16],
        "dropout": 0.0,
        "batch_norm": False,
        "mb_balance_weight": 0.1,
        "mb_sample_num": 2,
        "mb_adv_eps": 0.1,
        "mb_pgd_steps": 1,
        "mb_pgd_step_size": 0.05,
        "mb_fusion_hidden_dim": 8,
        "pamd_hidden_dim": 8,
        "pamd_aux_weight": 0.1,
        "pamd_layer_norm": True,
        "mixer_layers": 1,
        "feature_mixer_layers": 1,
        "fusion_mixer_layers": 1,
        "token_hidden_dim": 6,
        "channel_hidden_dim": 16,
        "fusion_channel_hidden_dim": 32,
        "num_heads": 2,
        "num_experts": 2,
        "moe_hidden_dim": 16,
        "attn_ffn_dim": 16,
        "num_specific_layers": 1,
        "num_cross_layers": 1,
        "fusion_hidden_dim": 8,
        "max_seq_len": 3,
    }
    data_config = {
        "id_feature_num": 64,
        "seq_len": 3,
        "use_mm_features": ["id", "image", "text"],
        "use_mm_seq_features": ["id", "image", "text"],
        "mm_dims": {"image": 3, "text": 2},
        "mm_seq_dims": {"image": 3, "text": 2},
        "user_features": ["id"],
        "user_features_dim": {},
    }
    return model_config, data_config


def make_batch(batch_size=2, all_padding=False, padded_value=0.0):
    user_ids = torch.arange(1, batch_size + 1, dtype=torch.long).unsqueeze(1)
    item_ids = torch.arange(5, 5 + batch_size, dtype=torch.long).unsqueeze(1)
    history_ids = torch.tensor([[0, 5, 6], [0, 0, 7]], dtype=torch.long)[:batch_size]
    history_mask = history_ids.ne(0)
    if all_padding:
        history_ids = torch.zeros_like(history_ids)
        history_mask = torch.zeros_like(history_mask)
    image = torch.arange(batch_size * 3, dtype=torch.float32).view(batch_size, 3)
    text = torch.arange(batch_size * 2, dtype=torch.float32).view(batch_size, 2)
    history_image = torch.zeros(batch_size, 3, 3)
    history_text = torch.zeros(batch_size, 3, 2)
    history_image[history_mask] = 1.0
    history_text[history_mask] = 2.0
    history_image[~history_mask] = padded_value
    history_text[~history_mask] = padded_value
    return Batch(
        user_features={"id": user_ids},
        item_features={"id": item_ids, "image": image, "text": text},
        history_features={
            "id": history_ids,
            "image": history_image,
            "text": history_text,
        },
        history_mask=history_mask,
        labels=torch.arange(batch_size, dtype=torch.float32).remainder(2),
    )


class CanonicalSpecializedTests(unittest.TestCase):
    def test_models_use_canonical_contract_and_backward(self):
        expected_capabilities = {
            "mb": HistoryCapability.POOLED_HISTORY,
            "pamd": HistoryCapability.POOLED_HISTORY,
            "mmmlp": HistoryCapability.SEQUENCE_TOKENS,
            "m3srec": HistoryCapability.SEQUENCE_TOKENS,
        }
        for name in MODEL_NAMES:
            with self.subTest(name=name):
                model_config, data_config = configs()
                model = create_model(name, model_config, data_config)
                self.assertIsInstance(model, BaseSeqModel)
                self.assertEqual(expected_capabilities[name], model.history_capability)
                output = model(make_batch())
                self.assertEqual((2,), tuple(output.logits.shape))
                objective = output.logits.sum() + output.auxiliary_loss()
                objective.backward()
                self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_named_auxiliary_losses_are_finite_scalars(self):
        expected = {
            "mb": {"mb_modality_balance"},
            "pamd": {"pamd_disentanglement"},
            "mmmlp": set(),
            "m3srec": set(),
        }
        for name in MODEL_NAMES:
            with self.subTest(name=name):
                model_config, data_config = configs()
                output = create_model(name, model_config, data_config)(make_batch())
                self.assertEqual(expected[name], set(output.auxiliary_losses))
                for loss in output.auxiliary_losses.values():
                    self.assertEqual(0, loss.ndim)
                    self.assertTrue(torch.isfinite(loss))

    def test_mb_disables_balance_loss_during_evaluation(self):
        model_config, data_config = configs()
        model = create_model("mb", model_config, data_config)
        model.eval()
        loss = model(make_batch()).auxiliary_losses["mb_modality_balance"]
        self.assertEqual(0.0, float(loss))

    def test_all_padding_history_is_finite_and_token_models_return_zero(self):
        for name in MODEL_NAMES:
            with self.subTest(name=name):
                model_config, data_config = configs()
                output = create_model(name, model_config, data_config)(make_batch(all_padding=True))
                self.assertTrue(torch.isfinite(output.logits).all())
                self.assertTrue(torch.isfinite(output.auxiliary_loss()))
                if name in {"mmmlp", "m3srec"}:
                    history = output.representations["history_fusion"]
                    self.assertTrue(torch.equal(history, torch.zeros_like(history)))

    def test_masked_left_padding_values_do_not_change_token_models(self):
        clean = make_batch(padded_value=0.0)
        contaminated = make_batch(padded_value=1000.0)
        for name in ("mmmlp", "m3srec"):
            with self.subTest(name=name):
                model_config, data_config = configs()
                model = create_model(name, model_config, data_config)
                model.eval()
                clean_output = model(clean)
                contaminated_output = model(contaminated)
                self.assertTrue(
                    torch.equal(
                        clean_output.representations["history_fusion"],
                        contaminated_output.representations["history_fusion"],
                    )
                )

    def test_batch_size_one_preserves_logit_rank(self):
        for name in MODEL_NAMES:
            with self.subTest(name=name):
                model_config, data_config = configs()
                output = create_model(name, model_config, data_config)(make_batch(batch_size=1))
                self.assertEqual((1,), tuple(output.logits.shape))
                self.assertEqual(0, output.auxiliary_loss().ndim)

    def test_context_fallback_does_not_mutate_batch(self):
        original = make_batch()
        item_ids = original.item_features["id"].clone()
        contextual = Batch(
            user_features=original.user_features,
            item_features={
                "id": original.item_features["id"],
                "text": original.item_features["text"],
            },
            history_features=original.history_features,
            history_mask=original.history_mask,
            labels=original.labels,
            context_features={"image": original.item_features["image"]},
        )
        for name in MODEL_NAMES:
            with self.subTest(name=name):
                model_config, data_config = configs()
                output = create_model(name, model_config, data_config)(contextual)
                self.assertEqual((2,), tuple(output.logits.shape))
        self.assertTrue(torch.equal(contextual.item_features["id"], item_ids))

    def test_invalid_specialized_configurations_are_rejected(self):
        model_config, data_config = configs()
        model_config["mb_pgd_steps"] = 0
        with self.assertRaises(ContractError):
            create_model("mb", model_config, data_config)

        model_config, data_config = configs()
        data_config["use_mm_features"] = ["id", "image"]
        data_config["use_mm_seq_features"] = ["id", "image"]
        with self.assertRaises(ContractError):
            create_model("pamd", model_config, data_config)

        model_config, data_config = configs()
        data_config["seq_len"] = 0
        with self.assertRaises(ContractError):
            create_model("mmmlp", model_config, data_config)

        model_config, data_config = configs()
        model_config["num_heads"] = 3
        with self.assertRaises(ContractError):
            create_model("m3srec", model_config, data_config)

        model_config, data_config = configs()
        model_config["m3_modalities"] = ["id", "video"]
        with self.assertRaises(ContractError):
            create_model("m3srec", model_config, data_config)

    def test_registry_uses_canonical_specialized_module(self):
        for name in ("mb", "pamd", "mmmlp", "m3srec"):
            specification = model_spec(name)
            self.assertEqual("mmctr.models.specialized", specification.module)


if __name__ == "__main__":
    unittest.main()

import unittest

import torch

from mmctr.core import Batch, ContractError
from mmctr.models.components import feature_presence
from mmctr.models import (
    EXECUTABLE_PIPELINE_MODELS,
    default_pipeline_preset,
    model_pipeline_coverage,
)
from mmctr.models.registry import available_models, create_model
from mmctr.quantization import ResidualQuantizer


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


def sequence_raw_values(model, batch):
    target_values = {}
    target_presence = {}
    history_values = {}
    history_presence = {}
    for name in model.feature_names:
        target = batch.item_features[name]
        history = batch.history_features[name]
        if name == "id":
            target = model.embedding(target).squeeze(1)
            history = model.embedding(history)
            target_presence[name] = torch.ones(batch.batch_size, dtype=torch.bool)
            history_presence[name] = batch.history_mask
        else:
            target_presence[name] = feature_presence(target)
            history_presence[name] = feature_presence(history)
        target_values[name] = target
        history_values[name] = history
    user_values = {}
    user_presence = {}
    for name in model.user_feature_names:
        user = batch.user_features[name]
        if name == "id":
            user = model.embedding(user).squeeze(1)
            user_presence[name] = torch.ones(batch.batch_size, dtype=torch.bool)
        else:
            user_presence[name] = feature_presence(user)
        user_values[name] = user
    return (
        target_values,
        target_presence,
        history_values,
        history_presence,
        user_values,
        user_presence,
    )


def make_rq(dimension, offset=0.0):
    quantizer = ResidualQuantizer(n_levels=2, codebook_size=3, dimension=dimension)
    values = torch.arange(2 * 3 * dimension, dtype=torch.float32)
    quantizer.set_codebooks(values.reshape(2, 3, dimension) / 10.0 + offset)
    return quantizer


class ModelPipelinePresetTests(unittest.TestCase):
    def test_id_baseline_presets_match_existing_shared_encoding_boundary(self):
        model_config = {
            "latent_dim": 4,
            "projection_dim": 5,
            "mlp_dims": [8],
            "dropout": 0.0,
            "batch_norm": False,
            "attn_layers": 1,
            "attn_heads": 1,
            "attn_size": 5,
        }
        data_config = {"id_feature_num": 32}
        batch = make_batch()
        for name in ("dnn", "dcn", "deepfm", "autoint"):
            with self.subTest(name=name):
                model = create_model(name, model_config, data_config)
                pipelines = default_pipeline_preset(name, model_config, data_config).build()
                pipelines["target"].projector.load_state_dict(model.mm_projector.state_dict())
                pipelines["history"].projector.load_state_dict(model.mm_seq_projector.state_dict())
                target_values = {
                    "id": model.embedding(
                        torch.cat([batch.user_features["id"], batch.item_features["id"]], dim=1)
                    ).flatten(start_dim=1)
                }
                history_values = {"id": model.embedding(batch.history_features["id"])}

                expected_target, expected_history = model.encode_ids(batch)
                actual_target = pipelines["target"](target_values).representation
                actual_history = pipelines["history"](
                    history_values, sequence_mask=batch.history_mask
                ).representation

                self.assertTrue(torch.allclose(expected_target, actual_target, atol=1e-7, rtol=0.0))
                self.assertTrue(
                    torch.allclose(expected_history, actual_history, atol=1e-7, rtol=0.0)
                )

    def test_every_registered_model_has_executable_or_explicit_model_specific_preset(self):
        model_config = {
            "latent_dim": 4,
            "projection_dim": 4,
            "rank": 2,
            "fusion_dim": 4,
            "mlp_dims": [8],
            "dropout": 0.0,
        }
        data_config = {
            "use_mm_features": ["id", "text", "image"],
            "use_mm_seq_features": ["id", "text", "image"],
            "mm_dims": {"text": 3, "image": 2},
            "mm_seq_dims": {"text": 3, "image": 2},
            "user_features": ["id"],
            "user_features_dim": {},
        }

        self.assertEqual(set(available_models()), set(model_pipeline_coverage()))
        for name in available_models():
            with self.subTest(name=name):
                preset = default_pipeline_preset(name, model_config, data_config)
                self.assertEqual(name in EXECUTABLE_PIPELINE_MODELS, preset.executable)
                if not preset.executable:
                    self.assertTrue(preset.model_specific_reason)
                    with self.assertRaisesRegex(ContractError, "model-specific"):
                        preset.build()
        self.assertEqual(
            "dnn_mm_seq",
            default_pipeline_preset("dnn_seq", model_config, data_config).model_name,
        )

    def test_dnn_mm_default_preset_resolves_target_and_pooled_history(self):
        model_config = {
            "latent_dim": 3,
            "projection_dim": 4,
            "modal_fusion_method": "cat",
        }
        data_config = {
            "use_mm_features": ["id", "image", "text"],
            "use_mm_seq_features": ["id", "text"],
            "mm_dims": {"image": 2, "text": 5},
            "mm_seq_dims": {"text": 5},
        }

        preset = default_pipeline_preset("dnn_mm", model_config, data_config)
        pipelines = preset.build()

        self.assertTrue(preset.executable)
        self.assertEqual(("target", "history"), pipelines.branches)
        self.assertEqual(("id", "image", "text"), pipelines["target"].modalities)
        self.assertEqual(("id", "text"), pipelines["history"].modalities)
        self.assertEqual(6, pipelines["target"].projector.input_dimensions["id"])
        self.assertEqual(3, pipelines["history"].projector.input_dimensions["id"])
        self.assertEqual("pool_then_fuse", pipelines["history"].config.topology)
        self.assertEqual(12, pipelines["target"].fusion.output_dim)
        self.assertEqual(8, pipelines["history"].fusion.output_dim)

    def test_dnn_mm_preset_matches_existing_default_branch_outputs(self):
        model_config = {
            "latent_dim": 4,
            "projection_dim": 4,
            "mlp_dims": [8],
            "dropout": 0.0,
            "batch_norm": False,
            "modal_fusion_method": "cat",
        }
        data_config = {
            "id_feature_num": 32,
            "use_mm_features": ["id", "text", "image"],
            "use_mm_seq_features": ["id", "text", "image"],
            "mm_dims": {"text": 3, "image": 2},
            "mm_seq_dims": {"text": 3, "image": 2},
        }
        model = create_model("dnn_mm", model_config, data_config)
        pipelines = default_pipeline_preset("dnn_mm", model_config, data_config).build()
        pipelines["target"].projector.load_state_dict(model.target_projectors.state_dict())
        pipelines["history"].projector.load_state_dict(model.history_projectors.state_dict())
        pipelines["target"].fusion.load_state_dict(model.target_fusion.state_dict())
        pipelines["history"].fusion.load_state_dict(model.history_fusion.state_dict())
        batch = make_batch()
        target_values = {
            "id": model.embedding(
                torch.cat([batch.user_features["id"], batch.item_features["id"]], dim=1)
            ).flatten(start_dim=1),
            "text": batch.item_features["text"],
            "image": batch.item_features["image"],
        }
        target_presence = {
            "id": torch.ones(batch.batch_size, dtype=torch.bool),
            "text": feature_presence(target_values["text"]),
            "image": feature_presence(target_values["image"]),
        }
        history_values = {
            "id": model.embedding(batch.history_features["id"]),
            "text": batch.history_features["text"],
            "image": batch.history_features["image"],
        }
        history_presence = {
            "id": batch.history_mask,
            "text": feature_presence(history_values["text"]),
            "image": feature_presence(history_values["image"]),
        }

        expected_target = model.target_fusion(model.project_target(batch)).fused
        projected_history = model.project_history(batch)
        expected_history = model.history_fusion(
            {
                name: model.masked_pool(value, batch.history_mask)
                for name, value in projected_history.items()
            }
        ).fused
        actual_target = pipelines["target"](target_values, presence=target_presence).representation
        actual_history = pipelines["history"](
            history_values,
            presence=history_presence,
            sequence_mask=batch.history_mask,
        ).representation

        self.assertTrue(torch.allclose(expected_target, actual_target, atol=1e-7, rtol=0.0))
        self.assertTrue(torch.allclose(expected_history, actual_history, atol=1e-7, rtol=0.0))

    def test_simcen_preset_matches_existing_ordered_field_boundary(self):
        model_config = {
            "latent_dim": 4,
            "projection_dim": 4,
            "mlp_dims": [8],
            "hidden_unit": [12],
            "dropout": 0.0,
            "batch_norm": False,
            "ego_batch_norm": False,
            "v1_batch_norm": False,
            "v2_batch_norm": False,
        }
        data_config = {
            "id_feature_num": 32,
            "use_mm_features": ["id", "text", "image"],
            "use_mm_seq_features": ["id", "text", "image"],
            "mm_dims": {"text": 3, "image": 2},
            "mm_seq_dims": {"text": 3, "image": 2},
        }
        model = create_model("simcen", model_config, data_config)
        pipelines = default_pipeline_preset("simcen", model_config, data_config).build()
        pipelines["target"].projector.load_state_dict(model.target_projectors.state_dict())
        pipelines["history"].projector.load_state_dict(model.history_projectors.state_dict())
        batch = make_batch()
        target_values = {
            "id": model.embedding(
                torch.cat([batch.user_features["id"], batch.item_features["id"]], dim=1)
            ).flatten(start_dim=1),
            "text": batch.item_features["text"],
            "image": batch.item_features["image"],
        }
        history_values = {
            "id": model.embedding(batch.history_features["id"]),
            "text": batch.history_features["text"],
            "image": batch.history_features["image"],
        }
        target_presence = {
            modality: (
                torch.ones(batch.batch_size, dtype=torch.bool)
                if modality == "id"
                else feature_presence(target_values[modality])
            )
            for modality in target_values
        }
        history_presence = {
            modality: (
                batch.history_mask
                if modality == "id"
                else feature_presence(history_values[modality])
            )
            for modality in history_values
        }

        projected_target = model.project_target(batch)
        projected_history = model.project_history(batch)
        expected_target = torch.cat(
            [projected_target[name] for name in model.feature_names], dim=-1
        )
        expected_history = torch.cat(
            [
                model.masked_pool(projected_history[name], batch.history_mask)
                for name in model.history_feature_names
            ],
            dim=-1,
        )
        actual_target = pipelines["target"](target_values, presence=target_presence).representation
        actual_history = pipelines["history"](
            history_values,
            presence=history_presence,
            sequence_mask=batch.history_mask,
        ).representation

        self.assertTrue(torch.allclose(expected_target, actual_target, atol=1e-7, rtol=0.0))
        self.assertTrue(torch.allclose(expected_history, actual_history, atol=1e-7, rtol=0.0))

    def test_qarm_preset_matches_existing_quantized_field_boundary(self):
        model_config = {
            "latent_dim": 4,
            "projection_dim": 4,
            "mlp_dims": [8],
            "dropout": 0.0,
            "batch_norm": False,
            "n_levels": 2,
            "codebook_size": 3,
            "cross_num": 2,
        }
        data_config = {
            "name": "fixture",
            "id_feature_num": 32,
            "use_mm_features": ["id", "image", "text"],
            "mm_seq_dims": {"image": 2, "text": 3},
            "user_features": ["id"],
            "user_features_dim": {},
        }
        model = create_model(
            "qarm",
            model_config,
            data_config,
            quantizers={"image": make_rq(2), "text": make_rq(3, 0.5)},
        )
        pipelines = default_pipeline_preset("qarm", model_config, data_config).build()
        pipelines["history"].projector.load_state_dict(model.projectors.state_dict())
        pipelines["user"].projector.load_state_dict(model.user_projectors.state_dict())
        batch = make_batch()
        target_values = {"id": model.embedding(batch.item_features["id"]).squeeze(1)}
        history_values = {"id": model.embedding(batch.history_features["id"])}
        for name in model.quantized_modalities:
            target_values[name] = model._encode(name, batch.item_features[name])
            history_values[name] = model._encode(name, batch.history_features[name])
        target_presence = {
            name: (
                torch.ones(batch.batch_size, dtype=torch.bool)
                if name == "id"
                else feature_presence(batch.item_features[name])
            )
            for name in model.feature_names
        }
        history_presence = {
            name: (
                batch.history_mask
                if name == "id"
                else feature_presence(batch.history_features[name])
            )
            for name in model.feature_names
        }
        user_values = {"id": model.embedding(batch.user_features["id"]).squeeze(1)}

        projected_target = model._project_quantized_target(batch)
        projected_history = model._project_quantized_history(batch)
        expected_target = torch.cat(
            [projected_target[name] for name in model.feature_names], dim=-1
        )
        expected_history = torch.cat(
            [
                model.masked_pool(projected_history[name], batch.history_mask)
                for name in model.feature_names
            ],
            dim=-1,
        )
        expected_user = model.project_user(batch)
        actual_target = pipelines["target"](target_values, presence=target_presence).representation
        actual_history = pipelines["history"](
            history_values,
            presence=history_presence,
            sequence_mask=batch.history_mask,
        ).representation
        actual_user = pipelines["user"](user_values).representation

        self.assertTrue(torch.allclose(expected_target, actual_target, atol=1e-7, rtol=0.0))
        self.assertTrue(torch.allclose(expected_history, actual_history, atol=1e-7, rtol=0.0))
        self.assertTrue(torch.allclose(expected_user, actual_user, atol=1e-7, rtol=0.0))

    def test_dmf_preset_matches_modality_centers_before_private_dta_head(self):
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
        }
        data_config = {
            "id_feature_num": 32,
            "use_mm_features": ["id", "text", "image"],
            "mm_seq_dims": {"text": 3, "image": 2},
            "user_features": ["id"],
            "user_features_dim": {},
        }
        model = create_model("dmf", model_config, data_config)
        pipelines = default_pipeline_preset("dmf", model_config, data_config).build()
        for name in model.non_id_features:
            pipelines["target"].projector[name].load_state_dict(model.projectors[name].state_dict())
        pipelines["target"].fusion.load_state_dict(model.modal_fusion.state_dict())
        pipelines["user"].projector.load_state_dict(model.user_projectors.state_dict())
        batch = make_batch()
        (
            target_values,
            target_presence,
            history_values,
            history_presence,
            user_values,
            user_presence,
        ) = sequence_raw_values(model, batch)
        target_values = {name: target_values[name] for name in model.non_id_features}
        target_presence = {name: target_presence[name] for name in model.non_id_features}
        history_values = {name: history_values[name] for name in model.non_id_features}
        history_presence = {name: history_presence[name] for name in model.non_id_features}

        projected_target = model.project_target(batch)
        projected_history = model.project_history(batch)
        expected_target = model.modal_fusion(
            {name: projected_target[name] for name in model.non_id_features}
        ).fused
        expected_history = model.modal_fusion(
            {name: projected_history[name] for name in model.non_id_features}
        ).fused
        expected_user = model.project_user(batch)
        actual_target = pipelines["target"](target_values, presence=target_presence).representation
        actual_history = pipelines["history"](
            history_values,
            presence=history_presence,
            sequence_mask=batch.history_mask,
        ).representation
        actual_user = pipelines["user"](user_values, presence=user_presence).representation

        self.assertTrue(torch.allclose(expected_target, actual_target, atol=1e-7, rtol=0.0))
        self.assertTrue(torch.allclose(expected_history, actual_history, atol=1e-7, rtol=0.0))
        self.assertTrue(torch.allclose(expected_user, actual_user, atol=1e-7, rtol=0.0))

    def test_dnn_mm_seq_preset_keeps_shared_item_projection_and_user_branch(self):
        model_config = {
            "latent_dim": 4,
            "projection_dim": 6,
            "modal_fusion_method": "add",
        }
        data_config = {
            "use_mm_features": ["id", "text", "image"],
            "mm_seq_dims": {"text": 3, "image": 2},
            "user_features": ["id", "profile"],
            "user_features_dim": {"profile": 5},
        }

        preset = default_pipeline_preset("dnn_mm_seq", model_config, data_config)
        pipelines = preset.build()

        self.assertEqual(("target", "history", "user"), pipelines.branches)
        self.assertIs(pipelines["target"].projector, pipelines["history"].projector)
        self.assertIsNot(pipelines["target"].fusion, pipelines["history"].fusion)
        self.assertEqual("pool_then_fuse", pipelines["history"].config.topology)
        self.assertEqual(12, pipelines["user"].fusion.output_dim)

    def test_dnn_mm_seq_preset_matches_existing_default_branch_outputs(self):
        model_config = {
            "latent_dim": 4,
            "projection_dim": 4,
            "mlp_dims": [8],
            "dropout": 0.0,
            "batch_norm": False,
            "modal_fusion_method": "add",
        }
        data_config = {
            "id_feature_num": 32,
            "use_mm_features": ["id", "text", "image"],
            "mm_seq_dims": {"text": 3, "image": 2},
            "user_features": ["id"],
            "user_features_dim": {},
        }
        model = create_model("dnn_mm_seq", model_config, data_config)
        pipelines = default_pipeline_preset("dnn_mm_seq", model_config, data_config).build()
        pipelines["target"].projector.load_state_dict(model.projectors.state_dict())
        pipelines["user"].projector.load_state_dict(model.user_projectors.state_dict())
        pipelines["target"].fusion.load_state_dict(model.target_fusion.state_dict())
        pipelines["history"].fusion.load_state_dict(model.history_fusion.state_dict())
        batch = make_batch()
        (
            target_values,
            target_presence,
            history_values,
            history_presence,
            user_values,
            user_presence,
        ) = sequence_raw_values(model, batch)

        expected_target = model.target_fusion(model.project_target(batch)).fused
        expected_history = model.history_fusion(
            {
                name: model.masked_pool(value, batch.history_mask)
                for name, value in model.project_history(batch).items()
            }
        ).fused
        expected_user = model.project_user(batch)
        actual_target = pipelines["target"](target_values, presence=target_presence).representation
        actual_history = pipelines["history"](
            history_values,
            presence=history_presence,
            sequence_mask=batch.history_mask,
        ).representation
        actual_user = pipelines["user"](user_values, presence=user_presence).representation

        self.assertTrue(torch.allclose(expected_target, actual_target, atol=1e-7, rtol=0.0))
        self.assertTrue(torch.allclose(expected_history, actual_history, atol=1e-7, rtol=0.0))
        self.assertTrue(torch.allclose(expected_user, actual_user, atol=1e-7, rtol=0.0))

    def test_lmf_and_mtfn_presets_fuse_history_tokens_before_mean_pooling(self):
        model_config = {
            "latent_dim": 3,
            "projection_dim": 4,
            "rank": 2,
            "fusion_dim": 5,
        }
        data_config = {
            "use_mm_features": ["id", "text"],
            "use_mm_seq_features": ["id", "text"],
            "mm_dims": {"text": 3},
            "mm_seq_dims": {"text": 3},
        }
        for name, expected_dim in (("lmf", 5), ("mtfn", 4)):
            with self.subTest(name=name):
                pipelines = default_pipeline_preset(name, model_config, data_config).build()
                self.assertEqual("fuse_then_pool", pipelines["history"].config.topology)
                self.assertEqual(expected_dim, pipelines["target"].output_dim)
                self.assertEqual(expected_dim, pipelines["history"].output_dim)
                self.assertEqual(
                    "MeanPooling", pipelines["history"].sequence_pooling.__class__.__name__
                )

    def test_lmf_and_mtfn_presets_match_existing_default_branch_outputs(self):
        model_config = {
            "latent_dim": 4,
            "projection_dim": 4,
            "mlp_dims": [8],
            "dropout": 0.0,
            "batch_norm": False,
            "rank": 2,
            "fusion_dim": 5,
        }
        data_config = {
            "id_feature_num": 32,
            "use_mm_features": ["id", "text", "image"],
            "use_mm_seq_features": ["id", "text", "image"],
            "mm_dims": {"text": 3, "image": 2},
            "mm_seq_dims": {"text": 3, "image": 2},
        }
        batch = make_batch()
        for name in ("lmf", "mtfn"):
            with self.subTest(name=name):
                model = create_model(name, model_config, data_config)
                pipelines = default_pipeline_preset(name, model_config, data_config).build()
                pipelines["target"].projector.load_state_dict(model.target_projectors.state_dict())
                pipelines["history"].projector.load_state_dict(
                    model.history_projectors.state_dict()
                )
                pipelines["target"].fusion.load_state_dict(model.target_fusion.state_dict())
                pipelines["history"].fusion.load_state_dict(model.history_fusion.state_dict())
                target_values = {
                    "id": model.embedding(
                        torch.cat([batch.user_features["id"], batch.item_features["id"]], dim=1)
                    ).flatten(start_dim=1),
                    "text": batch.item_features["text"],
                    "image": batch.item_features["image"],
                }
                history_values = {
                    "id": model.embedding(batch.history_features["id"]),
                    "text": batch.history_features["text"],
                    "image": batch.history_features["image"],
                }
                target_presence = {
                    modality: (
                        torch.ones(batch.batch_size, dtype=torch.bool)
                        if modality == "id"
                        else feature_presence(target_values[modality])
                    )
                    for modality in target_values
                }
                history_presence = {
                    modality: (
                        batch.history_mask
                        if modality == "id"
                        else feature_presence(history_values[modality])
                    )
                    for modality in history_values
                }

                expected_target = model.target_fusion(model.project_target(batch)).fused
                expected_history = model.masked_pool(
                    model.history_fusion(model.project_history(batch)).fused,
                    batch.history_mask,
                )
                actual_target = pipelines["target"](
                    target_values, presence=target_presence
                ).representation
                actual_history = pipelines["history"](
                    history_values,
                    presence=history_presence,
                    sequence_mask=batch.history_mask,
                ).representation

                self.assertTrue(torch.allclose(expected_target, actual_target, atol=1e-7, rtol=0.0))
                self.assertTrue(
                    torch.allclose(expected_history, actual_history, atol=1e-7, rtol=0.0)
                )

    def test_naml_preset_preserves_shared_maf_and_attention_pooling(self):
        model_config = {"latent_dim": 4, "projection_dim": 4}
        data_config = {
            "use_mm_features": ["id", "text", "image"],
            "mm_seq_dims": {"text": 3, "image": 2},
            "user_features": ["id"],
            "user_features_dim": {},
        }

        pipelines = default_pipeline_preset("naml", model_config, data_config).build()

        self.assertIs(pipelines["target"].projector, pipelines["history"].projector)
        self.assertIs(pipelines["target"].fusion, pipelines["history"].fusion)
        self.assertEqual("fuse_then_pool", pipelines["history"].config.topology)
        self.assertEqual(
            "AttentionPooling", pipelines["history"].sequence_pooling.__class__.__name__
        )

    def test_naml_preset_matches_existing_default_branch_outputs(self):
        model_config = {
            "latent_dim": 4,
            "projection_dim": 4,
            "mlp_dims": [8],
            "dropout": 0.0,
            "batch_norm": False,
        }
        data_config = {
            "id_feature_num": 32,
            "use_mm_features": ["id", "text", "image"],
            "mm_seq_dims": {"text": 3, "image": 2},
            "user_features": ["id"],
            "user_features_dim": {},
        }
        model = create_model("naml", model_config, data_config)
        pipelines = default_pipeline_preset("naml", model_config, data_config).build()
        pipelines["history"].projector.load_state_dict(model.projectors.state_dict())
        pipelines["user"].projector.load_state_dict(model.user_projectors.state_dict())
        pipelines["target"].fusion.load_state_dict(model.modal_fusion.state_dict())
        pipelines["history"].sequence_pooling.load_state_dict(model.user_encoder.state_dict())
        batch = make_batch()
        (
            target_values,
            target_presence,
            history_values,
            history_presence,
            user_values,
            user_presence,
        ) = sequence_raw_values(model, batch)

        expected_target = model.modal_fusion(model.project_target(batch)).fused
        expected_history = model.user_encoder(
            model.modal_fusion(model.project_history(batch)).fused,
            batch.history_mask,
        )
        expected_user = model.project_user(batch)
        actual_target = pipelines["target"](target_values, presence=target_presence).representation
        actual_history = pipelines["history"](
            history_values,
            presence=history_presence,
            sequence_mask=batch.history_mask,
        ).representation
        actual_user = pipelines["user"](user_values, presence=user_presence).representation

        self.assertTrue(torch.allclose(expected_target, actual_target, atol=1e-7, rtol=0.0))
        self.assertTrue(torch.allclose(expected_history, actual_history, atol=1e-7, rtol=0.0))
        self.assertTrue(torch.allclose(expected_user, actual_user, atol=1e-7, rtol=0.0))

    def test_make_preset_fuses_shared_target_for_din_history_pooling(self):
        model_config = {
            "latent_dim": 4,
            "projection_dim": 4,
            "modal_fusion_method": "cat",
            "mlp_dims": [7, 5],
            "dropout": 0.0,
        }
        data_config = {
            "use_mm_features": ["id", "text", "image"],
            "mm_seq_dims": {"text": 3, "image": 2},
            "user_features": ["id"],
            "user_features_dim": {},
        }

        pipelines = default_pipeline_preset("make", model_config, data_config).build()

        self.assertIs(pipelines["target"].projector, pipelines["history"].projector)
        self.assertIs(pipelines["target"].fusion, pipelines["history"].fusion)
        self.assertEqual(("id", "text", "image"), pipelines["history"].target_modalities)
        self.assertEqual(12, pipelines["history"].sequence_pooling.input_dim)
        self.assertEqual(
            (7, 5),
            tuple(
                layer.out_features
                for layer in pipelines["history"].sequence_pooling.network
                if isinstance(layer, torch.nn.Linear)
            )[:-1],
        )

    def test_make_preset_matches_existing_default_branch_outputs(self):
        model_config = {
            "latent_dim": 4,
            "projection_dim": 4,
            "mlp_dims": [8],
            "dropout": 0.0,
            "batch_norm": False,
            "modal_fusion_method": "cat",
            "tier_num": 4,
        }
        data_config = {
            "id_feature_num": 32,
            "use_mm_features": ["id", "text", "image"],
            "mm_seq_dims": {"text": 3, "image": 2},
            "user_features": ["id"],
            "user_features_dim": {},
        }
        model = create_model("make", model_config, data_config)
        pipelines = default_pipeline_preset("make", model_config, data_config).build()
        pipelines["history"].projector.load_state_dict(model.projectors.state_dict())
        pipelines["user"].projector.load_state_dict(model.user_projectors.state_dict())
        pipelines["target"].fusion.load_state_dict(model.modal_fusion.state_dict())
        pipelines["history"].sequence_pooling.load_state_dict(model.attention.state_dict())
        batch = make_batch()
        (
            target_values,
            target_presence,
            history_values,
            history_presence,
            user_values,
            user_presence,
        ) = sequence_raw_values(model, batch)

        expected_target = model.modal_fusion(model.project_target(batch)).fused
        expected_history = model.attention(
            model.modal_fusion(model.project_history(batch)).fused,
            batch.history_mask,
            expected_target,
        )
        expected_user = model.project_user(batch)
        actual_target = pipelines["target"](target_values, presence=target_presence).representation
        actual_history = pipelines["history"](
            history_values,
            presence=history_presence,
            sequence_mask=batch.history_mask,
            targets=target_values,
        ).representation
        actual_user = pipelines["user"](user_values, presence=user_presence).representation

        self.assertTrue(torch.allclose(expected_target, actual_target, atol=1e-7, rtol=0.0))
        self.assertTrue(torch.allclose(expected_history, actual_history, atol=1e-7, rtol=0.0))
        self.assertTrue(torch.allclose(expected_user, actual_user, atol=1e-7, rtol=0.0))


if __name__ == "__main__":
    unittest.main()

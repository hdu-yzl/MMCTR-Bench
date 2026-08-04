import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from mmctr.core import Batch, ContractError
from mmctr.models.base import BaseSeqModel, HistoryCapability
from mmctr.models.registry import create_model, create_model_from_artifacts, model_spec
from mmctr.quantization import (
    PSRQPretrainer,
    QuantizationArtifactError,
    ResidualQuantizer,
    psrq_artifact_path,
    quantizer_spec,
    rq_artifact_path,
)
from mmctr.quantization.training import copy_feature_tables


def configs():
    model_config = {
        "latent_dim": 4,
        "projection_dim": 4,
        "mlp_dims": [8],
        "psrq_dims": [6],
        "dropout": 0.0,
        "batch_norm": False,
        "n_levels": 2,
        "codebook_size": 3,
        "cross_num": 2,
    }
    data_config = {
        "name": "fixture",
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


def make_rq(dimension, offset=0.0):
    quantizer = ResidualQuantizer(
        n_levels=2,
        codebook_size=3,
        dimension=dimension,
    )
    values = torch.arange(2 * 3 * dimension, dtype=torch.float32)
    quantizer.set_codebooks(values.reshape(2, 3, dimension) / 10.0 + offset)
    return quantizer


def initialized_psrq():
    model_config, data_config = configs()
    model = PSRQPretrainer(model_config, data_config)
    with torch.no_grad():
        for module in model.modules():
            if hasattr(module, "initted") and hasattr(module, "embedding"):
                module.embedding.weight.copy_(
                    torch.arange(module.embedding.weight.numel(), dtype=torch.float32).reshape_as(
                        module.embedding.weight
                    )
                    / 20.0
                )
                module.initted.fill_(True)
    model.eval()
    return model


def make_batch(batch_size=2, all_padding=False, padded_value=0.0):
    user_ids = torch.arange(1, batch_size + 1, dtype=torch.long).unsqueeze(1)
    item_ids = torch.arange(5, 5 + batch_size, dtype=torch.long).unsqueeze(1)
    history_ids = torch.tensor([[0, 5, 6], [0, 0, 7]], dtype=torch.long)[:batch_size]
    history_mask = history_ids.ne(0)
    if all_padding:
        history_ids = torch.zeros_like(history_ids)
        history_mask = torch.zeros_like(history_mask)
    image = torch.arange(batch_size * 3, dtype=torch.float32).reshape(batch_size, 3)
    text = torch.arange(batch_size * 2, dtype=torch.float32).reshape(batch_size, 2)
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


class QuantizationArtifactTests(unittest.TestCase):
    def test_pretraining_feature_tables_own_writable_memory(self):
        source = np.arange(12, dtype=np.float32).reshape(4, 3)
        source.setflags(write=False)

        copied = copy_feature_tables({"image": source}, ("image",))

        self.assertTrue(copied["image"].numpy().flags.writeable)
        copied["image"][0, 0] = -1.0
        self.assertEqual(0.0, float(source[0, 0]))

    def test_rq_round_trip_preserves_codes_and_reconstruction(self):
        quantizer = make_rq(3)
        values = torch.tensor([[0.1, 0.2, 0.3], [1.0, 0.5, 0.25]])
        expected_codes, expected_values = quantizer.encode(values)
        self.assertTrue(torch.equal(expected_values, quantizer.decode(expected_codes)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image-rq"
            saved = quantizer.save(path, metadata={"dataset": "fixture", "modality": "image"})
            loaded = ResidualQuantizer.from_artifact(saved)
            actual_codes, actual_values = loaded.encode(values)
        self.assertTrue(torch.equal(expected_codes, actual_codes))
        self.assertTrue(torch.equal(expected_values, actual_values))
        self.assertEqual("fixture", loaded.artifact_metadata["dataset"])

    def test_psrq_pretraining_output_has_named_scalar_losses_and_backward(self):
        model = initialized_psrq()
        model.train()
        features = {
            "image": torch.arange(12, dtype=torch.float32).reshape(4, 3),
            "text": torch.arange(8, dtype=torch.float32).reshape(4, 2),
        }
        output = model(features)
        self.assertEqual({"image", "text", "joint"}, set(output.losses))
        self.assertTrue(all(loss.ndim == 0 for loss in output.losses.values()))
        output.total_loss().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_psrq_round_trip_preserves_item_codes(self):
        model = initialized_psrq()
        features = {
            "image": torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.0, 1.0]]),
            "text": torch.tensor([[1.0, 2.0], [0.0, 1.0]]),
        }
        expected_modalities, expected_joint = model.encode_items(features)
        with tempfile.TemporaryDirectory() as directory:
            path = model.save(Path(directory) / "psrq")
            loaded = PSRQPretrainer.from_artifact(path)
            actual_modalities, actual_joint = loaded.encode_items(features)
        self.assertTrue(torch.equal(expected_joint, actual_joint))
        for name in expected_modalities:
            self.assertTrue(torch.equal(expected_modalities[name], actual_modalities[name]))

    def test_array_checksum_rejects_tampering(self):
        quantizer = make_rq(3)
        with tempfile.TemporaryDirectory() as directory:
            path = quantizer.save(Path(directory) / "rq")
            with np.load(str(path), allow_pickle=False) as payload:
                manifest = np.array(payload["__manifest__"], copy=True)
                codebooks = np.array(payload["codebooks"], copy=True)
            codebooks[0, 0, 0] += 1.0
            np.savez(str(path), __manifest__=manifest, codebooks=codebooks)
            with self.assertRaisesRegex(QuantizationArtifactError, "checksum"):
                ResidualQuantizer.from_artifact(path)

    def test_manifest_is_versioned_json_and_pickle_free(self):
        with tempfile.TemporaryDirectory() as directory:
            path = make_rq(2).save(Path(directory) / "rq")
            with np.load(str(path), allow_pickle=False) as payload:
                manifest = json.loads(str(payload["__manifest__"].item()))
        self.assertEqual("mmctr-quantization-npz", manifest["format"])
        self.assertEqual(1, manifest["version"])
        self.assertEqual("residual-quantizer", manifest["kind"])


class CanonicalQuantizedModelTests(unittest.TestCase):
    def _model(self, name):
        model_config, data_config = configs()
        if name == "qarm":
            return create_model(
                name,
                model_config,
                data_config,
                quantizers={"image": make_rq(3), "text": make_rq(2, 0.5)},
            )
        return create_model(
            name,
            model_config,
            data_config,
            quantizer=initialized_psrq(),
        )

    def test_models_use_canonical_contract_and_backward(self):
        for name in ("qarm", "psrq"):
            with self.subTest(name=name):
                model = self._model(name)
                self.assertIsInstance(model, BaseSeqModel)
                self.assertEqual(HistoryCapability.SEQUENCE_TOKENS, model.history_capability)
                output = model(make_batch())
                self.assertEqual((2,), tuple(output.logits.shape))
                output.logits.sum().backward()
                self.assertTrue(
                    any(
                        parameter.grad is not None
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    )
                )

    def test_composition_loader_builds_models_from_stable_artifact_layout(self):
        model_config, data_config = configs()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for modality, dimension in (("image", 3), ("text", 2)):
                make_rq(dimension).save(
                    rq_artifact_path(root, "fixture", modality),
                    metadata={"dataset": "fixture", "modality": modality},
                )
            initialized_psrq().save(psrq_artifact_path(root, "fixture"))
            qarm = create_model_from_artifacts("qarm", model_config, data_config, root)
            psrq = create_model_from_artifacts("psrq", model_config, data_config, root)
            self.assertEqual((2,), tuple(qarm(make_batch()).logits.shape))
            self.assertEqual((2,), tuple(psrq(make_batch()).logits.shape))

    def test_batch_size_one_and_all_padding_are_stable(self):
        for name in ("qarm", "psrq"):
            with self.subTest(name=name):
                model = self._model(name)
                single = model(make_batch(batch_size=1, all_padding=True))
                self.assertEqual((1,), tuple(single.logits.shape))
                self.assertTrue(torch.isfinite(single.logits).all())
                history = single.representations["history_quantized"]
                self.assertTrue(torch.equal(history, torch.zeros_like(history)))

    def test_masked_padding_values_do_not_change_history_representation(self):
        clean = make_batch(padded_value=0.0)
        contaminated = make_batch(padded_value=1000.0)
        for name in ("qarm", "psrq"):
            with self.subTest(name=name):
                model = self._model(name)
                model.eval()
                clean_history = model(clean).representations["history_quantized"]
                contaminated_history = model(contaminated).representations["history_quantized"]
                self.assertTrue(torch.equal(clean_history, contaminated_history))

    def test_psrq_keeps_frozen_psrq_in_evaluation_mode(self):
        model = self._model("psrq")
        model.train()
        self.assertFalse(model.quantizer.training)
        self.assertFalse(any(parameter.requires_grad for parameter in model.quantizer.parameters()))

    def test_zero_target_modalities_remain_zero_after_projection(self):
        original = make_batch()
        missing = Batch(
            user_features=original.user_features,
            item_features={
                "id": original.item_features["id"],
                "image": torch.zeros_like(original.item_features["image"]),
                "text": torch.zeros_like(original.item_features["text"]),
            },
            history_features=original.history_features,
            history_mask=original.history_mask,
            labels=original.labels,
        )
        qarm_target = self._model("qarm")(missing).representations["target_quantized"]
        self.assertTrue(torch.equal(qarm_target[:, 4:], torch.zeros_like(qarm_target[:, 4:])))
        psrq_joint = self._model("psrq")(missing).representations["joint_quantized"]
        self.assertTrue(torch.equal(psrq_joint, torch.zeros_like(psrq_joint)))

    def test_context_fallback_does_not_mutate_batch(self):
        original = make_batch()
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
        original_ids = contextual.item_features["id"].clone()
        for name in ("qarm", "psrq"):
            with self.subTest(name=name):
                self.assertEqual((2,), tuple(self._model(name)(contextual).logits.shape))
        self.assertTrue(torch.equal(contextual.item_features["id"], original_ids))

    def test_incompatible_dependencies_are_rejected(self):
        model_config, data_config = configs()
        with self.assertRaisesRegex(ContractError, "match non-ID modalities"):
            create_model(
                "qarm",
                model_config,
                data_config,
                quantizers={"image": make_rq(3)},
            )
        wrong = dict(model_config)
        wrong["n_levels"] = 3
        with self.assertRaisesRegex(ContractError, "PSRQ benchmark consumer structure"):
            create_model("psrq", wrong, data_config, quantizer=initialized_psrq())

    def test_registry_separates_canonical_premodels(self):
        expected = {
            "qarm": ("mmctr.models.quantized", "QARM"),
            "psrq": ("mmctr.models.quantized", "MCCA"),
        }
        for name, (module, symbol) in expected.items():
            specification = model_spec(name)
            self.assertEqual(module, specification.module)
            self.assertEqual(symbol, specification.symbol)
        quantizers = {
            "rq": ("mmctr.quantization.residual", "ResidualQuantizer"),
            "psrq": ("mmctr.quantization.psrq", "PSRQPretrainer"),
        }
        for name, (module, symbol) in quantizers.items():
            specification = quantizer_spec(name)
            self.assertEqual(module, specification.module)
            self.assertEqual(symbol, specification.symbol)


if __name__ == "__main__":
    unittest.main()

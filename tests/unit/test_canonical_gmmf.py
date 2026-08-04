import json
import tempfile
import unittest
from pathlib import Path

import torch

from mmctr.core import Batch, ContractError
from mmctr.models.base import BaseSeqModel, HistoryCapability
from mmctr.models.registry import create_model, model_spec
from mmctr.training import (
    AlternatingPhase,
    CheckpointManager,
    TrainingEngine,
    build_phased_adam,
)


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


def configs():
    return (
        {
            "latent_dim": 4,
            "projection_dim": 4,
            "mlp_dims": [8],
            "dropout": 0.0,
            "batch_norm": False,
            "lambdas": {"text": 0.1, "image": 0.2},
            "N": 1,
            "lr_disc": 0.002,
            "lr_gen": 0.003,
            "l2": 0.0,
        },
        {
            "id_feature_num": 32,
            "use_mm_features": ["id", "text", "image"],
            "mm_seq_dims": {"text": 3, "image": 2},
            "user_features": ["id"],
            "user_features_dim": {},
        },
    )


BASELINE_PATH = Path(__file__).parents[1] / "baselines" / "canonical_gmmf_formula_cpu_v1.json"


class CanonicalGMMFTests(unittest.TestCase):
    def test_optimizer_groups_are_complete_disjoint_and_configuration_is_guarded(self):
        model_config, data_config = configs()
        model = create_model("gmmf", model_config, data_config)

        groups = model.optimization_parameter_groups()
        grouped_ids = [{id(parameter) for parameter in groups[name]} for name in groups]

        self.assertEqual({"main", "discriminator", "generator"}, set(groups))
        self.assertEqual(
            {id(parameter) for parameter in model.parameters()},
            set().union(*grouped_ids),
        )
        for index, first in enumerate(grouped_ids):
            for second in grouped_ids[index + 1 :]:
                self.assertFalse(first.intersection(second))
        invalid = {**model_config, "lambdas": {"text": 0.1}}
        with self.assertRaisesRegex(ContractError, "missing"):
            create_model("gmmf", invalid, data_config)
        invalid = {**model_config, "N": -1}
        with self.assertRaisesRegex(ContractError, "non-negative"):
            create_model("gmmf", invalid, data_config)

    def test_forward_matches_frozen_formula_without_padding_or_missing_modalities(self):
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        model_config, data_config = configs()
        model_config = {**model_config, "model_name": "gmmf"}
        complete_batch = make_batch()
        complete_batch = Batch(
            user_features=complete_batch.user_features,
            item_features=complete_batch.item_features,
            history_features={
                "id": torch.tensor([[3, 4], [4, 3]]),
                "text": torch.tensor(
                    [
                        [[1.0, 2.0, 3.0], [2.0, 1.0, 3.0]],
                        [[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]],
                    ]
                ),
                "image": torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]),
            },
            history_mask=torch.ones((2, 2), dtype=torch.bool),
            labels=complete_batch.labels,
        )
        torch.manual_seed(int(baseline["seed"]))
        canonical = create_model("gmmf", model_config, data_config)
        canonical.eval()
        canonical_output = canonical(complete_batch)
        expected = baseline["expected"]
        tolerance = baseline["tolerance"]
        self.assertTrue(
            torch.allclose(
                canonical_output.logits,
                torch.tensor(expected["logits"], dtype=canonical_output.logits.dtype),
                atol=tolerance["absolute"],
                rtol=tolerance["relative"],
            )
        )
        self.assertAlmostEqual(
            canonical_output.auxiliary_losses["gmmf_reconstruction"].item(),
            expected["reconstruction_loss"],
            places=7,
        )
        self.assertAlmostEqual(
            canonical.discriminator_loss(complete_batch).item(),
            expected["discriminator_loss"],
            places=7,
        )
        self.assertAlmostEqual(
            canonical.generator_loss(complete_batch).item(),
            expected["generator_loss"],
            places=7,
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in canonical.parameters()),
            expected["parameter_count"],
        )

    def test_all_padding_history_and_adversarial_objectives_are_finite(self):
        model_config, data_config = configs()
        model = create_model("gmmf", model_config, data_config)
        batch = make_batch()
        all_padding = Batch(
            user_features=batch.user_features,
            item_features=batch.item_features,
            history_features={
                name: torch.zeros_like(values) for name, values in batch.history_features.items()
            },
            history_mask=torch.zeros_like(batch.history_mask),
            labels=batch.labels,
        )

        output = model(all_padding)

        self.assertTrue(torch.isfinite(output.logits).all())
        self.assertTrue(
            torch.equal(
                output.representations["history_fusion"],
                torch.zeros_like(output.representations["history_fusion"]),
            )
        )
        self.assertTrue(torch.isfinite(model.discriminator_loss(all_padding)))
        self.assertTrue(torch.isfinite(model.generator_loss(all_padding)))

    def test_alternating_training_starts_at_configured_epoch(self):
        model_config, data_config = configs()
        model = create_model("gmmf", model_config, data_config)
        optimizer = build_phased_adam(
            model.optimization_parameter_groups(),
            {
                "main": 0.001,
                "discriminator": model.discriminator_learning_rate,
                "generator": model.generator_learning_rate,
            },
            model.weight_decay,
        )
        discriminator_before = {
            name: parameter.detach().clone()
            for name, parameter in model.discriminators.named_parameters()
        }
        generator_before = {
            name: parameter.detach().clone()
            for name, parameter in model.generators.named_parameters()
        }
        with tempfile.TemporaryDirectory(dir=".") as directory:
            engine = TrainingEngine(
                model,
                optimizer,
                torch.device("cpu"),
                CheckpointManager(Path(directory) / "checkpoints"),
                alternating_phases=(
                    AlternatingPhase(
                        "discriminator",
                        model.adversarial_start_epoch,
                        model.discriminator_loss,
                    ),
                    AlternatingPhase(
                        "generator", model.adversarial_start_epoch, model.generator_loss
                    ),
                ),
            )
            batch = make_batch()
            engine.train_epoch((batch,), epoch=0)
            self.assertTrue(
                all(
                    torch.equal(discriminator_before[name], parameter)
                    for name, parameter in model.discriminators.named_parameters()
                )
            )
            self.assertTrue(
                all(
                    torch.equal(generator_before[name], parameter)
                    for name, parameter in model.generators.named_parameters()
                )
            )

            engine.train_epoch((batch,), epoch=1)

        self.assertTrue(
            any(
                not torch.equal(discriminator_before[name], parameter)
                for name, parameter in model.discriminators.named_parameters()
            )
        )
        self.assertTrue(
            any(
                not torch.equal(generator_before[name], parameter)
                for name, parameter in model.generators.named_parameters()
            )
        )

    def test_forward_uses_canonical_contract_and_reconstruction_loss(self):
        model_config, data_config = configs()
        model = create_model("gmmf", model_config, data_config)

        self.assertIsInstance(model, BaseSeqModel)
        self.assertEqual(HistoryCapability.SEQUENCE_TOKENS, model.history_capability)
        output = model(make_batch())

        self.assertEqual((2,), tuple(output.logits.shape))
        self.assertEqual({"gmmf_reconstruction"}, set(output.auxiliary_losses))
        objective = output.logits.sum() + output.auxiliary_loss()
        objective.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_registry_uses_canonical_gmmf(self):
        specification = model_spec("gmmf")

        self.assertEqual("mmctr.models.gmmf", specification.module)
        self.assertEqual("gmmf_gan", specification.metadata["alternating_optimization"])


if __name__ == "__main__":
    unittest.main()

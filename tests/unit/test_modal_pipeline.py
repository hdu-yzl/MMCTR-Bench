import unittest
from typing import Any, Dict

import torch

from mmctr.core import ContractError, RegistryError
from mmctr.models.common.components import (
    ComponentConfig,
    ModalPipeline,
    ModalPipelineConfig,
    ModalPipelineSet,
)


class ModalPipelineTests(unittest.TestCase):
    def test_branch_set_parses_target_history_and_user_configs_by_branch_name(self):
        pipelines = ModalPipelineSet.from_mapping(
            {
                "target": {
                    "modalities": ["id", "text"],
                    "input_dimensions": {"id": 2, "text": 3},
                    "projection_dim": 4,
                    "fusion": "sum",
                },
                "history": {
                    "topology": "sequence_fusion",
                    "modalities": ["id", "text"],
                    "input_dimensions": {"id": 2, "text": 3},
                    "projection_dim": 4,
                    "fusion": "mean",
                },
            }
        )

        self.assertEqual(("target", "history"), pipelines.branches)
        self.assertEqual("feature_fusion", pipelines["target"].config.topology)
        self.assertEqual("sequence_fusion", pipelines["history"].config.topology)
        with self.assertRaisesRegex(ContractError, "unknown modal pipeline branch"):
            pipelines["user"]

    def test_invalid_or_incompatible_configs_fail_before_forward(self):
        base = {
            "branch": "history",
            "topology": "sequence_fusion",
            "modalities": ["id", "text"],
            "input_dimensions": {"id": 2, "text": 3},
            "projection_dim": 4,
            "fusion": {"name": "sum"},
        }
        invalid = [
            ({**base, "unknown": True}, "unknown"),
            ({**base, "input_dimensions": {"id": 2}}, "dimensions"),
            ({**base, "pooling": "mean"}, "does not accept pooling"),
            ({**base, "fusion": {"name": "sum", "rank": 2}}, "fusion keys"),
            ({**base, "branch": "target"}, "feature_fusion"),
        ]
        for values, message in invalid:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ContractError, message):
                    ModalPipelineConfig.from_mapping(values)

        unknown_fusion = ModalPipelineConfig.from_mapping(
            {**base, "fusion": {"name": "not_registered"}}
        )
        with self.assertRaisesRegex(RegistryError, "unknown fusion"):
            ModalPipeline(unknown_fusion)

    def test_resolved_mapping_builds_typed_branch_configuration_without_mutation(self):
        raw: Dict[str, Any] = {
            "branch": "history",
            "topology": "pool_then_fuse",
            "modalities": ["id", "image", "text"],
            "input_dimensions": {"id": 2, "image": 3, "text": 4},
            "projection_dim": 6,
            "pooling": {
                "id": {"name": "din", "options": {"hidden_dims": [5]}},
                "image": "mean",
                "text": {"name": "attention"},
            },
            "fusion": {"name": "cat"},
            "output_dim": 7,
        }
        original = {
            **raw,
            "modalities": list(raw["modalities"]),
            "input_dimensions": dict(raw["input_dimensions"]),
            "pooling": dict(raw["pooling"]),
            "fusion": dict(raw["fusion"]),
        }

        config = ModalPipelineConfig.from_mapping(raw)
        pipeline = ModalPipeline(config)

        self.assertEqual(
            "concatenate", pipeline.fusion.__class__.__name__.lower().replace("fusion", "")
        )
        self.assertEqual(("id",), pipeline.target_modalities)
        self.assertEqual(7, pipeline.output_dim)
        self.assertEqual(original, raw)

    def test_pool_then_fuse_projects_pools_fuses_and_adapts(self):
        config = ModalPipelineConfig(
            branch="history",
            topology="pool_then_fuse",
            modalities=("id", "text"),
            input_dimensions={"id": 2, "text": 3},
            projection_dim=4,
            pooling={
                "id": ComponentConfig("mean"),
                "text": ComponentConfig("sum"),
            },
            fusion=ComponentConfig("concatenate"),
            output_dim=5,
        )
        pipeline = ModalPipeline(config)
        values = {
            "id": torch.randn(2, 3, 2, requires_grad=True),
            "text": torch.randn(2, 3, 3, requires_grad=True),
        }
        original = {name: value.detach().clone() for name, value in values.items()}
        sequence_mask = torch.tensor([[True, True, False], [False, False, False]])
        presence = {
            "id": sequence_mask,
            "text": torch.tensor([[True, False, False], [False, False, False]]),
        }

        output = pipeline(values, presence=presence, sequence_mask=sequence_mask)

        self.assertEqual((2, 5), tuple(output.representation.shape))
        self.assertEqual(5, pipeline.output_dim)
        self.assertTrue(torch.equal(output.representation[1], torch.zeros(5)))
        self.assertEqual({}, dict(output.auxiliary_losses))
        for name, value in values.items():
            self.assertTrue(torch.equal(value.detach(), original[name]))
        output.representation.sum().backward()
        self.assertTrue(all(value.grad is not None for value in values.values()))

    def test_pool_then_fuse_passes_same_modality_targets_to_target_aware_pooling(self):
        config = ModalPipelineConfig(
            branch="history",
            topology="pool_then_fuse",
            modalities=("id", "text"),
            input_dimensions={"id": 2, "text": 3},
            projection_dim=4,
            pooling={
                "id": ComponentConfig("din", {"hidden_dims": (5,)}),
                "text": ComponentConfig("cross_attention", {"heads": 2}),
            },
            fusion=ComponentConfig("sum"),
        )
        pipeline = ModalPipeline(config)
        values = {
            "id": torch.randn(2, 3, 2),
            "text": torch.randn(2, 3, 3),
        }
        targets = {
            "id": torch.randn(2, 2),
            "text": torch.randn(2, 3),
        }
        sequence_mask = torch.tensor([[True, True, False], [True, False, False]])

        output = pipeline(values, sequence_mask=sequence_mask, targets=targets)

        self.assertEqual((2, 4), tuple(output.representation.shape))
        with self.assertRaisesRegex(ValueError, "targets"):
            pipeline(values, sequence_mask=sequence_mask)

    def test_fuse_then_pool_uses_fused_dimension_for_sequence_and_target(self):
        config = ModalPipelineConfig(
            branch="history",
            topology="fuse_then_pool",
            modalities=("id", "text"),
            input_dimensions={"id": 2, "text": 3},
            projection_dim=4,
            pooling=ComponentConfig("din", {"hidden_dims": (6,)}),
            fusion=ComponentConfig("concatenate"),
            output_dim=6,
        )
        pipeline = ModalPipeline(config)
        values = {
            "id": torch.randn(2, 3, 2, requires_grad=True),
            "text": torch.randn(2, 3, 3, requires_grad=True),
        }
        targets = {
            "id": torch.randn(2, 2, requires_grad=True),
            "text": torch.randn(2, 3, requires_grad=True),
        }
        sequence_mask = torch.tensor([[True, True, False], [False, False, False]])

        output = pipeline(values, sequence_mask=sequence_mask, targets=targets)

        self.assertEqual((2, 6), tuple(output.representation.shape))
        self.assertEqual(8, pipeline.fusion.output_dim)
        self.assertTrue(torch.equal(output.representation[1], torch.zeros(6)))
        output.representation.sum().backward()
        self.assertTrue(all(value.grad is not None for value in values.values()))
        self.assertTrue(all(value.grad is not None for value in targets.values()))

    def test_sequence_fusion_preserves_tokens_and_masks_adapter_bias(self):
        config = ModalPipelineConfig(
            branch="history",
            topology="sequence_fusion",
            modalities=("image", "text"),
            input_dimensions={"image": 2, "text": 3},
            projection_dim=4,
            pooling=None,
            fusion=ComponentConfig("mean"),
            output_dim=5,
        )
        pipeline = ModalPipeline(config)
        values = {
            "image": torch.randn(2, 3, 2, requires_grad=True),
            "text": torch.randn(2, 3, 3, requires_grad=True),
        }
        sequence_mask = torch.tensor([[True, False, True], [False, False, False]])
        presence = {
            "image": torch.tensor([[True, False, False], [False, False, False]]),
            "text": torch.tensor([[False, False, True], [False, False, False]]),
        }

        output = pipeline(values, presence=presence, sequence_mask=sequence_mask)

        self.assertEqual((2, 3, 5), tuple(output.representation.shape))
        expected_presence = torch.tensor([[True, False, True], [False, False, False]])
        self.assertTrue(torch.equal(output.presence, expected_presence))
        self.assertTrue(torch.equal(output.representation[0, 1], torch.zeros(5)))
        self.assertTrue(torch.equal(output.representation[1], torch.zeros(3, 5)))
        output.representation.sum().backward()
        self.assertTrue(all(value.grad is not None for value in values.values()))

        targets = {
            "image": torch.randn(2, 2),
            "text": torch.randn(2, 3),
        }
        with self.assertRaisesRegex(ContractError, "does not use targets"):
            pipeline(values, sequence_mask=sequence_mask, targets=targets)

    def test_target_and_user_branches_fuse_rank_two_features_without_history_mask(self):
        for branch in ("target", "user"):
            with self.subTest(branch=branch):
                config = ModalPipelineConfig(
                    branch=branch,
                    topology="feature_fusion",
                    modalities=("id", "text"),
                    input_dimensions={"id": 2, "text": 3},
                    projection_dim=4,
                    pooling=None,
                    fusion=ComponentConfig("sum"),
                    output_dim=5,
                )
                pipeline = ModalPipeline(config)
                values = {
                    "id": torch.randn(2, 2, requires_grad=True),
                    "text": torch.randn(2, 3, requires_grad=True),
                }
                presence = {
                    "id": torch.tensor([True, False]),
                    "text": torch.tensor([False, False]),
                }

                output = pipeline(values, presence=presence)

                self.assertEqual((2, 5), tuple(output.representation.shape))
                self.assertTrue(torch.equal(output.presence, torch.tensor([True, False])))
                self.assertTrue(torch.equal(output.representation[1], torch.zeros(5)))


if __name__ == "__main__":
    unittest.main()

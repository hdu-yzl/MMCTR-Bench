import sys
import unittest

import torch

from mmctr.core import ContractError
from mmctr.models.components.fusion import FusionOutput
from mmctr.models.components.fusion_registry import (
    FUSION_REGISTRY,
    create_fusion,
    fusion_capabilities,
)
from mmctr.models.multimodal import _build_fusion


class FusionComponentTests(unittest.TestCase):
    def values(self, rank=2):
        shape = (2, 4) if rank == 2 else (2, 3, 4)
        count = int(torch.tensor(shape).prod().item())
        return {
            "id": torch.arange(1, 1 + count).reshape(shape).float().requires_grad_(),
            "text": torch.ones(shape, requires_grad=True),
            "image": torch.full(shape, 2.0, requires_grad=True),
        }

    def create(self, name):
        kwargs = {}
        if name == "lmf":
            kwargs.update(rank=2, output_dim=5)
        if name == "mtfn":
            kwargs.update(rank=2)
        return create_fusion(name, ("id", "text", "image"), 4, **kwargs)

    def test_registry_is_lazy_and_aliases_are_stable(self):
        before = set(sys.modules)
        self.assertEqual(
            ("concatenate", "lmf", "maf", "mean", "mtfn", "sum"),
            FUSION_REGISTRY.names(),
        )
        self.assertEqual("concatenate", FUSION_REGISTRY.canonical_name("cat"))
        self.assertEqual("sum", FUSION_REGISTRY.canonical_name("add"))
        self.assertEqual("mean", FUSION_REGISTRY.canonical_name("average"))
        imported = set(sys.modules).difference(before)
        self.assertNotIn("mmctr.models.components.fusion", imported)

    def test_all_registered_fusions_support_rank_two_and_three_backward(self):
        expected_dimensions = {
            "concatenate": 12,
            "sum": 4,
            "mean": 4,
            "maf": 4,
            "lmf": 5,
            "mtfn": 4,
        }
        for name, output_dim in expected_dimensions.items():
            for rank in (2, 3):
                with self.subTest(name=name, rank=rank):
                    fusion = self.create(name)
                    values = self.values(rank)
                    result = fusion(values)
                    expected_prefix = (2,) if rank == 2 else (2, 3)
                    self.assertEqual(expected_prefix + (output_dim,), tuple(result.fused.shape))
                    self.assertEqual(output_dim, fusion.output_dim)
                    self.assertEqual({}, dict(result.auxiliary_losses))
                    result.fused.sum().backward()
                    parameters = tuple(fusion.parameters())
                    if parameters:
                        self.assertTrue(any(parameter.grad is not None for parameter in parameters))

    def test_presence_is_explicit_and_masks_maf_bias(self):
        fusion = self.create("maf")
        values = self.values()
        presence = {
            "id": torch.tensor([False, True]),
            "text": torch.tensor([False, False]),
            "image": torch.tensor([False, False]),
        }
        result = fusion(values, presence)
        self.assertTrue(torch.equal(result.fused[0], torch.zeros(4)))
        self.assertTrue(torch.isfinite(result.fused).all())

    def test_reduce_and_concat_have_exact_compatibility_semantics(self):
        values = self.values()
        concatenated = self.create("concatenate")(values).fused
        summed = self.create("sum")(values).fused
        averaged = self.create("mean")(values).fused
        self.assertTrue(
            torch.equal(concatenated, torch.cat([values[name] for name in values], dim=-1))
        )
        self.assertTrue(torch.equal(summed, torch.stack(tuple(values.values())).sum(0)))
        self.assertTrue(torch.equal(averaged, torch.stack(tuple(values.values())).mean(0)))

    def test_invalid_names_dtype_dimension_rank_and_presence_fail_at_boundary(self):
        fusion = self.create("sum")
        values = self.values()
        invalid_cases = [
            ({"id": values["id"], "text": values["text"]}, None, "exactly"),
            ({**values, "text": values["text"].long()}, None, "floating"),
            ({**values, "text": torch.ones(2, 5)}, None, "dimension"),
            ({**values, "text": torch.ones(2, 1, 1, 4)}, None, "rank"),
            (values, {"id": torch.ones(2, dtype=torch.bool)}, "presence"),
            (
                values,
                {name: torch.ones(3, dtype=torch.bool) for name in values},
                "prefix",
            ),
        ]
        for invalid_values, presence, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ContractError, message):
                    fusion(invalid_values, presence)

    def test_fusion_output_requires_named_scalar_losses(self):
        fused = torch.ones(2, 4)
        output = FusionOutput(fused, {"alignment": torch.tensor(0.25)})
        self.assertAlmostEqual(0.25, output.total_auxiliary_loss().item())
        with self.assertRaisesRegex(ContractError, "scalar"):
            FusionOutput(fused, {"alignment": torch.ones(2)})

    def test_registry_metadata_matches_runtime_capability(self):
        for name in FUSION_REGISTRY.names():
            with self.subTest(name=name):
                fusion = self.create(name)
                metadata = fusion_capabilities(name)
                self.assertEqual(metadata["allowed_ranks"], fusion.capability.allowed_ranks)
                self.assertEqual(metadata["output_dim_rule"], fusion.capability.output_dim_rule)
                self.assertEqual(
                    metadata["auxiliary_loss_names"],
                    fusion.capability.auxiliary_loss_names,
                )

    def test_canonical_compatibility_preserves_learned_state_key_layout(self):
        expected = {
            "maf": {"weights.id", "biases.id"},
            "lmf": {"factors.0", "fusion_weights", "fusion_bias"},
            "mtfn": {"heads.id.0.weight", "compress.weight", "compress.bias"},
        }
        for name, required in expected.items():
            with self.subTest(name=name):
                fusion = _build_fusion(name, ("id", "text"), 4, rank=2, output_dim=5)
                self.assertTrue(required.issubset(fusion.state_dict()))
                self.assertFalse(any(key.startswith("fusion.") for key in fusion.state_dict()))


if __name__ == "__main__":
    unittest.main()

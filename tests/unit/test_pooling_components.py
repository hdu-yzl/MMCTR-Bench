import unittest

import torch

from mmctr.core import ContractError
from mmctr.models.components.pooling_registry import (
    POOLING_REGISTRY,
    create_pooling,
    pooling_capabilities,
)


def make_inputs():
    sequence = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            [[9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]],
        ],
        requires_grad=True,
    )
    mask = torch.tensor([[True, False], [False, False]])
    target = torch.ones(2, 4, requires_grad=True)
    return sequence, mask, target


class PoolingRegistryTests(unittest.TestCase):
    def test_registry_names_aliases_and_capabilities(self):
        self.assertEqual(
            ("attention", "cross_attention", "din", "max", "mean", "sum"),
            POOLING_REGISTRY.names(),
        )
        self.assertEqual("mean", POOLING_REGISTRY.canonical_name("average"))
        for name in POOLING_REGISTRY.names():
            metadata = pooling_capabilities(name)
            self.assertEqual(3, metadata["input_rank"])
            self.assertEqual(2, metadata["output_rank"])
            self.assertTrue(metadata["mask_required"])
            self.assertEqual("preserves", metadata["output_dim_rule"])
        self.assertTrue(pooling_capabilities("din")["target_required"])
        self.assertTrue(
            pooling_capabilities("cross_attention")["target_required"]
        )

    def test_registry_constructs_every_pooling_component(self):
        for name in POOLING_REGISTRY.names():
            kwargs = {"dimension": 4}
            if name == "cross_attention":
                kwargs["heads"] = 2
            pooling = create_pooling(name, **kwargs)
            self.assertEqual(4, pooling.input_dim)
            self.assertEqual(4, pooling.output_dim)


class PoolingBehaviorTests(unittest.TestCase):
    def test_reductions_have_explicit_padding_semantics(self):
        sequence, mask, _ = make_inputs()
        mean = create_pooling("mean", 4)(sequence, mask)
        summed = create_pooling("sum", 4)(sequence, mask)
        maximum = create_pooling("max", 4)(sequence, mask)
        expected = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]])
        self.assertTrue(torch.equal(mean, expected))
        self.assertTrue(torch.equal(summed, expected))
        self.assertTrue(torch.equal(maximum, expected))
        (mean.sum() + summed.sum() + maximum.sum()).backward()
        self.assertIsNotNone(sequence.grad)

    def test_learned_pooling_shapes_backward_and_all_padding(self):
        for name in ("attention", "din", "cross_attention"):
            sequence, mask, target = make_inputs()
            kwargs = {"dimension": 4}
            if name == "cross_attention":
                kwargs["heads"] = 2
            pooling = create_pooling(name, **kwargs)
            required_target = target if pooling.capability.target_required else None
            output = pooling(sequence, mask, required_target)
            self.assertEqual((2, 4), tuple(output.shape))
            self.assertTrue(torch.equal(output[1], torch.zeros(4)))
            output.sum().backward()
            self.assertIsNotNone(sequence.grad)
            trainable = [
                parameter for parameter in pooling.parameters() if parameter.requires_grad
            ]
            self.assertTrue(trainable)
            self.assertTrue(any(parameter.grad is not None for parameter in trainable))

    def test_target_aware_and_shape_contracts_fail_early(self):
        sequence, mask, target = make_inputs()
        with self.assertRaises(ContractError):
            create_pooling("din", 4)(sequence, mask)
        with self.assertRaises(ContractError):
            create_pooling("cross_attention", 4, heads=3)
        with self.assertRaises(ContractError):
            create_pooling("mean", 4)(sequence[:, :, :3], mask)
        with self.assertRaises(ContractError):
            create_pooling("din", 4)(sequence, mask, target[:, :3])

    def test_compatibility_component_state_keys_remain_stable(self):
        attention = create_pooling("attention", 4)
        self.assertEqual(
            {"bias", "query", "projection.weight", "projection.bias"},
            set(attention.state_dict()),
        )
        din = create_pooling("din", 4, hidden_dims=(3,))
        self.assertEqual(
            {
                "network.0.weight",
                "network.0.bias",
                "network.1.alpha",
                "network.3.weight",
                "network.3.bias",
            },
            set(din.state_dict()),
        )


if __name__ == "__main__":
    unittest.main()

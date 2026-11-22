import unittest
from copy import deepcopy

import torch

from mmctr.core import ContractError
from mmctr.models.common.components.pooling import _Dice
from mmctr.models.common.components.pooling_registry import (
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
        self.assertTrue(pooling_capabilities("cross_attention")["target_required"])

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
            trainable = [parameter for parameter in pooling.parameters() if parameter.requires_grad]
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

    def test_component_state_retains_parameters_and_adds_dice_statistics(self):
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
                "network.1.running_mean",
                "network.1.running_var",
                "network.1.num_batches_tracked",
                "network.3.weight",
                "network.3.bias",
            },
            set(din.state_dict()),
        )

    def test_dice_training_updates_channel_statistics_and_eval_uses_them(self):
        dice = _Dice(2)
        values = torch.tensor([[1.0, 2.0], [3.0, 6.0]], requires_grad=True)
        trained = dice(values)
        torch.testing.assert_close(dice.running_mean, torch.tensor([0.2, 0.4]))
        torch.testing.assert_close(dice.running_var, torch.tensor([1.0, 1.3]))
        trained.sum().backward()
        self.assertTrue(torch.isfinite(values.grad).all())
        dice.eval()
        before = deepcopy(dice.state_dict())
        with torch.no_grad():
            output = dice(values)
            expected = values * torch.sigmoid(
                (values - dice.running_mean) / torch.sqrt(dice.running_var + dice.epsilon)
            )
            torch.testing.assert_close(output, expected)
            torch.testing.assert_close(output[:1], dice(values[:1]))
            torch.testing.assert_close(output.flip(0), dice(values.flip(0)))
        for name, value in before.items():
            self.assertTrue(torch.equal(value, dice.state_dict()[name]))

    def test_din_eval_is_invariant_to_batch_partition_and_restores_statistics(self):
        torch.manual_seed(11)
        pooling = create_pooling("din", 4, hidden_dims=(6, 3))
        sequence = torch.randn(3, 4, 4)
        target = torch.randn(3, 4)
        mask = torch.tensor(
            [[True, False, False, False], [True, True, True, False], [True, True, True, True]]
        )
        for _ in range(3):
            pooling(sequence, mask, target)
        pooling.eval()
        restored = create_pooling("din", 4, hidden_dims=(6, 3)).eval()
        restored.load_state_dict(pooling.state_dict())
        before = deepcopy(pooling.state_dict())
        with torch.no_grad():
            whole = pooling(sequence, mask, target)
            split = torch.cat(
                [pooling(sequence[i : i + 1], mask[i : i + 1], target[i : i + 1]) for i in range(3)]
            )
            torch.testing.assert_close(whole, split)
            order = torch.tensor([2, 0, 1])
            torch.testing.assert_close(
                whole[order], pooling(sequence[order], mask[order], target[order])
            )
            torch.testing.assert_close(whole, restored(sequence, mask, target))
        for name, value in before.items():
            self.assertTrue(torch.equal(value, pooling.state_dict()[name]))

    def test_din_training_ignores_padding_and_handles_single_token_and_empty_history(self):
        torch.manual_seed(17)
        pooling = create_pooling("din", 4, hidden_dims=(6, 3))
        padded_pooling = deepcopy(pooling)
        sequence = torch.randn(2, 2, 4)
        mask = torch.tensor([[True, True], [True, False]])
        target = torch.randn(2, 4)
        padded_sequence = torch.cat([sequence, torch.full((2, 3, 4), 1000.0)], dim=1)
        padded_mask = torch.cat([mask, torch.zeros(2, 3, dtype=torch.bool)], dim=1)
        torch.testing.assert_close(
            pooling(sequence, mask, target), padded_pooling(padded_sequence, padded_mask, target)
        )
        for name, value in pooling.state_dict().items():
            torch.testing.assert_close(value, padded_pooling.state_dict()[name])

        # Empty histories must not update normalization buffers at any hidden layer.
        before = deepcopy(pooling.state_dict())
        empty = pooling(sequence, torch.zeros_like(mask), target)
        self.assertTrue(torch.equal(empty, torch.zeros_like(empty)))
        for name, value in before.items():
            self.assertTrue(torch.equal(value, pooling.state_dict()[name]))
        single = pooling(sequence[:1, :1], torch.ones(1, 1, dtype=torch.bool), target[:1])
        single.sum().backward()
        self.assertTrue(torch.isfinite(single).all())
        for value in pooling.state_dict().values():
            self.assertTrue(torch.isfinite(value).all())


if __name__ == "__main__":
    unittest.main()

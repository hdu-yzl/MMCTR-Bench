import unittest

import torch

from mmctr.core import ContractError
from mmctr.models.components import (
    DimensionAdapter,
    NamedFeatureProjector,
    apply_feature_mask,
    apply_sequence_mask,
    feature_presence,
    masked_softmax,
)


class DimensionAdapterTests(unittest.TestCase):
    def test_identity_and_linear_adapters_preserve_prefix_dimensions(self):
        identity = DimensionAdapter(3, 3)
        rank_two = torch.randn(2, 3)
        self.assertIs(identity(rank_two), rank_two)

        linear = DimensionAdapter(3, 5)
        rank_three = torch.randn(2, 4, 3, requires_grad=True)
        output = linear(rank_three)
        self.assertEqual((2, 4, 5), tuple(output.shape))
        output.sum().backward()
        self.assertIsNotNone(rank_three.grad)

    def test_adapter_rejects_wrong_dtype_rank_and_dimension(self):
        adapter = DimensionAdapter(3, 2)
        with self.assertRaises(ContractError):
            adapter(torch.ones(2, 3, dtype=torch.long))
        with self.assertRaises(ContractError):
            adapter(torch.ones(3))
        with self.assertRaises(ContractError):
            adapter(torch.ones(2, 4))


class NamedFeatureProjectorTests(unittest.TestCase):
    def test_projects_exact_mapping_without_mutating_inputs(self):
        projector = NamedFeatureProjector({"image": 3, "text": 4}, 2)
        image = torch.randn(2, 3)
        text = torch.randn(2, 5, 4)
        original_image = image.clone()
        original_text = text.clone()
        projected = projector({"image": image, "text": text})
        self.assertEqual((2, 2), tuple(projected["image"].shape))
        self.assertEqual((2, 5, 2), tuple(projected["text"].shape))
        self.assertTrue(torch.equal(image, original_image))
        self.assertTrue(torch.equal(text, original_text))
        sum(value.sum() for value in projected.values()).backward()

    def test_projection_rejects_missing_unknown_and_bad_dimensions(self):
        projector = NamedFeatureProjector({"image": 3}, 2)
        with self.assertRaises(ContractError):
            projector({})
        with self.assertRaises(ContractError):
            projector({"image": torch.ones(2, 3), "text": torch.ones(2, 3)})
        with self.assertRaises(ContractError):
            projector({"image": torch.ones(2, 4)})

    def test_presence_mask_zeros_projection_bias(self):
        projector = NamedFeatureProjector({"image": 3}, 2)
        linear = projector["image"]
        with torch.no_grad():
            linear.weight.zero_()
            linear.bias.fill_(2.0)
        values = torch.zeros(2, 3)
        presence = torch.tensor([False, True])
        output = projector({"image": values}, {"image": presence})["image"]
        self.assertTrue(torch.equal(output[0], torch.zeros(2)))
        self.assertTrue(torch.equal(output[1], torch.full((2,), 2.0)))

    def test_state_keys_match_the_previous_module_dict_layout(self):
        projector = NamedFeatureProjector({"image": 3}, 2)
        self.assertEqual({"image.weight", "image.bias"}, set(projector.state_dict()))
        projector.replace("image", 4)
        self.assertEqual(4, projector.input_dimensions["image"])
        self.assertEqual({"image.weight", "image.bias"}, set(projector.state_dict()))


class MaskingTests(unittest.TestCase):
    def test_sequence_and_feature_masks_are_explicit(self):
        sequence = torch.ones(2, 3, 4)
        sequence_mask = torch.tensor([[True, False, True], [False, False, False]], dtype=torch.bool)
        masked = apply_sequence_mask(sequence, sequence_mask)
        self.assertEqual(8.0, masked.sum().item())
        self.assertEqual(24.0, sequence.sum().item())

        values = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
        presence = feature_presence(values)
        self.assertTrue(torch.equal(presence, torch.tensor([False, True])))
        self.assertTrue(torch.equal(apply_feature_mask(values, presence), values))

    def test_masked_softmax_returns_zero_for_all_padding(self):
        scores = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        mask = torch.tensor([[True, False], [False, False]])
        weights = masked_softmax(scores, mask)
        self.assertTrue(torch.equal(weights[0], torch.tensor([1.0, 0.0])))
        self.assertTrue(torch.equal(weights[1], torch.tensor([0.0, 0.0])))

    def test_masks_reject_wrong_shape_dtype_and_device_contracts(self):
        with self.assertRaises(ContractError):
            apply_sequence_mask(torch.ones(2, 3, 4), torch.ones(2, 3))
        with self.assertRaises(ContractError):
            apply_feature_mask(torch.ones(2, 3), torch.ones(2, 1, dtype=torch.bool))
        with self.assertRaises(ContractError):
            masked_softmax(torch.ones(2, 3), torch.ones(2, 2, dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()

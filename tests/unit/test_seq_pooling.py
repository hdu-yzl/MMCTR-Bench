import unittest

import torch

from models.layers.seq_pooling import get_pooling


class SequencePoolingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.values = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 0.0]],
                [[-1.0, 4.0], [1.0, 2.0]],
            ]
        )

    def test_reduction_pooling_values_and_shapes(self) -> None:
        expected = {
            "mean": torch.tensor([[2.0, 1.0], [0.0, 3.0]]),
            "sum": torch.tensor([[4.0, 2.0], [0.0, 6.0]]),
            "max": torch.tensor([[3.0, 2.0], [1.0, 4.0]]),
        }

        for name, expected_value in expected.items():
            with self.subTest(pooling=name):
                actual = get_pooling(name, dim=1)(self.values)
                self.assertEqual(actual.shape, (2, 2))
                self.assertTrue(torch.equal(actual, expected_value))

    def test_mean_pooling_backward(self) -> None:
        values = self.values.clone().requires_grad_(True)
        get_pooling("mean", dim=1)(values).sum().backward()
        self.assertIsNotNone(values.grad)
        self.assertTrue(torch.isfinite(values.grad).all())

    def test_unknown_pooling_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown pooling type"):
            get_pooling("not_registered")


if __name__ == "__main__":
    unittest.main()

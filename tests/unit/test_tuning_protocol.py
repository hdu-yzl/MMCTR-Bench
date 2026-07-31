import unittest

from utils.tuning_protocol import (
    SELECTION_SPLIT,
    SelectionMetrics,
    evaluate_for_selection,
    is_better,
)


class RecordingModel:
    def __init__(self):
        self.requested_splits = []

    def evalate(self, data_loader, split):
        self.requested_splits.append((data_loader, split))
        return 0.75, 0.25


class TuningProtocolTest(unittest.TestCase):
    def test_selection_evaluation_uses_validation_only(self):
        model = RecordingModel()
        data_loader = object()

        metrics = evaluate_for_selection(model, data_loader)

        self.assertEqual(SELECTION_SPLIT, "val")
        self.assertEqual(model.requested_splits, [(data_loader, "val")])
        self.assertEqual(metrics, SelectionMetrics(auc=0.75, loss=0.25))

    def test_strictly_higher_auc_wins(self):
        incumbent = SelectionMetrics(auc=0.75, loss=0.30)

        self.assertTrue(is_better(SelectionMetrics(auc=0.76, loss=0.40), incumbent))
        self.assertFalse(is_better(SelectionMetrics(auc=0.75, loss=0.20), incumbent))
        self.assertFalse(is_better(SelectionMetrics(auc=0.74, loss=0.10), incumbent))
        self.assertTrue(is_better(incumbent, None))


if __name__ == "__main__":
    unittest.main()

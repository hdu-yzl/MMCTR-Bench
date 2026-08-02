import unittest
from datetime import datetime, timedelta

from mmctr.core import ContractError
from mmctr.data.datasets.antm2c import Interaction, assert_no_future_leakage, build_histories


class AntM2CHistoryTests(unittest.TestCase):
    def interaction(self, event, item, minute, label):
        return Interaction(
            event_id=event,
            user_id="user-1",
            item_index=item,
            timestamp=datetime(2023, 8, 1) + timedelta(minutes=minute),
            label=label,
            split="train",
        )

    def test_single_scan_keeps_duplicate_item_events_without_future_leakage(self):
        records = [
            self.interaction("event-3", 1, 3, 0),
            self.interaction("event-1", 1, 1, 1),
            self.interaction("event-2", 1, 2, 1),
        ]
        histories = build_histories(records, sequence_length=3)
        self.assertEqual((0, 0, 0), histories[0].history_item_indices)
        self.assertEqual((0, 0, 1), histories[1].history_item_indices)
        self.assertEqual((0, 1, 1), histories[2].history_item_indices)
        assert_no_future_leakage(histories)

    def test_duplicate_event_id_is_rejected(self):
        record = self.interaction("same", 1, 1, 1)
        with self.assertRaisesRegex(ContractError, "unique"):
            build_histories([record, record], sequence_length=3)


if __name__ == "__main__":
    unittest.main()

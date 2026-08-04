import csv
import tempfile
import unittest
from pathlib import Path

from mmctr.data.datasets.antm2c import replay_raw_events


class AntM2CRawReplayTests(unittest.TestCase):
    def test_raw_parts_replay_stable_events_splits_items_and_histories(self) -> None:
        rows = [
            ["u1", "item-a", "2023-08-06 08:00:00", "0"],
            ["u1", "item-a", "2023-08-01 08:00:00", "1"],
            ["u1", "item-b", "2023-08-04 08:00:00", "1"],
        ]
        with tempfile.TemporaryDirectory() as directory:
            part = Path(directory) / "antm2c_10m_part0"
            with part.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    [
                        "user_id",
                        "item_id",
                        "log_time",
                        "label",
                        "bill_entity_seq",
                        "service_entity_seq",
                        "query_entity_seq",
                        "item_entity_names",
                        "item_title",
                        "scene",
                    ]
                )
                for row in rows:
                    writer.writerow(row + ["bill", "service", "query", "entity", "title", "2"])

            replay = replay_raw_events([part], sequence_length=2)

        self.assertEqual(("train", "val", "test"), tuple(row.split for row in replay.events))
        self.assertEqual(
            (
                "antm2c_10m_part0:000000000001",
                "antm2c_10m_part0:000000000002",
                "antm2c_10m_part0:000000000000",
            ),
            tuple(row.event_id for row in replay.events),
        )
        self.assertEqual((1, 1, 2), replay.item_index.encode_many(["item-a", "item-a", "item-b"]))
        self.assertEqual((0, 0), replay.histories[0].history_item_indices)
        self.assertEqual((0, 1), replay.histories[1].history_item_indices)
        self.assertEqual((1, 2), replay.histories[2].history_item_indices)


if __name__ == "__main__":
    unittest.main()

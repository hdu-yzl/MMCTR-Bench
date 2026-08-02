import unittest

import numpy as np

from mmctr.data.datasets.antm2c import build_feature_store, build_item_index


class AntM2CItemStoreTests(unittest.TestCase):
    def test_target_and_history_gather_share_one_feature_table(self):
        index = build_item_index(["item-b", "item-a", "item-b"])
        store, audit = build_feature_store(
            index,
            {"title_text": {"item-a": np.array([1.0, 2.0])}},
            {"title_text": 2},
        )
        target = store.gather("title_text", [index.encode("item-a")])
        history = store.gather(
            "title_text", [0, index.encode("item-b"), index.encode("item-a")]
        )
        self.assertTrue(np.array_equal(target[0], history[2]))
        self.assertTrue(np.array_equal(history[0], np.zeros(2)))
        self.assertEqual(("item-b",), audit.missing_by_feature["title_text"])

    def test_first_appearance_mapping_is_stable_and_contiguous(self):
        index = build_item_index([9, 4, 9, 7])
        self.assertEqual((1, 2, 1, 3), index.encode_many([9, 4, 9, 7]))
        self.assertEqual(3, index.item_count)


if __name__ == "__main__":
    unittest.main()

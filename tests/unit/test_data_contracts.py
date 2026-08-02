import unittest

import torch

from mmctr.data import DatasetManifest, HistoryMode, adapt_legacy_loader


class FakeLegacyLoader:
    def _batch(self):
        features = {"id": torch.tensor([[10, 20], [11, 21]], dtype=torch.long)}
        history = {"id": torch.tensor([[20, 0, 0], [21, 20, 0]], dtype=torch.long)}
        labels = torch.tensor([[0.0], [1.0]])
        return features, history, labels

    def get_data(self, split):
        yield self._batch()

    def get_data_seq(self, split):
        features, history, labels = self._batch()
        yield {"id": features["id"][:, 0:1]}, {"id": features["id"][:, 1:2]}, history, labels


class DatasetContractTests(unittest.TestCase):
    def config(self):
        return {
            "version": "fixture-v1",
            "seq_len": 3,
            "padding_id": 0,
            "mm_seq_dims": {"id": 1, "text": 4},
        }

    def test_manifest_fingerprint_is_stable(self):
        first = DatasetManifest.from_config("antm2c", self.config())
        second = DatasetManifest.from_config("antm2c", dict(self.config()))
        self.assertEqual(first.fingerprint, second.fingerprint)

    def test_three_dataset_adapters_emit_canonical_batches(self):
        for dataset_name in ("antm2c", "microlens", "tiktok"):
            with self.subTest(dataset_name=dataset_name):
                loader = adapt_legacy_loader(dataset_name, FakeLegacyLoader(), self.config())
                batch = next(iter(loader.iter_batches("validation")))
                self.assertEqual((2,), tuple(batch.labels.shape))
                self.assertEqual(dataset_name, batch.metadata["dataset"])
                self.assertEqual("val", batch.metadata["split"])
                self.assertEqual((2, 3), tuple(batch.history_mask.shape))

    def test_pooled_compatibility_adapter_splits_combined_ids(self):
        loader = adapt_legacy_loader(
            "antm2c",
            FakeLegacyLoader(),
            self.config(),
            history_mode=HistoryMode.POOLED_COMPAT,
        )
        batch = next(iter(loader.iter_batches("train")))
        self.assertTrue(torch.equal(batch.user_features["id"], torch.tensor([[10], [11]])))
        self.assertTrue(torch.equal(batch.item_features["id"], torch.tensor([[20], [21]])))


if __name__ == "__main__":
    unittest.main()

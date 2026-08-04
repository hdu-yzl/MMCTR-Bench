import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from mmctr.data.datasets.tiktok import TikTokLoader, prepare_tiktok
from mmctr.models.common.registry import create_model


class TikTokDatasetTests(unittest.TestCase):
    def test_official_splits_use_causal_history_and_user_safe_negatives(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            root = Path(directory)
            raw_dir = root / "raw"
            output_dir = root / "processed"
            raw_dir.mkdir()
            sequences = {
                "train": {"0": [0, 1, 2], "1": [2, 3]},
                "val": {"0": [3], "1": [1]},
                "test": {"0": [2], "1": [0]},
            }
            for split, values in sequences.items():
                (raw_dir / "{}.json".format(split)).write_text(json.dumps(values))
            feature_rows = 6
            np.save(
                raw_dir / "text_feat.npy",
                np.arange(feature_rows * 3, dtype=np.float32).reshape(feature_rows, 3),
            )
            np.save(
                raw_dir / "image_feat.npy",
                np.arange(feature_rows * 2, dtype=np.float32).reshape(feature_rows, 2),
            )
            np.save(
                raw_dir / "audio_feat.npy",
                np.arange(feature_rows, dtype=np.float32).reshape(feature_rows, 1),
            )
            for path in raw_dir.glob("*_feat.npy"):
                values = np.load(path)
                values[0] = 0.0
                np.save(path, values)

            manifest = prepare_tiktok(
                raw_dir,
                output_dir,
                sequence_length=3,
                negative_ratio=1,
                seed=11,
            )
            loader = TikTokLoader({"data_dir": str(output_dir)}, batch_size=32)
            batches = {split: next(loader.iter_batches(split)) for split in sequences}

            self.assertEqual(
                {"train": 10, "val": 4, "test": 4},
                {name: stats.samples for name, stats in manifest.splits.items()},
            )
            self.assertEqual(
                "official-split-causal-prefix-negative-v1",
                manifest.metadata["split_protocol"],
            )
            self.assertEqual(3, manifest.id_offsets["item"])

            validation = batches["val"]
            selected = validation.labels.eq(1.0) & validation.item_features["id"].flatten().eq(6)
            self.assertEqual(1, int(selected.sum()))
            self.assertTrue(
                torch.equal(
                    validation.history_features["id"][selected][0],
                    torch.tensor([3, 4, 5]),
                )
            )
            test = batches["test"]
            selected = test.labels.eq(1.0) & test.item_features["id"].flatten().eq(5)
            self.assertTrue(
                torch.equal(test.history_features["id"][selected][0], torch.tensor([4, 5, 6]))
            )
            full_positive_items = {
                1: {3, 4, 5, 6},
                2: {3, 4, 5, 6},
            }
            for batch in batches.values():
                negatives = batch.labels.eq(0.0)
                for user_id, item_id in zip(
                    batch.user_features["id"][negatives].flatten(),
                    batch.item_features["id"][negatives].flatten(),
                ):
                    self.assertNotIn(int(item_id), full_positive_items[int(user_id)])
                self.assertEqual(torch.float32, batch.item_features["text"].dtype)
                self.assertTrue(torch.equal(batch.history_mask, batch.history_features["id"].ne(0)))

            training = batches["train"]
            model = create_model(
                "dnn_mm_seq",
                {
                    "latent_dim": 4,
                    "projection_dim": 4,
                    "mlp_dims": [8],
                    "dropout": 0.0,
                    "batch_norm": False,
                    "modal_fusion_method": "add",
                },
                {
                    "id_feature_num": 8,
                    "use_mm_features": ["id", "text", "image", "audio"],
                    "mm_seq_dims": {"text": 3, "image": 2, "audio": 1},
                    "user_features": ["id"],
                    "user_features_dim": {},
                },
            )
            output = model(training)
            loss = (
                torch.nn.functional.binary_cross_entropy_with_logits(output.logits, training.labels)
                + output.auxiliary_loss()
            )
            loss.backward()
            self.assertTrue(torch.isfinite(output.logits).all())
            self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()

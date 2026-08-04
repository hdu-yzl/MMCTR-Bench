import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mmctr.data.datasets.microlens import MicroLensLoader, prepare_microlens
from mmctr.models.common.registry import create_model


class MicroLensDatasetTests(unittest.TestCase):
    def test_prepare_and_load_preserves_ids_modalities_and_split_manifest(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            root = Path(directory)
            raw_dir = root / "raw"
            output_dir = root / "processed"
            raw_dir.mkdir()
            item_features = pd.DataFrame(
                {
                    "item_id": [1, 2, 3],
                    "txt_emb_BERT": [
                        np.array([1.0, 1.5], dtype=np.float32),
                        np.array([2.0, 2.5], dtype=np.float32),
                        np.array([3.0, 3.5], dtype=np.float32),
                    ],
                    "img_emb_CLIPRN50": [
                        np.array([1.0, 0.0, 0.5], dtype=np.float32),
                        np.array([0.0, 1.0, 0.5], dtype=np.float32),
                        np.array([0.5, 0.0, 1.0], dtype=np.float32),
                    ],
                }
            )
            item_features.to_parquet(raw_dir / "item_feature.parquet", index=False)
            interactions = pd.DataFrame(
                {
                    "user_id": np.arange(1, 11),
                    "item_seq": [
                        [0, 1, 2],
                        [0, 2, 3],
                        [0, 3, 1],
                        [1, 2, 3],
                        [2, 3, 1],
                        [3, 1, 2],
                        [1, 3, 2],
                        [2, 1, 3],
                        [3, 2, 1],
                        [1, 2, 1],
                    ],
                    "item_id": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
                    "label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                }
            )
            interactions.to_parquet(raw_dir / "train.parquet", index=False)

            manifest = prepare_microlens(raw_dir, output_dir, sequence_length=2, seed=7)
            loader = MicroLensLoader({"data_dir": str(output_dir)}, batch_size=4)
            batches = {
                split: list(loader.iter_batches(split)) for split in ("train", "val", "test")
            }

            self.assertEqual(
                {"train": 8, "val": 1, "test": 1},
                {name: stats.samples for name, stats in manifest.splits.items()},
            )
            self.assertEqual(
                {"train": 8, "val": 1, "test": 1},
                {name: stats.users for name, stats in manifest.splits.items()},
            )
            self.assertTrue(all(stats.items for stats in manifest.splits.values()))
            self.assertEqual(
                10, sum(batch.batch_size for values in batches.values() for batch in values)
            )
            self.assertEqual("named-npy-v1", manifest.storage_format)
            saved_manifest = json.loads((output_dir / "manifest.json").read_text())
            self.assertEqual(manifest.fingerprint, saved_manifest["fingerprint"])

            expected_text = {
                1_000_001: torch.tensor([1.0, 1.5]),
                1_000_002: torch.tensor([2.0, 2.5]),
                1_000_003: torch.tensor([3.0, 3.5]),
            }
            for split, split_batches in batches.items():
                for batch in split_batches:
                    self.assertEqual(split, batch.metadata["split"])
                    self.assertEqual(torch.long, batch.item_features["id"].dtype)
                    self.assertEqual(torch.bool, batch.history_mask.dtype)
                    self.assertEqual(torch.float32, batch.item_features["text"].dtype)
                    for item_id, text in zip(
                        batch.item_features["id"].flatten(), batch.item_features["text"]
                    ):
                        self.assertTrue(torch.equal(expected_text[int(item_id)], text))
                    self.assertTrue(
                        torch.equal(batch.history_mask, batch.history_features["id"].ne(0))
                    )

            training_batch = batches["train"][0]
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
                    "id_feature_num": 1_000_004,
                    "use_mm_features": ["id", "text", "image"],
                    "mm_seq_dims": {"text": 2, "image": 3},
                    "user_features": ["id"],
                    "user_features_dim": {},
                },
            )
            output = model(training_batch)
            loss = (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    output.logits, training_batch.labels
                )
                + output.auxiliary_loss()
            )
            loss.backward()
            self.assertTrue(torch.isfinite(output.logits).all())
            self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()

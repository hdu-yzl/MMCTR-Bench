import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

from mmctr.core import ContractError
from mmctr.data.datasets.antm2c.benchmark import LegacyAntM2CLayout
from mmctr.data.datasets.antm2c.canonical import (
    convert_legacy_to_canonical,
    verify_canonical_dataset,
)
from mmctr.data.registry import create_data_loader


class AntM2CCanonicalTests(unittest.TestCase):
    @staticmethod
    def _example(
        packed_text: np.ndarray,
        image: np.ndarray,
        user_id: int,
        item_id: int,
        domain: float,
    ) -> bytes:
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "text_features": tf.train.Feature(
                        float_list=tf.train.FloatList(value=packed_text.reshape(-1))
                    ),
                    "image_features": tf.train.Feature(float_list=tf.train.FloatList(value=image)),
                    "id_feature": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[user_id, item_id])
                    ),
                    "label": tf.train.Feature(float_list=tf.train.FloatList(value=[1.0])),
                    "domain": tf.train.Feature(float_list=tf.train.FloatList(value=[domain])),
                    "user_seq": tf.train.Feature(int64_list=tf.train.Int64List(value=[0, 11])),
                }
            )
        )
        return example.SerializeToString()

    def test_conversion_resumes_and_publishes_a_sharded_canonical_loader(self):
        layout = LegacyAntM2CLayout(
            text_dimension=2,
            image_dimension=2,
            sequence_length=2,
            minimum_item_id=10,
        )
        text_store = np.asarray([[0.0, 0.0], [9.0, 10.0], [13.0, 14.0]], dtype=np.float32)
        image_store = np.asarray([[0.0, 0.0], [15.0, 16.0], [17.0, 18.0]], dtype=np.float32)
        packed = np.asarray(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
                [9.0, 10.0],
                [11.0, 12.0],
            ],
            dtype=np.float32,
        )
        mismatched = packed.copy()
        mismatched[4] = [99.0, 100.0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            np.save(source / "text_feature.npy", text_store)
            np.save(source / "image_feature.npy", image_store)
            for split_index, name in enumerate(
                ("train_shuffle_0.tfrecord", "val0.tfrecord", "test0.tfrecord")
            ):
                with tf.io.TFRecordWriter(str(source / name)) as writer:
                    writer.write(
                        self._example(
                            mismatched,
                            image_store[1],
                            user_id=5 + split_index,
                            item_id=10,
                            domain=float(split_index),
                        )
                    )
                    writer.write(
                        self._example(
                            packed,
                            image_store[1],
                            user_id=5 + split_index,
                            item_id=10,
                            domain=float(split_index),
                        )
                    )

            output = root / "canonical-v1"
            paused = convert_legacy_to_canonical(
                source,
                output,
                shard_size=2,
                read_batch_size=2,
                layout=layout,
                max_source_files=1,
            )
            self.assertFalse(paused.complete)
            self.assertFalse(output.exists())
            self.assertTrue(output.with_name("canonical-v1.incomplete").is_dir())

            completed = convert_legacy_to_canonical(
                source,
                output,
                shard_size=2,
                read_batch_size=2,
                layout=layout,
            )
            self.assertTrue(completed.complete)
            self.assertEqual(
                {"train": 1, "val": 1, "test": 1},
                dict(completed.manifest.metadata["title_store_conflicts"]),
            )
            self.assertEqual(2, completed.manifest.splits["train"].samples)
            loader = create_data_loader("antm2c", {"data_dir": str(output)}, batch_size=2)
            batches = {
                split: next(loader.iter_batches(split)) for split in ("train", "val", "test")
            }
            verified = verify_canonical_dataset(output)
            self.assertEqual(completed.manifest.fingerprint, verified.fingerprint)
            item_path = output / "items" / "features" / "text.npy"
            layout_path = output / "layout.json"
            original_item = item_path.read_bytes()
            original_layout = layout_path.read_bytes()
            item_values = np.load(item_path, mmap_mode="r+")
            item_values[1, 0] += 1.0
            item_values.flush()
            del item_values
            layout_payload = json.loads(layout_path.read_text(encoding="utf-8"))
            layout_payload["item_features"]["text"]["sha256"] = hashlib.sha256(
                item_path.read_bytes()
            ).hexdigest()
            layout_path.write_text(json.dumps(layout_payload), encoding="utf-8")
            try:
                with self.assertRaisesRegex(ContractError, "item feature hash"):
                    verify_canonical_dataset(output)
            finally:
                item_path.write_bytes(original_item)
                layout_path.write_bytes(original_layout)
            labels_path = output / "splits" / "train" / "shard-000000" / "labels.npy"
            labels = np.load(labels_path, mmap_mode="r+")
            labels[0] = 0.0
            labels.flush()
            with self.assertRaisesRegex(ContractError, "hash"):
                verify_canonical_dataset(output)

        self.assertEqual([5, 5], batches["train"].user_features["id"].reshape(-1).tolist())
        self.assertEqual([10, 10], batches["train"].item_features["id"].reshape(-1).tolist())
        self.assertEqual([[0, 11], [0, 11]], batches["train"].history_features["id"].tolist())
        self.assertTrue(
            np.array_equal(
                batches["train"].item_features["text"].numpy(),
                np.repeat(text_store[1:2], 2, axis=0),
            )
        )
        self.assertEqual((2, 2), tuple(batches["train"].metadata["event_id"].shape))
        self.assertEqual([2.0, 2.0], batches["test"].metadata["domain"].tolist())


if __name__ == "__main__":
    unittest.main()

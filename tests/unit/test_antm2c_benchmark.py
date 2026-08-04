import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

from mmctr.data.datasets.antm2c.benchmark import (
    LegacyAntM2CLayout,
    benchmark_prepared_slice,
    iter_legacy_tfrecord_batches,
    prepare_benchmark_slice,
)
from mmctr.data.datasets.antm2c.array_store import load_array_store


class AntM2CBenchmarkTests(unittest.TestCase):
    def test_legacy_reader_emits_slice_free_canonical_batch(self):
        layout = LegacyAntM2CLayout(
            text_dimension=2,
            image_dimension=2,
            sequence_length=2,
            minimum_item_id=10,
        )
        packed_text = np.asarray(
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
        text_store = np.asarray([[0.0, 0.0], [9.0, 10.0], [13.0, 14.0]], dtype=np.float32)
        image_store = np.asarray([[0.0, 0.0], [15.0, 16.0], [17.0, 18.0]], dtype=np.float32)
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "text_features": tf.train.Feature(
                        float_list=tf.train.FloatList(value=packed_text.reshape(-1))
                    ),
                    "image_features": tf.train.Feature(
                        float_list=tf.train.FloatList(value=image_store[1])
                    ),
                    "id_feature": tf.train.Feature(int64_list=tf.train.Int64List(value=[5, 10])),
                    "label": tf.train.Feature(float_list=tf.train.FloatList(value=[1.0])),
                    "domain": tf.train.Feature(float_list=tf.train.FloatList(value=[3.0])),
                    "user_seq": tf.train.Feature(int64_list=tf.train.Int64List(value=[0, 11])),
                }
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record_path = root / "sample.tfrecord"
            with tf.io.TFRecordWriter(str(record_path)) as writer:
                writer.write(example.SerializeToString())
            np.save(root / "text_feature.npy", text_store)
            np.save(root / "image_feature.npy", image_store)

            batch = next(
                iter_legacy_tfrecord_batches(
                    [record_path],
                    root,
                    batch_size=1,
                    layout=layout,
                )
            )

        self.assertEqual([5], batch.user_features["id"].reshape(-1).tolist())
        self.assertEqual([10], batch.item_features["id"].reshape(-1).tolist())
        self.assertEqual([[0, 11]], batch.history_features["id"].tolist())
        self.assertEqual(
            [
                "bill_text",
                "domain",
                "entity_text",
                "query_text",
                "service_text",
                "text",
                "time_context",
            ],
            sorted(batch.context_features),
        )
        self.assertTrue(np.array_equal(batch.item_features["text"].numpy(), text_store[1:2]))
        self.assertTrue(
            np.array_equal(
                batch.history_features["image"].numpy(),
                image_store[[0, 2]][None, :, :],
            )
        )
        self.assertEqual([3.0], batch.metadata["domain"].tolist())

    def test_prepared_slice_compares_identical_events_and_writes_report(self):
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
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "text_features": tf.train.Feature(
                        float_list=tf.train.FloatList(value=packed.reshape(-1))
                    ),
                    "image_features": tf.train.Feature(
                        float_list=tf.train.FloatList(value=image_store[1])
                    ),
                    "id_feature": tf.train.Feature(int64_list=tf.train.Int64List(value=[5, 10])),
                    "label": tf.train.Feature(float_list=tf.train.FloatList(value=[1.0])),
                    "domain": tf.train.Feature(float_list=tf.train.FloatList(value=[3.0])),
                    "user_seq": tf.train.Feature(int64_list=tf.train.Int64List(value=[0, 11])),
                }
            )
        )
        mismatched_packed = packed.copy()
        mismatched_packed[4] = [99.0, 100.0]
        mismatched = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "text_features": tf.train.Feature(
                        float_list=tf.train.FloatList(value=mismatched_packed.reshape(-1))
                    ),
                    "image_features": tf.train.Feature(
                        float_list=tf.train.FloatList(value=image_store[1])
                    ),
                    "id_feature": tf.train.Feature(int64_list=tf.train.Int64List(value=[5, 10])),
                    "label": tf.train.Feature(float_list=tf.train.FloatList(value=[1.0])),
                    "domain": tf.train.Feature(float_list=tf.train.FloatList(value=[3.0])),
                    "user_seq": tf.train.Feature(int64_list=tf.train.Int64List(value=[0, 11])),
                }
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            np.save(source / "text_feature.npy", text_store)
            np.save(source / "image_feature.npy", image_store)
            for name in ("train_shuffle_0.tfrecord", "val0.tfrecord", "test0.tfrecord"):
                with tf.io.TFRecordWriter(str(source / name)) as writer:
                    writer.write(mismatched.SerializeToString())
                    writer.write(example.SerializeToString())

            output = root / "benchmark-v1"
            prepared = prepare_benchmark_slice(
                source,
                output,
                samples_per_split=1,
                layout=layout,
            )
            repeated = prepare_benchmark_slice(
                source,
                output,
                samples_per_split=1,
                layout=layout,
            )
            _, splits, _ = load_array_store(output / "named_arrays")
            report = benchmark_prepared_slice(output, batch_size=1, repeats=1)

            self.assertEqual(prepared, repeated)
            self.assertEqual({"train": 1, "val": 1, "test": 1}, prepared.split_samples)
            self.assertEqual({"train": 2, "val": 2, "test": 2}, prepared.scanned_samples)
            self.assertEqual({"train": 1, "val": 1, "test": 1}, prepared.skipped_title_mismatches)
            self.assertEqual(1, splits["train"].sample_count)
            self.assertEqual(3, report.legacy.samples)
            self.assertEqual(3, report.named_arrays.samples)
            self.assertAlmostEqual(report.legacy.checksum, report.named_arrays.checksum, places=6)
            self.assertGreater(report.legacy_interaction_bytes, 0)
            self.assertGreater(report.named_interaction_bytes, 0)
            self.assertGreater(report.shared_item_bytes, 0)
            self.assertIsNone(report.gpu_wait_ms)
            self.assertTrue((output / "benchmark_report.json").is_file())


if __name__ == "__main__":
    unittest.main()

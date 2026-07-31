import importlib.util
import tempfile
import unittest
from pathlib import Path


TENSORFLOW_AVAILABLE = importlib.util.find_spec("tensorflow") is not None


@unittest.skipUnless(TENSORFLOW_AVAILABLE, "TensorFlow is an optional data dependency")
class TensorFlowTFRecordSmokeTest(unittest.TestCase):
    def test_tfrecord_round_trip_preserves_benchmark_fields(self) -> None:
        import tensorflow as tf

        temporary_root = Path(__file__).resolve().parents[2] / ".tmp" / "tests"
        temporary_root.mkdir(parents=True, exist_ok=True)

        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "id_feature": tf.train.Feature(int64_list=tf.train.Int64List(value=[7, 11])),
                    "history": tf.train.Feature(int64_list=tf.train.Int64List(value=[2, 3, 5])),
                    "label": tf.train.Feature(float_list=tf.train.FloatList(value=[1.0])),
                }
            )
        )

        with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            record_path = Path(directory) / "sample.tfrecord"
            with tf.io.TFRecordWriter(str(record_path)) as writer:
                writer.write(example.SerializeToString())

            dataset = tf.data.TFRecordDataset([str(record_path)], num_parallel_reads=1)
            options = tf.data.Options()
            options.threading.private_threadpool_size = 1
            dataset = dataset.with_options(options)
            serialized = next(iter(dataset))
            parsed = tf.io.parse_single_example(
                serialized,
                {
                    "id_feature": tf.io.FixedLenFeature([2], tf.int64),
                    "history": tf.io.FixedLenFeature([3], tf.int64),
                    "label": tf.io.FixedLenFeature([1], tf.float32),
                },
            )

        self.assertEqual(parsed["id_feature"].numpy().tolist(), [7, 11])
        self.assertEqual(parsed["history"].numpy().tolist(), [2, 3, 5])
        self.assertEqual(parsed["label"].numpy().tolist(), [1.0])

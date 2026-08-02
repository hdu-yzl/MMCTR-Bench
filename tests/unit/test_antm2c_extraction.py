import tempfile
import unittest
from pathlib import Path

import numpy as np

from mmctr.data.datasets.antm2c.extraction import (
    ExtractionInput,
    iter_extracted_features,
    run_batch_extraction,
)


class RecordingEncoder:
    def __init__(self, calls):
        self.calls = calls

    def encode(self, values):
        self.calls.append(tuple(values))
        return np.asarray([[len(value), 1.0] for value in values], dtype=np.float32)


class AntM2CExtractionTests(unittest.TestCase):
    def test_batch_extraction_loads_once_and_resumes(self):
        records = [
            ExtractionInput("a", "one"),
            ExtractionInput("b", ""),
            ExtractionInput("c", "three"),
        ]
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            manifest = run_batch_extraction(
                records,
                Path(directory),
                "service_text",
                2,
                "source-v1",
                lambda: RecordingEncoder(calls),
                batch_size=2,
            )
            self.assertEqual(3, manifest.completed)
            self.assertEqual(("b",), manifest.missing_keys)
            self.assertEqual(2, len(calls))
            resumed = run_batch_extraction(
                records,
                Path(directory),
                "service_text",
                2,
                "source-v1",
                lambda: self.fail("completed extraction must not reload the encoder"),
                batch_size=2,
            )
            self.assertEqual(manifest, resumed)
            shards = list(iter_extracted_features(Path(directory)))
            self.assertEqual(("a", "b"), shards[0][0])
            self.assertTrue(np.array_equal(shards[0][1][1], np.zeros(2)))


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

import torch

from mmctr.analysis import (
    ModalityDropout,
    TransformedDataLoader,
    load_robustness_study_config,
    load_robustness_study_matrix,
    save_robustness_study_matrix,
)
from mmctr.core import Batch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def make_batch():
    return Batch(
        user_features={"id": torch.tensor([[1], [2], [3], [4]])},
        item_features={
            "id": torch.tensor([[5], [6], [7], [8]]),
            "text": torch.ones(4, 3),
            "image": torch.ones(4, 2),
        },
        history_features={
            "id": torch.tensor([[5, 0], [6, 5], [7, 6], [8, 7]]),
            "text": torch.ones(4, 2, 3),
            "image": torch.ones(4, 2, 2),
        },
        history_mask=torch.tensor([[True, False], [True, True], [True, True], [True, True]]),
        labels=torch.tensor([1.0, 0.0, 1.0, 0.0]),
        context_features={"query_text": torch.ones(4, 3)},
        metadata={"split": "test", "batch_index": 3},
    )


class RobustnessProtocolTests(unittest.TestCase):
    def test_study_config_builds_versioned_experiment_tasks(self):
        config = """\
dataset: fixture
data_fingerprint: data-v1
data: {name: fixture}
model_configs:
  dnn_mm: {latent_dim: 4}
models: [dnn_mm]
modalities: [text, image]
probabilities: [0.0, 0.5]
seeds: [3, 5]
splits: [train, val, test]
"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "robustness.yaml"
            matrix_path = root / "matrix.json"
            config_path.write_text(config, encoding="utf-8")

            tasks = load_robustness_study_config(config_path)
            save_robustness_study_matrix(tasks, matrix_path)
            restored = load_robustness_study_matrix(matrix_path)

            self.assertEqual([task.key for task in tasks], [task.key for task in restored])
            self.assertEqual(4, len(tasks))
            analysis = tasks[-1].resolved_config["analysis"]
            self.assertEqual("modality-dropout-v1", analysis["protocol"])
            self.assertEqual(("text", "image"), analysis["modalities"])
            self.assertEqual(("train", "val", "test"), analysis["splits"])
            payload = json.loads(matrix_path.read_text(encoding="utf-8"))
            self.assertEqual("robustness-study-matrix-v1", payload["schema"])

    def test_legacy_robustness_training_entry_is_removed(self):
        self.assertFalse((REPOSITORY_ROOT / "src/analysis/modal_robustness.py").exists())

    def test_loader_applies_the_protocol_only_to_declared_splits(self):
        batch = make_batch()

        class Loader:
            dataset_name = "fixture"
            manifest = object()

            def iter_batches(self, _split):
                yield batch

        wrapper = TransformedDataLoader(
            Loader(),
            ModalityDropout(("text",), probability=1.0, seed=9),
            splits=("train",),
        )

        train = next(wrapper.iter_batches("train"))
        validation = next(wrapper.iter_batches("val"))

        self.assertEqual(0, torch.count_nonzero(train.item_features["text"]))
        self.assertTrue(torch.equal(validation.item_features["text"], batch.item_features["text"]))

    def test_seeded_dropout_is_non_mutating_and_consistent_across_target_history(self):
        batch = make_batch()
        transform = ModalityDropout(("text", "image"), probability=0.5, seed=17)
        first = transform(batch)
        repeated = transform(batch)
        second = ModalityDropout(("text", "image"), probability=0.5, seed=17)(batch)

        self.assertTrue(torch.equal(first.item_features["text"], repeated.item_features["text"]))
        self.assertTrue(torch.equal(first.item_features["text"], second.item_features["text"]))
        self.assertTrue(
            torch.equal(first.history_features["image"], second.history_features["image"])
        )
        for modality in ("text", "image"):
            target_missing = first.item_features[modality].eq(0).all(dim=-1)
            history_values = first.history_features[modality]
            history_missing = history_values.eq(0).flatten(start_dim=1).all(dim=1)
            self.assertTrue(torch.equal(target_missing, history_missing))
        self.assertTrue(torch.equal(batch.item_features["text"], torch.ones(4, 3)))
        self.assertTrue(torch.equal(batch.history_features["image"], torch.ones(4, 2, 2)))
        self.assertEqual("test", first.metadata["split"])
        self.assertEqual(17, first.metadata["robustness"]["seed"])

        all_missing = ModalityDropout(("text",), probability=1.0, seed=1)(batch)
        self.assertEqual(0, torch.count_nonzero(all_missing.item_features["text"]))
        self.assertEqual(0, torch.count_nonzero(all_missing.history_features["text"]))


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

import torch

from mmctr.core import Batch, ModelOutput
from mmctr.data import DatasetManifest
from mmctr.training import CheckpointManager, TrainingEngine, build_optimizer


class TinyLoader:
    dataset_name = "fixture"
    manifest = DatasetManifest(
        name="fixture",
        version="v1",
        storage_format="memory",
        sequence_length=2,
        padding_id=0,
        feature_dimensions={"id": 1},
    )

    def iter_batches(self, split):
        labels = torch.tensor([0.0, 1.0])
        yield Batch(
            user_features={"id": torch.tensor([[1], [2]])},
            item_features={"score": torch.tensor([[0.0], [1.0]])},
            history_features={"id": torch.tensor([[0, 0], [1, 0]])},
            history_mask=torch.tensor([[False, False], [True, False]]),
            labels=labels,
            metadata={"split": split},
        )


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, batch):
        return ModelOutput(self.linear(batch.item_features["score"]))


class TrainingEngineTests(unittest.TestCase):
    def test_train_save_load_and_resume(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            model = TinyModel()
            optimizer = build_optimizer(model, "adam", 0.01)
            checkpoints = CheckpointManager(Path(directory) / "checkpoints")
            engine = TrainingEngine(model, optimizer, torch.device("cpu"), checkpoints)
            result = engine.fit(TinyLoader(), 2, 1, "fixture-run", Path(directory))
            self.assertTrue(result.succeeded)
            self.assertTrue(checkpoints.best_path.is_file())
            self.assertTrue(checkpoints.last_path.is_file())
            next_epoch = engine.resume()
            self.assertGreaterEqual(next_epoch, 1)
            resumed = engine.fit(
                TinyLoader(),
                next_epoch + 1,
                1,
                "fixture-run",
                Path(directory),
                start_epoch=next_epoch,
            )
            self.assertTrue(resumed.succeeded)

    def test_test_split_is_not_read_during_fit(self):
        class GuardedLoader(TinyLoader):
            def iter_batches(self, split):
                if split == "test":
                    raise AssertionError("fit must not read test")
                return super().iter_batches(split)

        with tempfile.TemporaryDirectory(dir=".") as directory:
            model = TinyModel()
            optimizer = build_optimizer(model, "sgd", 0.01)
            engine = TrainingEngine(
                model,
                optimizer,
                torch.device("cpu"),
                CheckpointManager(Path(directory) / "checkpoints"),
            )
            engine.fit(GuardedLoader(), 1, 1, "fixture-run", Path(directory))


if __name__ == "__main__":
    unittest.main()

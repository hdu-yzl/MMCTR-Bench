"""Exercise tuning through real training/evaluation with a forbidden trial test split."""

import json
import logging

import pytest
import torch

from mmctr.core import Batch, ModelOutput
from mmctr.training import CheckpointManager, TrainingEngine
from mmctr.training.entrypoint import Trainer
from mmctr.utils.run_context import create_run_context


class SearchTrainer(Trainer):
    def __init__(self, root, candidate, calls):
        self.model = torch.nn.Linear(1, 1)
        self.model_config = dict(candidate)
        self.data_config = {"fixture": {"fingerprint": "fixture-sha"}}
        self.train_config = {"max_epochs": 2, "early_stop_patience": 1, "seed": 2025}
        self.run_context = create_run_context(root, "search", "fixture", "tiny", {})
        self.logger = logging.getLogger(__name__)
        model = self.model

        class Loader:
            def iter_batches(self, split):
                calls.append(split)
                if split == "test":
                    assert (root / "study" / "frozen_selection.json").is_file()
                yield Batch(
                    user_features={"id": torch.tensor([[1], [2]])},
                    item_features={"value": torch.tensor([[0.0], [1.0]])},
                    history_features={"id": torch.zeros(2, 1, dtype=torch.long)},
                    history_mask=torch.zeros(2, 1, dtype=torch.bool),
                    labels=torch.tensor([0.0, 1.0]),
                )

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = model

            def forward(self, batch):
                return ModelOutput(self.head(batch.item_features["value"]))

        self.model = Model()
        self.dataloader = Loader()
        self.engine = TrainingEngine(
            self.model,
            torch.optim.Adam(self.model.parameters(), lr=candidate["lr"]),
            torch.device("cpu"),
            CheckpointManager(self.run_context.checkpoints_dir, persist=False),
        )


def test_search_freezes_all_validation_trials_before_one_final_test(tmp_path):
    from mmctr.experiments.tuning import ValidationOnlyTuner, run_validation_search

    calls = []
    tuner = ValidationOnlyTuner(tmp_path, "study", "tiny-search", "fixture-sha", [2025])
    result = run_validation_search(
        tuner,
        [{"lr": 0.01}, {"lr": 0.001}],
        lambda candidate: SearchTrainer(tmp_path, candidate, calls),
    )
    assert calls.count("test") == 1
    assert calls[-1] == "test"
    assert set(result.metrics) == {"test_auc", "test_loss"}
    history = json.loads((tuner.study_dir / "trial_history.json").read_text())
    assert len(history["trials"]) == 2
    assert "test_auc" not in json.dumps(history)
    assert (tuner.study_dir / "final_test_result.json").is_file()
    with pytest.raises(ValueError, match="already"):
        run_validation_search(tuner, [{"lr": 0.1}], lambda _: None)

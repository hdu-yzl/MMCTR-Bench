import numpy as np
import pytest
import torch

from mmctr.analysis.benchmark_protocols import (
    ColdSubsetLoader,
    LegacyRobustnessLoader,
    cold_selection,
    compare_auc,
    population_summary,
)
from mmctr.analysis.legacy_alignment import LegacyAlignmentObjective
from mmctr.core import Batch, ContractError


def batch():
    return Batch(
        user_features={"id": torch.arange(1, 17).reshape(-1, 1)},
        item_features={
            "id": torch.arange(17, 33).reshape(-1, 1),
            "text": torch.ones(16, 4),
            "image": torch.ones(16, 4) * 2,
        },
        history_features={
            "id": torch.ones(16, 5, dtype=torch.long),
            "text": torch.ones(16, 5, 4),
            "image": torch.ones(16, 5, 4) * 2,
        },
        history_mask=torch.ones(16, 5, dtype=torch.bool),
        labels=torch.arange(16).remainder(2).float(),
    )


class Source:
    def __init__(self):
        self.batch = batch()

    def iter_batches(self, split):
        yield self.batch


def test_comparator_sign_units_and_unevaluable():
    result = compare_auc(0.78, 0.8)
    assert result["auc_delta"] == pytest.approx(-0.02)
    assert result["auc_absolute_difference"] == pytest.approx(0.02)
    assert result["relative_change_percent"] == pytest.approx(-2.5)
    assert result["direction"] == "decrease"
    assert compare_auc(None, 0.8)["auc_delta"] is None
    with pytest.raises(ContractError):
        compare_auc(float("nan"), 0.8)


def test_legacy_cold_boundary_and_overlap():
    u = np.array([0, 1, 4, 5, 8])
    i = np.array([6, 0, 5, 4, 0])
    assert cold_selection(u, i, "union", "zero").tolist() == [True, True, False, False, True]
    assert cold_selection(u, i, "user", "low_lt5").tolist() == [True, True, True, False, False]
    assert population_summary(np.array([]))["reason"] == "empty"
    assert population_summary(np.array([1, 1]))["reason"] == "single_class"


def test_cold_loader_preserves_training_and_filters_evaluation():
    source = Source()
    mask = np.arange(16) < 6
    loader = ColdSubsetLoader(source, {"val": mask, "test": mask})
    assert next(loader.iter_batches("train")) is source.batch
    test = next(loader.iter_batches("test"))
    assert test.batch_size == 6
    assert torch.equal(test.labels, source.batch.labels[:6])
    assert torch.equal(test.history_mask, source.batch.history_mask[:6])


def test_robustness_legacy_mask_clean_validation_and_nonmutation():
    source = Source()
    loader = LegacyRobustnessLoader(source, 0.5)
    assert next(loader.iter_batches("val")) is source.batch
    a = next(loader.iter_batches("test"))
    b = next(loader.iter_batches("test"))
    expected = torch.tensor(np.random.RandomState(42 * 1_000_003).random(16) < 0.5)
    assert torch.equal(a.item_features["text"].eq(0).all(-1), expected)
    assert torch.equal(a.item_features["text"], b.item_features["text"])
    assert torch.equal(a.history_features["image"].eq(0).all(-1).all(-1), expected)
    assert torch.equal(a.item_features["id"], source.batch.item_features["id"])
    assert torch.equal(a.history_mask, source.batch.history_mask)
    assert source.batch.item_features["text"].eq(1).all()
    next(loader.iter_batches("train"))
    epoch1 = next(loader.iter_batches("train"))
    expected1 = torch.tensor(np.random.RandomState(42 * 1_000_003 + 100_003).random(16) < 0.5)
    assert torch.equal(epoch1.item_features["text"].eq(0).all(-1), expected1)


@pytest.mark.parametrize("method", ["kl", "contrastive", "cosine", "mmd"])
def test_alignment_gradients_and_zero_weight(method):
    torch.manual_seed(19)
    features = {name: torch.randn(8, 6, requires_grad=True) for name in ("id", "text", "image")}
    objective = LegacyAlignmentObjective(method, list(features), 6, 0.3)
    loss = objective(features)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0
        for value in features.values()
    )
    assert LegacyAlignmentObjective(method, list(features), 6, 0)(features).item() == 0


def test_adversarial_separate_parameter_updates():
    torch.manual_seed(23)
    features = {name: torch.randn(8, 6, requires_grad=True) for name in ("id", "text")}
    module = LegacyAlignmentObjective("adversarial", list(features), 6, 0.3)
    optimizer = torch.optim.Adam(module.parameters(), lr=0.001)
    before = [p.detach().clone() for p in module.parameters()]
    module(features).backward()
    assert features["text"].grad.abs().sum() > 0
    assert all(p.grad is None for p in module.parameters())
    for value in features.values():
        value.grad = None
    module.discriminator_loss(features).backward()
    assert all(value.grad is None for value in features.values())
    optimizer.step()
    assert any(not torch.equal(old, new) for old, new in zip(before, module.parameters()))


def test_mmd_degenerate_batch_is_finite():
    values = {
        "id": torch.zeros(1, 4, requires_grad=True),
        "text": torch.zeros(1, 4, requires_grad=True),
    }
    loss = LegacyAlignmentObjective("mmd", list(values), 4, 0.3)(values)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(values["id"].grad).all()


@pytest.mark.parametrize("native_phased", [False, True])
def test_alignment_engine_updates_discriminator_and_preserves_native_phase(tmp_path, native_phased):
    from types import SimpleNamespace
    from mmctr.analysis.alignment_training import attach_alignment
    from mmctr.core import ModelOutput
    from mmctr.training.checkpointing import CheckpointManager
    from mmctr.training.engine import AlternatingPhase, TrainingEngine
    from mmctr.training.optimizers import PhasedAdam

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.id_projection = torch.nn.Linear(4, 6)
            self.content_projection = torch.nn.Linear(4, 6)
            self.head = torch.nn.Linear(6, 1)
            self.native = torch.nn.Parameter(torch.tensor(2.0))

        def forward(self, value):
            logits = self.head(self.content_projection(value.item_features["text"]))
            return ModelOutput(logits, auxiliary_losses={"native": logits.square().mean() * 0.01})

    def extract(model, value):
        return {
            "id": model.id_projection(value.item_features["image"]),
            "text": model.content_projection(value.item_features["text"]),
        }

    model = Backbone()
    parameters = [p for name, p in model.named_parameters() if name != "native"]
    phases = ()
    if native_phased:
        optimizer = PhasedAdam(
            {"main": parameters, "native": [model.native]},
            {"main": 0.001, "native": 0.002},
            weight_decay=1e-7,
        )
        phases = (AlternatingPhase("native", 0, lambda value: model.native.square()),)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-7)
    engine = TrainingEngine(
        model,
        optimizer,
        torch.device("cpu"),
        CheckpointManager(tmp_path, persist=False),
        alternating_phases=phases,
    )
    trainer = SimpleNamespace(model=model, engine=engine)
    objective = LegacyAlignmentObjective("adversarial", ["id", "text"], 6, 0.3)
    wrapper = attach_alignment(trainer, objective, extract)
    before = [p.detach().clone() for p in objective.parameters()]
    content_before = model.content_projection.weight.detach().clone()
    engine.train_epoch(Source().iter_batches("train"), 0)
    assert any(not torch.equal(a, b) for a, b in zip(before, objective.parameters()))
    assert not torch.equal(content_before, model.content_projection.weight)
    assert trainer.model is engine.model is wrapper
    assert engine.optimizer.param_groups[-1]["weight_decay"] == 1e-7
    if native_phased:
        assert model.native.item() < 2
        assert engine.optimizer.param_groups[1]["lr"] == 0.002
        assert engine.alternating_phases[0] is phases[0]
    wrapper.eval()
    assert "diagnostic_alignment" not in wrapper(batch()).auxiliary_losses
    assert not list(tmp_path.rglob("*.pt"))


def test_cold_populations_count_candidates_and_ignore_evaluation_support(tmp_path):
    from mmctr.analysis.cold_populations import build_cold_populations

    arrays = {
        "train": ([1, 1, 1, 1, 1, 2], [11, 11, 11, 11, 11, 12], [0, 0, 0, 0, 1, 1]),
        "val": ([1, 2, 3, 3], [11, 11, 12, 13], [0, 1, 0, 1]),
        "test": ([1, 2, 3, 3], [11, 11, 12, 13], [1, 0, 1, 0]),
    }
    for split, values in arrays.items():
        path = tmp_path / "splits" / split
        path.mkdir(parents=True)
        for field, value in zip(("user_ids", "item_ids", "labels"), values):
            np.save(path / (field + ".npy"), np.array(value, dtype=np.int64))
        np.save(path / "history_item_ids.npy", np.zeros((len(values[0]), 2), dtype=np.int64))
    masks, summary = build_cold_populations(tmp_path, "union", "zero")
    assert masks["test"].tolist() == [False, False, True, True]
    assert summary["val"]["auc_evaluable"]
    masks, _ = build_cold_populations(tmp_path, "user", "low_lt5")
    assert masks["test"].tolist() == [False, True, True, True]
    masks, summary = build_cold_populations(tmp_path, "item", "zero")
    assert masks["test"].tolist() == [False, False, False, True]
    assert summary["test"]["reason"] == "single_class"

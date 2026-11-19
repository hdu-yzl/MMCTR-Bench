"""Verify the common search and both limits without a real GPU or long waits."""

import json
from types import SimpleNamespace

import pytest
import torch

from mmctr.core import ContractError, RunResult
from mmctr.experiments.search_protocol import SHARED_SEARCH_SPACE
from mmctr.experiments.tuning import (
    ValidationOnlyTuner,
    load_tuning_search,
    run_validation_search,
)
from mmctr.utils.run_context import create_run_context


def test_shared_search_replays_for_models_and_records_its_fixed_space(tmp_path):
    pytest.importorskip("optuna")
    candidates, protocol = load_tuning_search()
    fixed = tmp_path / "fixed.yaml"
    fixed.write_text("model: {similarity_source: legacy_projected}\n")
    projected, projected_protocol = load_tuning_search(fixed)
    assert len(candidates) == 50
    assert protocol["seed"] == 2025
    assert protocol["mode"] == "shared_random"
    assert protocol["space"] == {key: list(value) for key, value in SHARED_SEARCH_SPACE.items()}
    for original, variant in zip(candidates, projected):
        assert original["training"] == variant["training"]
        assert original["model"]["dropout"] == variant["model"]["dropout"]
        assert variant["model"]["similarity_source"] == "legacy_projected"
        for key, choices in SHARED_SEARCH_SPACE.items():
            section, name = key.split(".")
            assert original[section][name] in choices
    assert projected_protocol["fixed_model_overrides"] == {"similarity_source": "legacy_projected"}
    fixed.write_text("model: {dropout: 0.9}\n")
    with pytest.raises(ValueError, match="searched dropout"):
        load_tuning_search(fixed)


@pytest.mark.parametrize(
    "model_name",
    ["dcn", "lmf", "diff_msin", "marn", "dmf", "simcen", "make", "em3", "gmmf", "qarm"],
)
def test_model_spaces_materialize_and_keep_paired_common_draws(model_name, tmp_path):
    pytest.importorskip("optuna")
    shared, _ = load_tuning_search()
    candidates, protocol = load_tuning_search(model_name=model_name)
    replay, replay_protocol = load_tuning_search(model_name=model_name.upper())
    assert candidates == replay
    assert protocol == replay_protocol
    assert len(candidates) == 50
    assert protocol["model_name"] == model_name
    for candidate, common in zip(candidates, shared):
        assert candidate["training"] == common["training"]
        assert candidate["model"]["dropout"] == common["model"]["dropout"]
        for key, choices in protocol["space"].items():
            section, name = key.split(".")
            assert candidate[section][name] in choices
    fixed = tmp_path / "collision.yaml"
    name = next(key.split(".")[1] for key in protocol["space"] if key not in SHARED_SEARCH_SPACE)
    fixed.write_text("model: {" + name + ": 99}\n")
    with pytest.raises(ValueError, match="searched model parameters"):
        load_tuning_search(fixed, model_name=model_name)
    fixed.write_text("model: {tiktok: {dropout: 0.9}}\n")
    with pytest.raises(ValueError, match="searched model parameters"):
        load_tuning_search(fixed, model_name=model_name)


def test_structured_choices_are_independent_and_exclusions_are_explicit():
    pytest.importorskip("optuna")
    simcen, protocol = load_tuning_search(model_name="simcen")
    assert protocol["excluded_choices"]["model.hidden_unit"]["values"] == [
        [512, 512, 512],
        [128, 128, 128],
    ]
    assert all(candidate["model"]["hidden_unit"] == [480, 480, 480] for candidate in simcen)
    candidates, protocol = load_tuning_search(model_name="gmmf")
    assert {candidate["model"]["lambdas"]["text"] for candidate in candidates} == {0.1, 0.3}
    candidates[0]["model"]["lambdas"]["text"] = 999
    assert all(candidate["model"]["lambdas"]["text"] != 999 for candidate in candidates[1:])
    assert protocol["space"]["model.lambdas"][0]["text"] == 0.1
    psrq, protocol = load_tuning_search(model_name="mcca")
    assert protocol["model_name"] == "psrq"
    assert "codebook_size" in protocol["pretraining_only_space"]
    assert all(set(candidate["model"]) == {"dropout"} for candidate in psrq)


class TimedTrainer:
    """Simulate work between loader boundaries, leaving all tensors on the CPU."""

    def __init__(self, tuner, candidate, clock, calls, device):
        self.candidate, self.clock, self.calls = candidate, clock, calls
        self.model_config = dict(candidate)
        self.train_config = {"seed": 2025}
        self.data_config = {"fixture": {"fingerprint": "fixture-sha"}}
        self.model = torch.nn.Linear(1, 1)
        self.model.weight.data.fill_(candidate["id"])
        self.engine = SimpleNamespace(device=torch.device(device))
        self.run_context = create_run_context(
            tuner.study_dir / "runs", "search", "fixture", "tiny", {}
        )
        self.tuner = tuner

        class Loader:
            def iter_batches(self, split):
                assert split != "test"
                yield None

        self.dataloader = Loader()

    def run(self, evaluate_test):
        assert evaluate_test is False
        self.calls.append(self.candidate["id"])
        for _ in self.dataloader.iter_batches("train"):
            self.clock[0] += self.candidate.get("hours", 0) * 3600
            if self.candidate.get("fail"):
                raise RuntimeError("synthetic training failure")
        self.run_context.finalize("completed")
        return RunResult(
            self.run_context.run_id,
            "completed",
            {"val_auc": 0.5 + self.candidate["id"] / 1000, "val_log_loss": 0.7, "best_epoch": 0},
        )

    def evaluate_final(self, epoch):
        assert self.tuner.study_dir.joinpath("frozen_selection.json").exists()
        assert self.model.weight.item() == self.candidate["id"]
        self.calls.append("test")
        return RunResult(self.run_context.run_id, "completed", {"test_auc": 0.6})


def timed_search(tmp_path, monkeypatch, device="cuda"):
    from mmctr.experiments import search_protocol

    tuner = ValidationOnlyTuner(tmp_path, "study", "fixture-search", "fixture-sha", [2025])
    clock, calls = [0.0], []
    monkeypatch.setattr(search_protocol.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr("gc.collect", lambda: 0)

    def factory(candidate):
        return TimedTrainer(tuner, candidate, clock, calls, device)

    return tuner, calls, factory


def test_fifty_attempt_cap_counts_failures_and_keeps_one_frozen_test(tmp_path, monkeypatch):
    tuner, calls, factory = timed_search(tmp_path, monkeypatch)
    candidates = [{"id": i, "fail": i == 49} for i in range(55)]
    run_validation_search(tuner, candidates, factory)
    history = json.loads((tuner.study_dir / "trial_history.json").read_text())
    selection = json.loads((tuner.study_dir / "frozen_selection.json").read_text())
    assert calls == list(range(50)) + ["test"]
    assert len(history["trials"]) == 50
    assert history["trials"][-1]["status"] == "failed"
    assert history["budget"]["attempted_trials"] == 50
    assert history["stop_reason"] == "max_trials"
    assert selection["selected_trial_id"] == "48"
    assert selection["budget"] == history["budget"]
    assert "test_auc" not in json.dumps(history)


def test_gpu_budget_interrupts_inflight_trial_then_tests_previous_winner(tmp_path, monkeypatch):
    tuner, calls, factory = timed_search(tmp_path, monkeypatch)
    run_validation_search(
        tuner, [{"id": 0, "hours": 49}, {"id": 1, "hours": 2}, {"id": 2}], factory
    )
    history = json.loads((tuner.study_dir / "trial_history.json").read_text())
    selected = json.loads((tuner.study_dir / "frozen_selection.json").read_text())
    assert calls == [0, 1, "test"]
    assert selected["selected_trial_id"] == "0"
    assert history["stop_reason"] == "gpu_hours"
    assert history["budget"]["gpu_hours"] == 51
    assert history["trials"][1]["status"] == "failed"
    assert "TuningBudgetExceeded" in history["trials"][1]["error"]


def test_exhausted_first_trial_leaves_history_without_test_or_selection(tmp_path, monkeypatch):
    tuner, calls, factory = timed_search(tmp_path, monkeypatch)
    with pytest.raises(ContractError, match="no completed"):
        run_validation_search(tuner, [{"id": 0, "hours": 50}, {"id": 1}], factory)
    history = json.loads((tuner.study_dir / "trial_history.json").read_text())
    assert calls == [0]
    assert history["stop_reason"] == "gpu_hours"
    assert history["budget"]["gpu_hours"] == 50
    assert not (tuner.study_dir / "frozen_selection.json").exists()
    assert not (tuner.study_dir / "final_test_result.json").exists()
    with pytest.raises(ValueError, match="prior attempts"):
        run_validation_search(tuner, [{"id": 0}], factory)


def test_cpu_wall_time_does_not_claim_gpu_hours(tmp_path, monkeypatch):
    tuner, calls, factory = timed_search(tmp_path, monkeypatch, "cpu")
    run_validation_search(tuner, [{"id": 0, "hours": 100}, {"id": 1, "hours": 100}], factory)
    history = json.loads((tuner.study_dir / "trial_history.json").read_text())
    assert calls == [0, 1, "test"]
    assert history["budget"]["gpu_hours"] == 0
    assert history["budget"]["cpu_hours"] == 200
    assert history["stop_reason"] == "candidates_exhausted"


def test_factory_failure_still_consumes_allocated_gpu_budget(tmp_path, monkeypatch):
    from mmctr.experiments import search_protocol

    tuner = ValidationOnlyTuner(tmp_path, "study", "fixture-search", "fixture-sha", [2025])
    clock, attempts = [0.0], []
    monkeypatch.setattr(search_protocol.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr("gc.collect", lambda: 0)

    def factory(candidate):
        attempts.append(candidate)
        clock[0] += 50 * 3600
        raise RuntimeError("initialization failed")

    with pytest.raises(ContractError, match="no completed"):
        run_validation_search(tuner, [{"id": 0}, {"id": 1}], factory, device_type="cuda")
    history = json.loads((tuner.study_dir / "trial_history.json").read_text())
    assert attempts == [{"id": 0}]
    assert history["budget"]["gpu_hours"] == 50
    assert history["stop_reason"] == "gpu_hours"
    assert history["trials"][0]["status"] == "failed"


def test_invalid_provenance_stops_study_even_with_an_earlier_valid_winner(tmp_path, monkeypatch):
    tuner, calls, factory = timed_search(tmp_path, monkeypatch)

    def wrong_data(candidate):
        trainer = factory(candidate)
        if candidate["id"] == 1:
            trainer.data_config["fixture"]["fingerprint"] = "different-data"
        return trainer

    with pytest.raises(ContractError, match="fingerprint"):
        run_validation_search(tuner, [{"id": 0}, {"id": 1}, {"id": 2}], wrong_data)
    history = json.loads((tuner.study_dir / "trial_history.json").read_text())
    assert calls == [0]
    assert history["stop_reason"] == "invalid_trial"
    assert len(history["trials"]) == 2
    assert not (tuner.study_dir / "frozen_selection.json").exists()

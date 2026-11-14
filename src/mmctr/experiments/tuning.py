"""Validation-only model selection with an explicit frozen final-test boundary."""

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Sequence, Tuple

from mmctr.core import ContractError, RunResult

from .runner import ExperimentTask


TUNING_SCHEMA_VERSION = 1


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(name): _freeze(item) for name, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(name): _plain(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _atomic_json(path: Path, values: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(values, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass(frozen=True)
class TuningTrial:
    """One recorded run and the exact candidate configuration it evaluated."""

    trial_id: str
    config: Mapping[str, Any]
    result: RunResult

    def __post_init__(self) -> None:
        if not isinstance(self.trial_id, str) or not self.trial_id:
            raise ContractError("trial_id must be a non-empty string")
        if not isinstance(self.config, Mapping):
            raise ContractError("trial config must be a mapping")
        if not isinstance(self.result, RunResult):
            raise ContractError("trial result must be a RunResult")
        object.__setattr__(self, "config", _freeze(self.config))


@dataclass(frozen=True)
class FrozenSelection:
    """Immutable provenance required before constructing the one final-test task."""

    study_id: str
    experiment_id: str
    trial_id: str
    run_id: str
    config: Mapping[str, Any]
    validation_auc: float
    validation_log_loss: float
    seeds: Tuple[int, ...]
    data_fingerprint: str
    path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "config", _freeze(self.config))
        object.__setattr__(self, "seeds", tuple(int(seed) for seed in self.seeds))
        object.__setattr__(self, "path", Path(self.path).resolve())


class ValidationOnlyTuner:
    """Select by validation AUC and keep final test evaluation out of trial history."""

    def __init__(
        self,
        output_root: Path,
        study_id: str,
        experiment_id: str,
        data_fingerprint: str,
        seeds: Sequence[int],
    ) -> None:
        for name, value in (
            ("study_id", study_id),
            ("experiment_id", experiment_id),
            ("data_fingerprint", data_fingerprint),
        ):
            if not isinstance(value, str) or not value:
                raise ContractError("{} must be a non-empty string".format(name))
        normalised_seeds = tuple(int(seed) for seed in seeds)
        if Path(study_id).name != study_id or study_id in {".", ".."}:
            raise ContractError("study_id must be a safe path component")
        if not normalised_seeds or len(set(normalised_seeds)) != len(normalised_seeds):
            raise ContractError("tuning seeds must be non-empty and unique")
        self.output_root = Path(output_root).expanduser().resolve()
        self.study_id = study_id
        self.experiment_id = experiment_id
        self.data_fingerprint = data_fingerprint
        self.seeds = normalised_seeds
        self.study_dir = self.output_root / self.study_id

    @staticmethod
    def _validate_trial_metrics(trials: Sequence[TuningTrial]) -> None:
        for trial in trials:
            contaminated = [
                name for name in trial.result.metrics if name.lower().startswith("test")
            ]
            if contaminated:
                raise ContractError(
                    "test metrics are forbidden during tuning selection: {}".format(contaminated)
                )

    def freeze(self, trials: Sequence[TuningTrial]) -> FrozenSelection:
        """Write all trials and atomically freeze the best validation-only candidate.

        Failed trials remain in history but cannot win. Selection uses strictly greater
        validation AUC, so ties retain the first candidate and never consult test data.
        """

        if (self.study_dir / "frozen_selection.json").exists():
            raise ContractError("selection is already frozen; use a new study ID")
        trial_tuple = tuple(trials)
        if not trial_tuple or any(not isinstance(trial, TuningTrial) for trial in trial_tuple):
            raise ContractError("tuning study requires at least one TuningTrial")
        if len({trial.trial_id for trial in trial_tuple}) != len(trial_tuple):
            raise ContractError("tuning trial IDs must be unique")
        self._validate_trial_metrics(trial_tuple)
        candidates = []
        history = []
        for trial in trial_tuple:
            history.append(
                {
                    "trial_id": trial.trial_id,
                    "run_id": trial.result.run_id,
                    "status": trial.result.status,
                    "config": _plain(trial.config),
                    "metrics": dict(trial.result.metrics),
                    "error": trial.result.error,
                }
            )
            if trial.result.status != "completed":
                continue
            try:
                validation_auc = float(trial.result.metrics["val_auc"])
            except KeyError as error:
                raise ContractError("completed tuning trial is missing val_auc") from error
            try:
                validation_loss = float(trial.result.metrics["val_log_loss"])
            except KeyError as error:
                raise ContractError("completed tuning trial is missing val_log_loss") from error
            if not math.isfinite(validation_auc) or not math.isfinite(validation_loss):
                raise ContractError("validation metrics must be finite")
            candidates.append((trial, validation_auc, validation_loss))
        if not candidates:
            raise ContractError("tuning study has no completed validation trial")
        selected, validation_auc, validation_loss = candidates[0]
        for candidate, candidate_auc, candidate_loss in candidates[1:]:
            if candidate_auc > validation_auc:
                selected, validation_auc, validation_loss = (
                    candidate,
                    candidate_auc,
                    candidate_loss,
                )
        history_payload = {
            "schema_version": TUNING_SCHEMA_VERSION,
            "study_id": self.study_id,
            "experiment_id": self.experiment_id,
            "data_fingerprint": self.data_fingerprint,
            "seeds": list(self.seeds),
            "selection_split": "val",
            "objective": "val_auc",
            "trials": history,
        }
        _atomic_json(self.study_dir / "trial_history.json", history_payload)
        selection_payload = {
            "schema_version": TUNING_SCHEMA_VERSION,
            "study_id": self.study_id,
            "experiment_id": self.experiment_id,
            "data_fingerprint": self.data_fingerprint,
            "seeds": list(self.seeds),
            "selection_split": "val",
            "objective": "val_auc",
            "selected_trial_id": selected.trial_id,
            "selected_run_id": selected.result.run_id,
            "validation_auc": validation_auc,
            "validation_log_loss": validation_loss,
            "frozen_config": _plain(selected.config),
        }
        selection_path = self.study_dir / "frozen_selection.json"
        _atomic_json(selection_path, selection_payload)
        return FrozenSelection(
            study_id=self.study_id,
            experiment_id=self.experiment_id,
            trial_id=selected.trial_id,
            run_id=selected.result.run_id,
            config=selected.config,
            validation_auc=validation_auc,
            validation_log_loss=validation_loss,
            seeds=self.seeds,
            data_fingerprint=self.data_fingerprint,
            path=selection_path,
        )

    def _require_selection(self, selection: FrozenSelection) -> None:
        if (
            not isinstance(selection, FrozenSelection)
            or selection.study_id != self.study_id
            or selection.experiment_id != self.experiment_id
            or selection.data_fingerprint != self.data_fingerprint
            or selection.seeds != self.seeds
            or selection.path != (self.study_dir / "frozen_selection.json").resolve()
            or not selection.path.is_file()
        ):
            raise ContractError("final test requires this tuner's frozen selection")
        payload = json.loads(selection.path.read_text(encoding="utf-8"))
        expected = {
            "study_id": selection.study_id,
            "experiment_id": selection.experiment_id,
            "data_fingerprint": selection.data_fingerprint,
            "seeds": list(selection.seeds),
            "selected_trial_id": selection.trial_id,
            "selected_run_id": selection.run_id,
            "frozen_config": _plain(selection.config),
            "validation_auc": selection.validation_auc,
            "validation_log_loss": selection.validation_log_loss,
        }
        if any(payload.get(name) != value for name, value in expected.items()):
            raise ContractError("persisted frozen selection differs from selection provenance")

    def final_test_task(
        self,
        selection: FrozenSelection,
        task_id: str,
        dataset: str,
        model: str,
        seed: int,
    ) -> ExperimentTask:
        """Construct the final-test task only after a persisted selection exists."""

        self._require_selection(selection)
        config: Dict[str, Any] = dict(_plain(selection.config))
        config.update(
            {
                "stage": "final_test",
                "data_fingerprint": selection.data_fingerprint,
                "selection": {
                    "study_id": selection.study_id,
                    "experiment_id": selection.experiment_id,
                    "trial_id": selection.trial_id,
                    "run_id": selection.run_id,
                    "validation_auc": selection.validation_auc,
                    "seeds": list(selection.seeds),
                },
            }
        )
        return ExperimentTask(task_id, dataset, model, seed, config)

    def record_final_test_result(self, selection: FrozenSelection, result: RunResult) -> Path:
        """Persist test metrics separately so they cannot feed the selection artifact."""

        self._require_selection(selection)
        if not isinstance(result, RunResult) or result.status != "completed":
            raise ContractError("final test result must be a completed RunResult")
        if not result.metrics or any(
            not name.lower().startswith("test_") for name in result.metrics
        ):
            raise ContractError("final test result may contain only test_* metrics")
        path = self.study_dir / "final_test_result.json"
        if path.exists():
            raise ContractError("final test result already exists")
        _atomic_json(
            path,
            {
                "schema_version": TUNING_SCHEMA_VERSION,
                "study_id": self.study_id,
                "selected_trial_id": selection.trial_id,
                "selected_run_id": selection.run_id,
                "run_id": result.run_id,
                "data_fingerprint": selection.data_fingerprint,
                "metrics": dict(result.metrics),
            },
        )
        return path


def run_validation_search(tuner, candidates, trainer_factory):
    """Execute candidates without test access, freeze a winner, then test its weights once.

    ``trainer_factory(candidate)`` constructs the normal Trainer. Only the best model
    state is retained on CPU; the final Trainer loads it without retraining. A study
    lock and immutable selection prevent accidental reruns under the same identity.
    """
    import fcntl
    import gc
    import torch

    tuner.study_dir.mkdir(parents=True, exist_ok=True)
    with (tuner.study_dir / ".search.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (tuner.study_dir / "frozen_selection.json").exists():
            raise ValueError("study is already frozen; use a new study ID")
        trials = []
        best_auc = float("-inf")
        best_state = None
        best_epoch = 0
        for index, candidate in enumerate(candidates):
            trainer = trainer_factory(candidate)
            if tuple([trainer.train_config["seed"]]) != tuner.seeds:
                raise ContractError("executable search requires its declared single tuning seed")
            fingerprints = [value.get("fingerprint") for value in trainer.data_config.values()]
            if fingerprints != [tuner.data_fingerprint]:
                raise ContractError("trial data fingerprint differs from the tuning study")
            result = trainer.run(evaluate_test=False)
            resolved = {
                "candidate": _plain(candidate),
                "model": _plain(trainer.model_config),
                "training": _plain(trainer.train_config),
                "data": _plain(trainer.data_config),
            }
            trials.append(TuningTrial(str(index), resolved, result))
            if result.metrics["val_auc"] > best_auc:
                best_auc = result.metrics["val_auc"]
                best_epoch = int(result.metrics["best_epoch"])
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in trainer.model.state_dict().items()
                }
            del trainer
            gc.collect()
        selection = tuner.freeze(trials)
        final = trainer_factory(_plain(selection.config["candidate"]))
        try:
            runtime = {"checkpoint_dir", "log_dir", "output_root", "cuda"}
            frozen_train = {
                k: v for k, v in selection.config["training"].items() if k not in runtime
            }
            final_train = {k: v for k, v in final.train_config.items() if k not in runtime}
            if (
                _plain(final.model_config) != _plain(selection.config["model"])
                or _plain(final.data_config) != _plain(selection.config["data"])
                or _plain(final_train) != _plain(frozen_train)
            ):
                raise ContractError("final configuration differs from frozen selection")
            if best_state is None:
                raise ContractError("no selected model weights")
            final.model.load_state_dict(best_state)
            final.run_context.update_metadata(
                {
                    "stage": "final_test",
                    "selection_path": str(selection.path),
                    "selected_run_id": selection.run_id,
                }
            )
            with torch.no_grad():
                result = final.evaluate_final(best_epoch)
            tuner.record_final_test_result(selection, result)
            final.run_context.finalize("completed", summary=dict(result.metrics))
            return result
        except BaseException as error:
            final.run_context.finalize("failed", error=repr(error))
            raise


__all__ = [
    "FrozenSelection",
    "TUNING_SCHEMA_VERSION",
    "TuningTrial",
    "ValidationOnlyTuner",
    "run_validation_search",
]

"""Shared categorical random search and allocated single-GPU tuning budget."""

import math
import time
from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional


MAX_TRIALS = 50
MAX_GPU_HOURS = 50
SEARCH_SEED = 2025
MODEL_SEARCH_SEED = 2026
SHARED_SEARCH_SPACE = {
    "training.lr": (1e-3, 1e-4, 1e-5),
    "training.l2": (1e-5, 1e-6, 1e-7),
    "model.dropout": (0.0, 0.1, 0.2, 0.3, 0.5),
}


def shared_candidates(
    model_overrides: Optional[Mapping[str, Any]] = None,
    model_space: Optional[Mapping[str, Any]] = None,
    max_trials: int = MAX_TRIALS,
) -> List[Dict[str, Any]]:
    """Draw the requested trials, independently of model and training RNGs."""
    TuningBudget(max_trials=max_trials)
    try:
        import optuna
    except ImportError as error:
        raise ValueError(
            "shared random search requires the optuna extra: install .[optuna]"
        ) from error

    fixed = dict(model_overrides or {})
    if set(fixed) & {"model_name", "dropout"}:
        raise ValueError("fixed model overrides cannot replace model_name or searched dropout")
    extra = dict(model_space or {})
    searched = {"dropout", *extra}
    if set(fixed) & set(extra) or any(
        isinstance(value, Mapping) and set(value) & searched for value in fixed.values()
    ):
        raise ValueError("fixed model overrides cannot replace searched model parameters")
    sampler = optuna.samplers.RandomSampler(seed=SEARCH_SEED)
    # The sampler needs no objective feedback for independent random draws.
    study = optuna.create_study(direction="maximize", sampler=sampler)
    candidates = []
    for _ in range(max_trials):
        trial = study.ask()
        values = {
            key: trial.suggest_categorical(key, choices)
            for key, choices in SHARED_SEARCH_SPACE.items()
        }
        candidates.append(
            {
                "training": {"lr": values["training.lr"], "l2": values["training.l2"]},
                "model": dict(deepcopy(fixed), dropout=values["model.dropout"]),
            }
        )
        study.tell(trial, 0.0)
    if extra:
        # Keep the shared draws paired across models with different extra dimensions.
        model_study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.RandomSampler(seed=MODEL_SEARCH_SEED)
        )
        for candidate in candidates:
            trial = model_study.ask()
            for name, choices in extra.items():
                # Optuna categorical choices must be primitives. Decode structured
                # values from stable indices and persist their actual values.
                if any(isinstance(value, (list, dict)) for value in choices):
                    index = trial.suggest_categorical("model." + name, list(range(len(choices))))
                    value = deepcopy(choices[index])
                else:
                    value = trial.suggest_categorical("model." + name, choices)
                candidate["model"][name] = value
            model_study.tell(trial, 0.0)
    return candidates


class TuningBudgetExceeded(RuntimeError):
    """The current trial has exhausted the study's allocated GPU wall time."""


class TuningBudget:
    """Count attempted trials and trial wall time on one allocated CUDA device.

    CPU trial time is recorded separately and does not consume GPU-hours. A CUDA
    trial includes construction, fitting, validation and cleanup; final evaluation
    is outside the search budget. Time checks are cooperative at batch boundaries.
    """

    def __init__(
        self,
        device_type: str = "cpu",
        max_trials: int = MAX_TRIALS,
        max_gpu_hours: float = MAX_GPU_HOURS,
        max_hours: Optional[float] = None,
    ) -> None:
        if isinstance(max_trials, bool) or not isinstance(max_trials, int) or max_trials < 1:
            raise ValueError("max_trials must be a positive integer")
        for name, value in (("max_gpu_hours", max_gpu_hours), ("max_hours", max_hours)):
            if value is not None and (
                isinstance(value, bool) or not math.isfinite(value) or value <= 0
            ):
                raise ValueError("{} must be positive and finite".format(name))
        self.max_trials = max_trials
        self.max_gpu_hours = max_gpu_hours
        self.max_hours = max_hours
        self.attempts = 0
        self.gpu_seconds = 0.0
        self.cpu_seconds = 0.0
        self.started_at: Optional[float] = None
        self.on_gpu = device_type == "cuda"

    @property
    def exhausted(self) -> bool:
        return self.stop_reason != "candidates_exhausted"

    def start_trial(self) -> None:
        self.attempts += 1
        self.started_at = time.monotonic()

    def check(self) -> None:
        elapsed = time.monotonic() - self.started_at if self.started_at is not None else 0.0
        if self.on_gpu and self.gpu_seconds + elapsed >= self.max_gpu_hours * 3600:
            raise TuningBudgetExceeded("GPU-hour tuning budget exhausted")
        if self.max_hours is not None:
            if self.gpu_seconds + self.cpu_seconds + elapsed >= self.max_hours * 3600:
                raise TuningBudgetExceeded("trial-hour tuning budget exhausted")

    def finish_trial(self) -> Dict[str, Any]:
        elapsed = time.monotonic() - self.started_at if self.started_at is not None else 0.0
        if self.on_gpu:
            self.gpu_seconds += elapsed
        else:
            self.cpu_seconds += elapsed
        self.started_at = None
        return {"duration_seconds": elapsed, "device_type": "cuda" if self.on_gpu else "cpu"}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_trials": self.max_trials,
            "max_gpu_hours": self.max_gpu_hours,
            "max_hours": self.max_hours,
            "attempted_trials": self.attempts,
            "gpu_hours": self.gpu_seconds / 3600,
            "cpu_hours": self.cpu_seconds / 3600,
            "accounting": "single-device allocated trial wall time; final test excluded",
            "check_interval": "batch boundaries; in-flight work may finish after the limit",
        }

    @property
    def stop_reason(self) -> str:
        if self.gpu_seconds >= self.max_gpu_hours * 3600:
            return "gpu_hours"
        if self.max_hours is not None:
            if self.gpu_seconds + self.cpu_seconds >= self.max_hours * 3600:
                return "trial_hours"
        if self.attempts >= self.max_trials:
            return "max_trials"
        return "candidates_exhausted"


class BudgetedLoader:
    """Keep train/validation iteration within a cooperative tuning deadline."""

    def __init__(self, loader, budget: TuningBudget) -> None:
        self.loader = loader
        self.budget = budget

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def iter_batches(self, split):
        batches = iter(self.loader.iter_batches(split))
        while True:
            self.budget.check()
            try:
                batch = next(batches)
            except StopIteration:
                break
            self.budget.check()
            yield batch

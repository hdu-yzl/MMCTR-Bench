"""Explicit single-phase and checkpoint-compatible phased optimizers."""

from typing import Any, Iterable, Mapping, Optional, Set, Tuple

import torch


class PhasedAdam(torch.optim.Adam):
    """Adam whose disjoint parameter groups are stepped by an explicit phase name."""

    def __init__(
        self,
        parameter_groups: Mapping[str, Iterable[torch.nn.Parameter]],
        learning_rates: Mapping[str, float],
        weight_decay: float = 0.0,
    ) -> None:
        if not isinstance(parameter_groups, Mapping) or not parameter_groups:
            raise ValueError("phased optimizer requires named parameter groups")
        if set(parameter_groups) != set(learning_rates):
            raise ValueError("phased optimizer groups and learning rates must match")
        if "main" not in parameter_groups:
            raise ValueError("phased optimizer requires a main parameter group")
        if weight_decay < 0.0:
            raise ValueError("weight_decay cannot be negative")
        groups = []
        parameter_ids: Set[int] = set()
        phases = []
        for phase, raw_parameters in parameter_groups.items():
            if not isinstance(phase, str) or not phase:
                raise ValueError("optimizer phase names must be non-empty strings")
            parameters = tuple(raw_parameters)
            if not parameters:
                raise ValueError("optimizer phase {!r} has no parameters".format(phase))
            if any(not isinstance(parameter, torch.nn.Parameter) for parameter in parameters):
                raise ValueError("optimizer phases must contain torch Parameters")
            duplicates = parameter_ids.intersection(id(parameter) for parameter in parameters)
            if duplicates:
                raise ValueError("phased optimizer parameter groups must be disjoint")
            parameter_ids.update(id(parameter) for parameter in parameters)
            learning_rate = float(learning_rates[phase])
            if learning_rate <= 0.0:
                raise ValueError("optimizer learning rates must be positive")
            phases.append(phase)
            groups.append(
                {
                    "params": parameters,
                    "lr": learning_rate,
                    "weight_decay": float(weight_decay),
                    "phase": phase,
                }
            )
        self.phases: Tuple[str, ...] = tuple(phases)
        super().__init__(groups)

    def step_phase(self, phase: str, closure=None):
        """Update only one phase while retaining other groups' pending gradients."""

        if phase not in self.phases:
            raise ValueError("unknown optimizer phase {!r}".format(phase))
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        hidden_gradients = []
        for group in self.param_groups:
            if group["phase"] == phase:
                continue
            for parameter in group["params"]:
                hidden_gradients.append((parameter, parameter.grad))
                parameter.grad = None
        try:
            optimizer_loss = super().step()
        finally:
            for parameter, gradient in hidden_gradients:
                parameter.grad = gradient
        return loss if closure is not None else optimizer_loss

    def step(self, closure=None):
        """Retain the normal optimizer interface by treating it as the main phase."""

        return self.step_phase("main", closure=closure)


def build_phased_adam(
    parameter_groups: Mapping[str, Iterable[torch.nn.Parameter]],
    learning_rates: Mapping[str, float],
    weight_decay: float = 0.0,
) -> PhasedAdam:
    """Build one serializable Adam state over explicit disjoint training phases."""

    return PhasedAdam(parameter_groups, learning_rates, weight_decay)


def build_optimizer(
    model: torch.nn.Module,
    name: str,
    learning_rate: float,
    weight_decay: float = 0.0,
    options: Optional[Mapping[str, Any]] = None,
) -> torch.optim.Optimizer:
    """Build one supported optimizer from validated scalar settings."""

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if weight_decay < 0:
        raise ValueError("weight_decay cannot be negative")
    kwargs = dict(options or {})
    kwargs.update({"lr": float(learning_rate), "weight_decay": float(weight_decay)})
    optimizers = {
        "sgd": torch.optim.SGD,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }
    try:
        optimizer_class = optimizers[str(name).lower()]
    except KeyError as error:
        raise ValueError("unsupported optimizer: {!r}".format(name)) from error
    return optimizer_class(model.parameters(), **kwargs)


__all__ = ["PhasedAdam", "build_optimizer", "build_phased_adam"]

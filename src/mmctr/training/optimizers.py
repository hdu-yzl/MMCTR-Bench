"""Explicit optimizer construction outside model classes."""

from typing import Any, Mapping, Optional

import torch


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


__all__ = ["build_optimizer"]

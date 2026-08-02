"""Training lifecycle callbacks."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class EarlyStopping:
    """Maximise a validation metric with deterministic patience handling."""

    patience: int
    min_delta: float = 0.0
    best: Optional[float] = None
    best_epoch: Optional[int] = None
    bad_epochs: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.patience, bool) or self.patience < 1:
            raise ValueError("patience must be at least 1")
        if self.min_delta < 0:
            raise ValueError("min_delta cannot be negative")

    def step(self, value: float, epoch: int) -> bool:
        """Record a value and return whether it is a new best."""

        improved = self.best is None or value > self.best + self.min_delta
        if improved:
            self.best = float(value)
            self.best_epoch = int(epoch)
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience


__all__ = ["EarlyStopping"]

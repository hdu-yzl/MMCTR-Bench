"""Typed configuration schemas for runtime-critical settings."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Union, cast


PathLike = Union[str, Path]


class ConfigValidationError(ValueError):
    """Raised when one or more configuration constraints are violated."""

    def __init__(self, issues: List[str]):
        self.issues = tuple(issues)
        super().__init__("invalid configuration: " + "; ".join(self.issues))


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _resolve_path(value: Any, key: str, project_root: Path, issues: List[str]) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        issues.append("{} must be a non-empty path".format(key))
        return project_root
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


@dataclass(frozen=True)
class TrainingConfig:
    """Validated training configuration consumed by runtime code."""

    max_epochs: int
    lr: float
    l2: float
    batch_size: int
    optim: str
    early_stop_patience: int
    checkpoint_dir: Path
    output_root: Path
    cuda: int
    log_dir: Path
    log_interval: int
    seed: int

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        project_root: PathLike,
    ) -> "TrainingConfig":
        if not isinstance(values, Mapping):
            raise ConfigValidationError(["training config must be a mapping"])

        expected = {
            "max_epochs",
            "lr",
            "l2",
            "batch_size",
            "optim",
            "early_stop_patience",
            "checkpoint_dir",
            "output_root",
            "cuda",
            "log_dir",
            "log_interval",
            "seed",
        }
        actual = set(values)
        issues = []
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        if missing:
            issues.append("missing keys: {}".format(", ".join(missing)))
        if unknown:
            issues.append("unknown keys: {}".format(", ".join(unknown)))

        def integer(key: str, minimum: int) -> int:
            value = values.get(key)
            if not _is_int(value):
                issues.append("{} must be an integer >= {}".format(key, minimum))
                return minimum
            integer_value = cast(int, value)
            if integer_value < minimum:
                issues.append("{} must be an integer >= {}".format(key, minimum))
                return minimum
            return integer_value

        def number(key: str, minimum: float, strict: bool = False) -> float:
            value = values.get(key)
            if not _is_number(value):
                operator = ">" if strict else ">="
                issues.append("{} must be a number {} {}".format(key, operator, minimum))
                return minimum
            number_value = float(cast(Union[int, float], value))
            invalid = number_value <= minimum if strict else number_value < minimum
            if invalid:
                operator = ">" if strict else ">="
                issues.append("{} must be a number {} {}".format(key, operator, minimum))
                return minimum
            return number_value

        max_epochs = integer("max_epochs", 1)
        early_stop_patience = integer("early_stop_patience", 1)
        batch_size = integer("batch_size", 1)
        log_interval = integer("log_interval", 1)
        seed = integer("seed", 0)
        cuda = integer("cuda", -1)
        lr = number("lr", 0.0, strict=True)
        l2 = number("l2", 0.0)

        optim_value = values.get("optim")
        if not isinstance(optim_value, str) or optim_value.lower() not in {"sgd", "adam", "adamw"}:
            issues.append("optim must be one of: sgd, adam, adamw")
            optim = "adamw"
        else:
            optim = optim_value.lower()

        if early_stop_patience > max_epochs:
            issues.append("early_stop_patience must be <= max_epochs")

        root = Path(project_root).expanduser().resolve()
        checkpoint_dir = _resolve_path(values.get("checkpoint_dir"), "checkpoint_dir", root, issues)
        output_root = _resolve_path(values.get("output_root"), "output_root", root, issues)
        log_dir = _resolve_path(values.get("log_dir"), "log_dir", root, issues)

        if issues:
            raise ConfigValidationError(issues)
        return cls(
            max_epochs=max_epochs,
            lr=lr,
            l2=l2,
            batch_size=batch_size,
            optim=optim,
            early_stop_patience=early_stop_patience,
            checkpoint_dir=checkpoint_dir,
            output_root=output_root,
            cuda=cuda,
            log_dir=log_dir,
            log_interval=log_interval,
            seed=seed,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_epochs": self.max_epochs,
            "lr": self.lr,
            "l2": self.l2,
            "batch_size": self.batch_size,
            "optim": self.optim,
            "early_stop_patience": self.early_stop_patience,
            "checkpoint_dir": str(self.checkpoint_dir),
            "output_root": str(self.output_root),
            "cuda": self.cuda,
            "log_dir": str(self.log_dir),
            "log_interval": self.log_interval,
            "seed": self.seed,
        }

"""Run defaults independent of dataset/model optimization parameters."""

from pathlib import Path
from typing import Any, Dict, Mapping

from .schema import PathLike, TrainingConfig


def runtime_defaults(project_root: PathLike) -> Dict[str, Any]:
    """Resolve conventional runtime locations; callers can override run arguments."""
    root = Path(project_root).expanduser().resolve()
    return {
        "checkpoint_dir": str(root / "outputs/checkpoints"),
        "quantization_artifact_dir": str(root / "outputs/quantization/artifacts"),
        "output_root": str(root / "outputs"),
        "cuda": 1,
        "log_dir": str(root / "outputs/logs"),
        "log_interval": 1000,
        "seed": 2025,
    }


def resolve_training_config(
    parameters: Mapping[str, Any], project_root: PathLike
) -> TrainingConfig:
    """Require model-owned hyperparameters and add only runtime defaults."""
    return TrainingConfig.from_mapping(
        {**runtime_defaults(project_root), **parameters}, project_root
    )

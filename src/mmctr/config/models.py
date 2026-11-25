"""Load authoritative per-dataset model parameters and publish validation winners."""

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .loading import load_yaml_mapping
from .schema import ConfigValidationError


MODEL_TRAINING_KEYS = frozenset(
    {"lr", "l2", "batch_size", "optim", "max_epochs", "early_stop_patience"}
)


def model_config_path(directory: Path, dataset: str, model: str) -> Path:
    """Use canonical names as path components, without allowing directory traversal."""

    for name in (dataset, model):
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ConfigValidationError(["invalid model config name: {!r}".format(name)])
    return Path(directory).expanduser().resolve() / dataset / (model + ".yaml")


def load_model_config(
    directory: Path, dataset: str, model: str, data_fingerprint: Optional[str] = None
) -> Dict[str, Any]:
    """Return saved defaults only when they match the requested dataset identity."""

    path = model_config_path(directory, dataset, model)
    payload = load_yaml_mapping(path)
    for key, expected in (
        ("schema_version", 1),
        ("dataset", dataset),
        ("model_name", model),
    ):
        if payload.get(key) != expected:
            raise ConfigValidationError(["model config {} mismatch: {}".format(key, path)])
    if (
        data_fingerprint is not None
        and payload.get("data_fingerprint") is not None
        and payload["data_fingerprint"] != data_fingerprint
    ):
        raise ConfigValidationError(["model config data_fingerprint mismatch: {}".format(path)])
    for key in ("model", "training", "selection"):
        if not isinstance(payload.get(key), dict):
            raise ConfigValidationError(["model config {} must be a mapping".format(key)])
    if "model_name" in payload["model"]:
        raise ConfigValidationError(["model_name cannot be a model override"])
    unknown = set(payload["training"]) - MODEL_TRAINING_KEYS
    if unknown:
        raise ConfigValidationError(["unsupported model training keys: {}".format(sorted(unknown))])
    if "pretraining" in payload:
        values = payload["pretraining"]
        if not isinstance(values, dict) or set(values) != MODEL_TRAINING_KEYS:
            raise ConfigValidationError(["pretraining must contain the six training parameters"])
    return payload


def write_model_config(selection_path: Path, directory: Path) -> Path:
    """Atomically publish a frozen validation winner, never test metrics or runtime paths.

    Each pair has its own file, so independent searches do not overwrite each other's
    parameters. A later successful search for the same pair replaces that pair's defaults.
    """

    selection_path = Path(selection_path).resolve()
    raw = selection_path.read_bytes()
    selection = json.loads(raw)
    if selection.get("selection_split") != "val" or selection.get("objective") != "val_auc":
        raise ConfigValidationError(["write-back requires a validation-AUC selection"])
    frozen = selection["frozen_config"]
    if len(frozen["data"]) != 1:
        raise ConfigValidationError(["write-back requires exactly one dataset"])
    dataset, data = next(iter(frozen["data"].items()))
    model_config = dict(frozen["model"])
    model = model_config.pop("model_name")
    scoped = model_config.get(dataset, model_config)
    model_config = {
        k: v for k, v in scoped.items() if k not in ("antm2c", "microlens", "tiktok", "ebnerd")
    }
    fingerprint = selection["data_fingerprint"]
    if not fingerprint or data.get("fingerprint") != fingerprint:
        raise ConfigValidationError(["selection data fingerprint differs from frozen config"])
    for key in ("validation_auc", "validation_log_loss"):
        if not math.isfinite(float(selection[key])):
            raise ConfigValidationError(["write-back requires finite validation metrics"])
    training = {key: frozen["training"][key] for key in sorted(MODEL_TRAINING_KEYS)}
    payload = {
        "schema_version": 1,
        "dataset": dataset,
        "model_name": model,
        "data_fingerprint": fingerprint,
        "model": model_config,
        "training": training,
        "selection": {
            "study_id": selection["study_id"],
            "trial_id": selection["selected_trial_id"],
            "run_id": selection["selected_run_id"],
            "seeds": selection["seeds"],
            "validation_auc": selection["validation_auc"],
            "validation_log_loss": selection["validation_log_loss"],
            "search_protocol": selection.get("search_protocol", {}),
            "source": str(selection_path),
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
    }
    destination = model_config_path(directory, dataset, model)
    if destination.exists():
        previous = load_model_config(directory, dataset, model)
        if "pretraining" in previous:
            payload["pretraining"] = previous["pretraining"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(destination.parent),
            prefix="." + destination.name + "-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            yaml.safe_dump(payload, handle, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return destination

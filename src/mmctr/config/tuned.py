"""Persist validation-selected defaults independently for each dataset/model pair."""

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .loading import load_yaml_mapping
from .schema import ConfigValidationError


TUNED_TRAINING_KEYS = frozenset(
    {"lr", "l2", "batch_size", "optim", "max_epochs", "early_stop_patience"}
)


def tuned_config_path(directory: Path, dataset: str, model: str) -> Path:
    """Use canonical names as path components, without allowing directory traversal."""

    for name in (dataset, model):
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ConfigValidationError(["invalid tuned config name: {!r}".format(name)])
    return Path(directory).expanduser().resolve() / dataset / (model + ".yaml")


def load_tuned_config(
    directory: Path, dataset: str, model: str, data_fingerprint: str
) -> Optional[Dict[str, Any]]:
    """Return saved defaults only when they match the requested dataset identity."""

    path = tuned_config_path(directory, dataset, model)
    if not path.is_file():
        return None
    payload = load_yaml_mapping(path)
    for key, expected in (
        ("schema_version", 1),
        ("dataset", dataset),
        ("model_name", model),
        ("data_fingerprint", data_fingerprint),
    ):
        if payload.get(key) != expected:
            raise ConfigValidationError(["tuned config {} mismatch: {}".format(key, path)])
    for key in ("model", "training", "selection"):
        if not isinstance(payload.get(key), dict):
            raise ConfigValidationError(["tuned config {} must be a mapping".format(key)])
    if "model_name" in payload["model"]:
        raise ConfigValidationError(["model_name cannot be a tuned model override"])
    unknown = set(payload["training"]) - TUNED_TRAINING_KEYS
    if unknown:
        raise ConfigValidationError(["unsupported tuned training keys: {}".format(sorted(unknown))])
    return payload


def write_tuned_config(selection_path: Path, directory: Path) -> Path:
    """Atomically publish a frozen validation winner, never test metrics or runtime paths.

    Each pair has its own file, so independent searches do not overwrite each other's
    defaults. A later successful search for the same pair replaces that pair's defaults.
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
    fingerprint = selection["data_fingerprint"]
    if not fingerprint or data.get("fingerprint") != fingerprint:
        raise ConfigValidationError(["selection data fingerprint differs from frozen config"])
    training = {key: frozen["training"][key] for key in sorted(TUNED_TRAINING_KEYS)}
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
    destination = tuned_config_path(directory, dataset, model)
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

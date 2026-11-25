"""Model-owned hyperparameters and independent quantizer settings need no shared YAML."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from mmctr.config import (
    load_model_config,
    load_yaml_mapping,
    resolve_training_config,
    runtime_defaults,
)


ROOT = Path(__file__).resolve().parents[2]


def test_every_model_resolves_without_shared_training_file():
    assert not (ROOT / "configs/training").exists()
    for path in (ROOT / "configs/models").glob("*/*.yaml"):
        payload = load_model_config(ROOT / "configs/models", path.parent.name, path.stem)
        resolved = resolve_training_config(payload["training"], ROOT)
        assert resolved.lr == payload["training"]["lr"]
        assert resolved.seed == 2025
        assert resolved.quantization_artifact_dir == ROOT / "outputs/quantization/artifacts"
        if path.stem == "psrq":
            pretraining = resolve_training_config(payload["pretraining"], ROOT)
            assert pretraining.lr == 0.001
            assert pretraining.max_epochs == 20


def test_missing_model_hyperparameter_has_no_hidden_fallback(tmp_path):
    payload = load_model_config(ROOT / "configs/models", "tiktok", "dnn")
    del payload["training"]["lr"]
    with pytest.raises(ValueError, match="missing keys: lr"):
        resolve_training_config(payload["training"], tmp_path)
    assert runtime_defaults(tmp_path)["output_root"] == str(tmp_path / "outputs")


def test_psrq_entrypoint_uses_pretraining_not_ctr_parameters(tmp_path, monkeypatch):
    from mmctr.quantization import psrq_entrypoint as entrypoint

    parameters = load_model_config(ROOT / "configs/models", "tiktok", "psrq")
    parameters["training"]["lr"] = 0.9
    parameters["pretraining"].update(lr=0.002, batch_size=8, max_epochs=6)
    parameters["model"].update(dropout=0.4, quantizer_dropout=0.0)
    calls = {}
    monkeypatch.setattr(entrypoint, "PROJECT_ROOT", tmp_path)
    model_path = tmp_path / "configs/models/tiktok/psrq.yaml"
    model_path.parent.mkdir(parents=True)
    model_path.write_text(yaml.safe_dump(parameters))
    dataset = load_yaml_mapping(ROOT / "configs/datasets/catalog.yaml")["tiktok"]
    dataset["use_mm_features"] = ["id", "text"]
    catalog = tmp_path / "configs/datasets/catalog.yaml"
    catalog.parent.mkdir(parents=True)
    catalog.write_text(yaml.safe_dump({"tiktok": dataset}))

    def loader(dataset, data, batch_size):
        calls["loader_batch"] = batch_size
        return SimpleNamespace(get_multi_modal=lambda: {"text": np.ones((3, 2))})

    class Quantizer:
        codebook_size = 4

        def __init__(self, config, data):
            calls["dropout"] = config["dropout"]

        def save(self, destination, metadata):
            calls["destination"] = destination

    def optimizer(model, optim, lr, l2):
        calls["lr"] = lr
        return object()

    def fit(model, features, optim, epochs, batch_size, device):
        calls.update(epochs=epochs, batch_size=batch_size)
        return 0.0

    monkeypatch.setattr(entrypoint, "get_data_loader", loader)
    monkeypatch.setattr(entrypoint, "PSRQPretrainer", Quantizer)
    monkeypatch.setattr(entrypoint, "build_optimizer", optimizer)
    monkeypatch.setattr(entrypoint, "fit_psrq", fit)
    entrypoint.train("tiktok", cuda=-1)
    assert calls["lr"] == 0.002
    assert calls["epochs"] == 6
    assert calls["batch_size"] == calls["loader_batch"] == 8
    assert calls["dropout"] == 0.0
    assert str(calls["destination"]).startswith(str(tmp_path / "outputs/quantization/artifacts"))

import subprocess
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import yaml

from mmctr.config import load_yaml_mapping
from mmctr.quantization import ensure, psrq_artifact_path, rq_artifact_path
from mmctr.quantization import psrq_entrypoint, rq_entrypoint
from mmctr.quantization.artifacts import load_quantization_artifact, save_quantization_artifact
from mmctr.quantization.psrq import PSRQPretrainer
from tests.unit.test_canonical_quantization import configs


@pytest.fixture
def project(tmp_path, monkeypatch):
    config, data = configs()
    data.update(name="tiktok", data_dir="data")
    catalog = tmp_path / "configs/datasets/catalog.yaml"
    catalog.parent.mkdir(parents=True)
    catalog.write_text(yaml.safe_dump({"tiktok": data}))
    training = dict(
        lr=0.001, l2=1e-7, batch_size=4, optim="adam", max_epochs=1, early_stop_patience=1
    )
    for model in ("qarm", "psrq", "rq"):
        path = tmp_path / "configs/models/tiktok" / (model + ".yaml")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(
                dict(
                    schema_version=1,
                    dataset="tiktok",
                    model_name=model,
                    model=dict(config, n_init=1, max_iter=2),
                    training=training,
                    pretraining=training,
                    selection={},
                )
            )
        )
    features = {
        name: np.random.RandomState(width).normal(size=(8, width)).astype("float32")
        for name, width in data["mm_dims"].items()
    }
    loader = Mock()
    loader.get_multi_modal.return_value = features
    for module in (ensure, rq_entrypoint, psrq_entrypoint):
        monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    for module in (rq_entrypoint, psrq_entrypoint):
        monkeypatch.setattr(module, "get_data_loader", lambda *a: loader)
    torch.set_num_threads(1)
    calls = []
    real_run = subprocess.run

    def run(command, **kwargs):
        if len(command) < 3 or not command[2].startswith("mmctr.quantization."):
            return real_run(command, **kwargs)
        calls.append(command)
        if command[2].endswith("rq_entrypoint") and not command[2].endswith("psrq_entrypoint"):
            rq_entrypoint.main(command[3:])
        else:
            psrq_entrypoint.main(command[3:])

    monkeypatch.setattr(ensure.subprocess, "run", run)
    return tmp_path / "outputs/quantization/artifacts", config, data, calls


def test_missing_artifacts_fit_real_quantizers_then_reuse_without_training(project):
    root, config, data, calls = project
    ensure.ensure_quantizers("tiktok", ["qarm", "psrq"], cuda=-1)
    assert len(calls) == 2
    assert all(
        ensure.artifact_problem(model, config, data, root) is None for model in ("qarm", "psrq")
    )
    before = {p: p.read_bytes() for p in root.rglob("*.npz")}
    ensure.ensure_quantizers("tiktok", ["qarm", "psrq"], cuda=-1)
    assert len(calls) == 2
    assert all(p.read_bytes() == contents for p, contents in before.items())


@pytest.mark.parametrize("damage", ["missing", "corrupt", "incompatible"])
def test_partial_rq_recovery_preserves_valid_modalities(project, damage):
    root, _, _, calls = project
    ensure.ensure_quantizers("tiktok", ["qarm"])
    image = rq_artifact_path(root, "tiktok", "image")
    original = image.read_bytes()
    text = rq_artifact_path(root, "tiktok", "text")
    if damage == "missing":
        text.unlink()
    elif damage == "corrupt":
        text.write_bytes(b"broken npz")
    else:
        text.write_bytes(original)  # Valid NPZ, wrong modality and dimension.
    ensure.ensure_quantizers("tiktok", ["qarm"])
    assert calls[-1][calls[-1].index("--modalities") + 1 :] == ["text"]
    assert image.read_bytes() == original
    assert not psrq_artifact_path(root, "tiktok").exists()


def test_dry_run_and_non_quantized_selection_never_fit_or_write(project, capsys):
    root, _, _, calls = project
    ensure.ensure_quantizers("tiktok", ["dnn"])
    assert not root.exists()
    ensure.ensure_quantizers("tiktok", ["psrq"], cuda=3, use_local_data=True, dry_run=True)
    assert not calls and not root.exists()
    output = capsys.readouterr().out
    assert "--cuda 3 --use-local-data" in output
    assert "rq_entrypoint" not in output.replace("psrq_entrypoint", "")


def test_psrq_incomplete_state_is_refitted_without_rq(project):
    root, config, data, calls = project
    ensure.ensure_quantizers("tiktok", ["psrq"], cuda=-1)
    path = psrq_artifact_path(root, "tiktok")
    metadata, arrays = load_quantization_artifact(path, PSRQPretrainer.artifact_kind)
    arrays.pop(next(iter(arrays)))
    save_quantization_artifact(path, PSRQPretrainer.artifact_kind, metadata, arrays)
    ensure.ensure_quantizers("tiktok", ["psrq"], cuda=-1)
    assert len(calls) == 2
    assert not (root / "rq").exists()
    assert ensure.artifact_problem("psrq", config, data, root) is None


def test_pretraining_failure_and_missing_postfit_artifacts_propagate(project, monkeypatch):
    _, _, _, calls = project
    monkeypatch.setattr(
        ensure.subprocess, "run", Mock(side_effect=subprocess.CalledProcessError(1, []))
    )
    with pytest.raises(subprocess.CalledProcessError):
        ensure.ensure_quantizers("tiktok", ["qarm", "psrq"])
    monkeypatch.setattr(ensure.subprocess, "run", Mock())
    with pytest.raises(RuntimeError, match="did not produce usable artifacts"):
        ensure.ensure_quantizers("tiktok", ["qarm", "psrq"])
    assert not calls


def test_qarm_config_mismatch_fails_before_expensive_pretraining(project):
    _, _, _, calls = project
    path = ensure.PROJECT_ROOT / "configs/models/tiktok/rq.yaml"
    payload = load_yaml_mapping(path)
    payload["model"]["n_levels"] += 1
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="QARM and RQ n_levels must match"):
        ensure.ensure_quantizers("tiktok", ["qarm"])
    assert not calls

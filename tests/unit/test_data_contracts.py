from dataclasses import replace
from types import SimpleNamespace

import pytest

from mmctr.core import ContractError
from mmctr.data import DatasetManifest, ensure_canonical_loader


def test_native_loader_retains_audited_manifest_and_rejects_wrong_identity():
    manifest = DatasetManifest("tiktok", "v1", "memory", 3, 0, {"id": 1})
    source = SimpleNamespace(
        dataset_name="tiktok", manifest=manifest, iter_batches=lambda _: iter(())
    )
    assert ensure_canonical_loader("tiktok", source) is source
    assert source.manifest is manifest
    assert manifest.fingerprint == replace(manifest).fingerprint
    with pytest.raises(ContractError, match="identity"):
        ensure_canonical_loader("microlens", source)
    source.manifest = replace(manifest, name="microlens")
    with pytest.raises(ContractError, match="identity"):
        ensure_canonical_loader("tiktok", source)


def test_non_native_loader_fails_before_training():
    with pytest.raises(ContractError, match="iter_batches"):
        ensure_canonical_loader("tiktok", object())

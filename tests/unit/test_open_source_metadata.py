from pathlib import Path

from mmctr.models.common.registry import MODEL_REGISTRY


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_paper_model_taxonomy_matches_the_public_registry() -> None:
    id_only = {"dnn", "deepfm", "din", "autoint", "dcn"}
    multimodal = {
        "mmmlp",
        "diff_msin",
        "dmf",
        "make",
        "m3srec",
        "em3",
        "psrq",
        "qarm",
        "naml",
        "mb",
        "lmf",
        "simcen",
        "mtfn",
        "pamd",
        "gmmf",
        "marn",
    }
    software_variants = {"dnn_mm", "dnn_mm_seq"}

    assert len(id_only) == 5
    assert len(multimodal) == 16
    assert set(MODEL_REGISTRY.names()) == id_only | multimodal | software_variants
    assert MODEL_REGISTRY.canonical_name("mcca") == "psrq"


def test_local_paper_is_explicitly_excluded_from_git() -> None:
    ignore_rules = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "/MMCTR_Benchmark.pdf" in ignore_rules


def test_source_distribution_includes_public_provenance_files() -> None:
    manifest = (PROJECT_ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    for required in (
        "CITATION.cff",
        "CONTRIBUTING.md",
        "data/README.md",
        "docs/data.md",
        "docs/models.md",
        "docs/benchmark-protocol.md",
        "recursive-include scripts *.sh",
    ):
        assert required in manifest


def test_apache_license_metadata_is_consistent() -> None:
    license_text = (PROJECT_ROOT / "LICENSE").read_text(encoding="utf-8")
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    citation = (PROJECT_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    manifest = (PROJECT_ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert "Apache License" in license_text
    assert "Version 2.0, January 2004" in license_text
    assert 'license = { file = "LICENSE" }' in pyproject
    assert '"License :: OSI Approved :: Apache Software License"' in pyproject
    assert "license: Apache-2.0" in citation
    assert "include LICENSE" in manifest

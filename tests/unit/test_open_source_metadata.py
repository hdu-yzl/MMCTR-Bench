from pathlib import Path

from mmctr.models.registry import MODEL_REGISTRY


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_public_provenance_docs_cover_all_datasets_and_models() -> None:
    data_readme = (PROJECT_ROOT / "data" / "README.md").read_text(encoding="utf-8")
    references = (PROJECT_ROOT / "docs" / "references.md").read_text(encoding="utf-8")

    for dataset in ("AntM2C", "MicroLens", "TikTok"):
        assert dataset in data_readme
        assert dataset in references

    for model_name in MODEL_REGISTRY.names():
        assert f"`{model_name}`" in references

    assert "https://github.com/westlake-repl/MicroLens" in data_readme
    assert "https://www.atecup.com/home" in data_readme
    assert "https://github.com/nickwzk/InvRL" in data_readme
    assert "does not redistribute" in data_readme


def test_public_readme_links_provenance_documents() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "data/README.md" in readme
    assert "docs/references.md" in readme


def test_final_readme_documents_one_click_training() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    for required in (
        "scripts/train_model.sh",
        "scripts/train_all_models.sh",
        "scripts/pretrain_quantizers.sh",
        "data/raw/antm2c",
        "data/raw/microlens",
        "data/raw/tiktok",
        "outputs/",
        "23 canonical CTR models",
    ):
        assert required in readme

    assert "release candidate" not in readme.lower()
    assert "refactored with the assistance of AI" in readme


def test_release_checklist_keeps_known_blockers_explicit() -> None:
    checklist = (PROJECT_ROOT / "docs" / "release-checklist.md").read_text(encoding="utf-8")

    for required in (
        "python -m build",
        "repository-external",
        "LICENSE",
        "AntM2C",
        "SHA-256",
        "git ls-files",
    ):
        assert required in checklist

    assert "OSS-001" in checklist
    assert "REL-001" in checklist


def test_source_distribution_includes_public_provenance_files() -> None:
    manifest = (PROJECT_ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    for required in (
        "CITATION.cff",
        "CONTRIBUTING.md",
        "data/README.md",
        "docs/references.md",
        "docs/release-checklist.md",
    ):
        assert required in manifest


def test_apache_license_metadata_is_public_and_excludes_third_party_data() -> None:
    """Keep the software license consistent while excluding third-party data and checkpoints."""
    license_text = (PROJECT_ROOT / "LICENSE").read_text(encoding="utf-8")
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    citation = (PROJECT_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    manifest = (PROJECT_ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert "Apache License" in license_text
    assert "Version 2.0, January 2004" in license_text
    assert 'license = { file = "LICENSE" }' in pyproject
    assert '"License :: OSI Approved :: Apache Software License"' in pyproject
    assert "Apache-2.0" in readme
    assert "third-party datasets and checkpoints are not covered" in readme.lower()
    assert "license: Apache-2.0" in citation
    assert "include LICENSE" in manifest

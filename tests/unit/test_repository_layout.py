import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_repository_has_one_configuration_and_output_interface() -> None:
    assert not (PROJECT_ROOT / "config").exists()
    assert not (PROJECT_ROOT / "experiments").exists()
    assert not (PROJECT_ROOT / "setup.py").exists()
    assert not (PROJECT_ROOT / "__init__.py").exists()
    assert not (PROJECT_ROOT / "src" / "__init__.py").exists()

    for relative in (
        "configs/datasets/catalog.yaml",
        "configs/models/catalog.yaml",
        "configs/training/default.yaml",
        "configs/local/paths.example.yaml",
    ):
        assert (PROJECT_ROOT / relative).is_file()


def test_dataset_filesystem_interface_is_tracked_as_documentation() -> None:
    for area in ("raw", "processed"):
        assert (PROJECT_ROOT / "data" / area / "README.md").is_file()
        for dataset in ("antm2c", "microlens", "tiktok"):
            assert (PROJECT_ROOT / "data" / area / dataset / "README.md").is_file()


def test_public_markdown_is_english_only() -> None:
    failures = []
    for path in PROJECT_ROOT.rglob("*.md"):
        relative = path.relative_to(PROJECT_ROOT)
        if "outputs" in relative.parts or any(part.startswith(".") for part in relative.parts):
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if any("\u3400" <= character <= "\u9fff" for character in line):
                failures.append("{}:{}".format(relative.as_posix(), line_number))
    assert failures == []


def test_source_tree_and_repository_exclude_generated_research_artifacts() -> None:
    assert not (PROJECT_ROOT / "src" / "analysis").exists()
    assert not (PROJECT_ROOT / "reports" / "figures").exists()


def test_one_click_launchers_are_executable_release_sources() -> None:
    """Keep launchers executable in the source tree and included in source distributions."""
    scripts = PROJECT_ROOT / "scripts"
    for name in ("train_model.sh", "train_all_models.sh", "pretrain_quantizers.sh"):
        path = scripts / name
        assert path.is_file()
        assert os.access(path, os.X_OK)

    manifest = (PROJECT_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "recursive-include scripts *.sh" in manifest

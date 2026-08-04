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


def test_source_tree_contains_runtime_only_and_figures_live_in_reports() -> None:
    assert not (PROJECT_ROOT / "src" / "analysis").exists()
    figures = tuple((PROJECT_ROOT / "reports" / "figures").glob("*"))
    assert len(figures) == 19
    assert all(path.suffix.lower() in {".pdf", ".png"} for path in figures)

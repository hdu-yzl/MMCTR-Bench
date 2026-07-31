import tempfile
import unittest
from pathlib import Path

from mmctr.config import (
    ConfigValidationError,
    LocalPaths,
    load_dataset_catalog,
    load_local_paths,
    resolve_dataset_config,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATH = REPOSITORY_ROOT / "configs" / "local" / "paths.example.yaml"


class LocalPathsTest(unittest.TestCase):
    def test_example_plus_one_environment_override_is_usable(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_directory = Path(temporary_directory) / "antm2c"
            data_directory.mkdir()

            paths = load_local_paths(
                EXAMPLE_PATH,
                environ={"MMCTR_ANTM2C_DATA_DIR": str(data_directory)},
            )

            self.assertEqual(paths.datasets, {"antm2c": data_directory.resolve()})
            self.assertIsNone(paths.output_root)

    def test_environment_overrides_local_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            file_data = root / "from-file"
            environment_data = root / "from-environment"
            file_data.mkdir()
            environment_data.mkdir()
            local_file = root / "paths.yaml"
            local_file.write_text(
                "datasets:\n  antm2c: {!s}\n".format(file_data),
                encoding="utf-8",
            )

            paths = load_local_paths(
                local_file,
                environ={"MMCTR_ANTM2C_DATA_DIR": str(environment_data)},
            )

            self.assertEqual(paths.datasets["antm2c"], environment_data.resolve())

    def test_missing_file_and_environment_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing = Path(temporary_directory) / "paths.yaml"

            with self.assertRaisesRegex(ConfigValidationError, "paths.example.yaml"):
                load_local_paths(missing, environ={})

    def test_relative_and_missing_local_directories_are_rejected(self):
        with self.assertRaisesRegex(ConfigValidationError, "must be absolute"):
            LocalPaths.from_mapping({"datasets": {"antm2c": "relative/path"}})

        with tempfile.TemporaryDirectory() as temporary_directory:
            missing = Path(temporary_directory) / "missing"
            with self.assertRaisesRegex(ConfigValidationError, "directory does not exist"):
                LocalPaths.from_mapping({"datasets": {"antm2c": str(missing)}})

    def test_dataset_resolution_is_absolute_and_does_not_mutate_input(self):
        original = {
            "name": "antm2c",
            "data_dir": "data/processed/antm2c",
            "using_local_data": False,
        }

        canonical = resolve_dataset_config("antm2c", original, REPOSITORY_ROOT)

        self.assertEqual(
            canonical["data_dir"],
            str((REPOSITORY_ROOT / "data/processed/antm2c").resolve()),
        )
        self.assertFalse(canonical["using_local_data"])
        self.assertEqual(original["data_dir"], "data/processed/antm2c")

        with tempfile.TemporaryDirectory() as temporary_directory:
            local_directory = Path(temporary_directory).resolve()
            local_paths = LocalPaths.from_mapping({"datasets": {"antm2c": str(local_directory)}})
            local = resolve_dataset_config(
                "antm2c", original, REPOSITORY_ROOT, local_paths=local_paths
            )
            self.assertEqual(local["data_dir"], str(local_directory))
            self.assertTrue(local["using_local_data"])

    def test_example_drives_selected_dataset_catalog_without_missing_local_yaml(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_directory = Path(temporary_directory) / "antm2c"
            data_directory.mkdir()

            catalog = load_dataset_catalog(
                REPOSITORY_ROOT / "config/data.yaml",
                "antm2c",
                project_root=REPOSITORY_ROOT,
                use_local_data=True,
                local_paths_path=EXAMPLE_PATH,
                environ={"MMCTR_ANTM2C_DATA_DIR": str(data_directory)},
            )

            self.assertEqual(catalog["antm2c"]["data_dir"], str(data_directory.resolve()))
            self.assertTrue(catalog["antm2c"]["using_local_data"])

    def test_source_tree_has_no_missing_legacy_local_config_references(self):
        offenders = []
        for path in (REPOSITORY_ROOT / "src").rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            if "local_data.yaml" in source or "local_seq_data.yaml" in source:
                offenders.append(path.relative_to(REPOSITORY_ROOT).as_posix())

        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()

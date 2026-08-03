import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from mmctr.config import (
    ConfigValidationError,
    TrainingConfig,
    load_training_config,
    load_yaml_mapping,
    merge_config_layers,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRAIN_CONFIG_PATH = REPOSITORY_ROOT / "config" / "train.yaml"


class TrainingConfigTest(unittest.TestCase):
    def _valid_mapping(self):
        return load_yaml_mapping(TRAIN_CONFIG_PATH)

    def test_repository_training_config_is_typed_and_paths_are_root_relative(self):
        config = load_training_config(TRAIN_CONFIG_PATH)

        self.assertEqual(config.optim, "adamw")
        self.assertEqual(config.output_root, (REPOSITORY_ROOT / "outputs").resolve())
        self.assertEqual(
            config.checkpoint_dir,
            (REPOSITORY_ROOT / "experiments" / "checkpoints").resolve(),
        )
        self.assertEqual(
            config.quantization_artifact_dir,
            (REPOSITORY_ROOT / "experiments" / "quantization").resolve(),
        )
        self.assertIsInstance(config.lr, float)
        with self.assertRaises(FrozenInstanceError):
            config.max_epochs = 10

    def test_missing_unknown_type_range_and_cross_field_errors_are_rejected(self):
        cases = []

        missing = self._valid_mapping()
        missing.pop("seed")
        cases.append((missing, "missing keys: seed"))

        unknown = self._valid_mapping()
        unknown["mystery"] = 1
        cases.append((unknown, "unknown keys: mystery"))

        wrong_type = self._valid_mapping()
        wrong_type["batch_size"] = True
        cases.append((wrong_type, "batch_size must be an integer"))

        bad_range = self._valid_mapping()
        bad_range["lr"] = 0
        cases.append((bad_range, "lr must be a number >"))

        bad_cross_field = self._valid_mapping()
        bad_cross_field["early_stop_patience"] = 6
        cases.append((bad_cross_field, "early_stop_patience must be <= max_epochs"))

        for values, expected_message in cases:
            with self.subTest(expected_message=expected_message):
                with self.assertRaisesRegex(ConfigValidationError, expected_message):
                    TrainingConfig.from_mapping(values, REPOSITORY_ROOT)

    def test_duplicate_yaml_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "duplicate.yaml"
            path.write_text("seed: 1\nseed: 2\n", encoding="utf-8")

            with self.assertRaisesRegex(ConfigValidationError, "duplicate YAML key 'seed'"):
                load_yaml_mapping(path)

    def test_tracked_executable_yaml_files_have_unique_keys(self):
        config_directory = REPOSITORY_ROOT / "config"
        self.assertFalse((config_directory / "best_params.yaml").exists())

        for path in sorted(config_directory.glob("*.yaml")):
            with self.subTest(path=path.name):
                self.assertIsInstance(load_yaml_mapping(path), dict)
        self.assertIsInstance(
            load_yaml_mapping(REPOSITORY_ROOT / "configs/local/paths.example.yaml"),
            dict,
        )

    def test_layers_merge_recursively_without_mutating_inputs(self):
        training = {"lr": 0.001, "nested": {"keep": 1, "replace": 1}}
        dataset = {"batch_size": 64, "nested": {"replace": 2}}
        cli = {"lr": 0.01}

        resolved = merge_config_layers(training, dataset, None, cli)

        self.assertEqual(
            resolved,
            {"lr": 0.01, "batch_size": 64, "nested": {"keep": 1, "replace": 2}},
        )
        self.assertEqual(training["nested"]["replace"], 1)
        self.assertEqual(dataset["nested"]["replace"], 2)


if __name__ == "__main__":
    unittest.main()

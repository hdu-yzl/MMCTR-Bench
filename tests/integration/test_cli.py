import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"


class CliIntegrationTest(unittest.TestCase):
    def _run(self, *arguments):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(SOURCE_ROOT)
        with tempfile.TemporaryDirectory() as temporary_directory:
            return subprocess.run(
                [sys.executable, "-m", "mmctr.cli"] + list(arguments),
                cwd=temporary_directory,
                env=environment,
                capture_output=True,
                text=True,
                timeout=60,
            )

    def test_root_and_train_help_are_available(self):
        root = self._run("--help")
        train = self._run("train", "--help")

        self.assertEqual(root.returncode, 0, msg=root.stderr)
        self.assertIn("validate-config", root.stdout)
        self.assertIn("list-models", root.stdout)
        self.assertEqual(train.returncode, 0, msg=train.stderr)
        self.assertIn("--use-local-data", train.stdout)
        self.assertNotIn("tensorflow", (root.stderr + train.stderr).lower())

    def test_list_commands_are_deterministic(self):
        models = self._run("list-models")
        datasets = self._run("list-datasets")

        self.assertEqual(models.returncode, 0, msg=models.stderr)
        self.assertEqual(datasets.returncode, 0, msg=datasets.stderr)
        model_names = models.stdout.splitlines()
        dataset_names = datasets.stdout.splitlines()
        self.assertEqual(model_names, sorted(model_names))
        self.assertIn("dnn", model_names)
        self.assertIn("mcca", model_names)
        self.assertEqual(dataset_names, ["antm2c", "microlens", "tiktok"])
        self.assertNotIn("tensorflow", (models.stderr + datasets.stderr).lower())

    def test_validate_config_outputs_resolved_json(self):
        result = self._run(
            "validate-config",
            "--config",
            str(REPOSITORY_ROOT / "config/train.yaml"),
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        resolved = json.loads(result.stdout)
        self.assertEqual(resolved["optim"], "adamw")
        self.assertTrue(Path(resolved["output_root"]).is_absolute())

    def test_cli_and_catalog_imports_do_not_load_training_frameworks(self):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(SOURCE_ROOT)
        source = (
            "import sys\n"
            "import mmctr.cli\n"
            "import mmctr.models\n"
            "import mmctr.data\n"
            "assert 'torch' not in sys.modules\n"
            "assert 'tensorflow' not in sys.modules\n"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = subprocess.run(
                [sys.executable, "-c", source],
                cwd=temporary_directory,
                env=environment,
                capture_output=True,
                text=True,
                timeout=60,
            )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)

    def test_legacy_trainer_import_does_not_parse_process_arguments(self):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(SOURCE_ROOT)
        source = (
            "import sys\n"
            "sys.argv = ['host-process', '--unrelated-host-argument']\n"
            "from trainers.Trainers import Trainer\n"
            "assert Trainer.__name__ == 'Trainer'\n"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = subprocess.run(
                [sys.executable, "-c", source],
                cwd=temporary_directory,
                env=environment,
                capture_output=True,
                text=True,
                timeout=60,
            )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()

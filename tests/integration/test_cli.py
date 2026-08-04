import json
import hashlib
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
        self.assertIn("list-quantizers", root.stdout)
        self.assertEqual(train.returncode, 0, msg=train.stderr)
        self.assertIn("--use-local-data", train.stdout)
        self.assertNotIn("tensorflow", (root.stderr + train.stderr).lower())

    def test_list_commands_are_deterministic(self):
        models = self._run("list-models")
        datasets = self._run("list-datasets")
        quantizers = self._run("list-quantizers")

        self.assertEqual(models.returncode, 0, msg=models.stderr)
        self.assertEqual(datasets.returncode, 0, msg=datasets.stderr)
        self.assertEqual(quantizers.returncode, 0, msg=quantizers.stderr)
        model_names = models.stdout.splitlines()
        dataset_names = datasets.stdout.splitlines()
        quantizer_names = quantizers.stdout.splitlines()
        self.assertEqual(model_names, sorted(model_names))
        self.assertIn("dnn", model_names)
        self.assertIn("mcca", model_names)
        self.assertEqual(dataset_names, ["antm2c", "microlens", "tiktok"])
        self.assertEqual(quantizer_names, ["psrq", "rq"])
        self.assertNotIn(
            "tensorflow", (models.stderr + datasets.stderr + quantizers.stderr).lower()
        )

    def test_validate_config_outputs_resolved_json(self):
        result = self._run(
            "validate-config",
            "--config",
            str(REPOSITORY_ROOT / "configs/training/default.yaml"),
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        resolved = json.loads(result.stdout)
        self.assertEqual(resolved["optim"], "adamw")
        self.assertTrue(Path(resolved["output_root"]).is_absolute())

    def test_plan_fusion_study_writes_canonical_matrix(self):
        config = """\
dataset: fixture
data_fingerprint: data-v1
data:
  name: fixture
  id_feature_num: 30
  use_mm_features: [id, text]
  use_mm_seq_features: [id, text]
  mm_dims: {id: 0, text: 3}
  mm_seq_dims: {id: 0, text: 3}
  user_features: [id]
  user_features_dim: {id: 0}
model_configs:
  dnn_mm:
    latent_dim: 4
    projection_dim: 4
    mlp_dims: [8]
    dropout: 0.0
    batch_norm: false
models: [dnn_mm]
fusions: [cat, mean]
seeds: [3, 5]
"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "fusion.yaml"
            output_path = root / "matrix.json"
            config_path.write_text(config, encoding="utf-8")

            result = self._run(
                "plan-fusion-study",
                "--config",
                str(config_path),
                "--output",
                str(output_path),
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            summary = json.loads(result.stdout)
            matrix = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(4, summary["task_count"])
            self.assertEqual(str(output_path.resolve()), summary["matrix_path"])
            self.assertEqual("fusion-study-matrix-v1", matrix["schema"])
            self.assertNotIn("models.ctr_models", result.stderr)
            self.assertNotIn("analysis.fusion_analysis", result.stderr)

    def test_plan_robustness_study_writes_canonical_matrix(self):
        config = """\
dataset: fixture
data_fingerprint: data-v1
data: {name: fixture}
model_configs:
  dnn_mm: {latent_dim: 4}
models: [dnn_mm]
modalities: [text, image]
probabilities: [0.0, 0.5]
seeds: [3, 5]
splits: [train, val, test]
"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "robustness.yaml"
            output_path = root / "matrix.json"
            config_path.write_text(config, encoding="utf-8")

            result = self._run(
                "plan-robustness-study",
                "--config",
                str(config_path),
                "--output",
                str(output_path),
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            summary = json.loads(result.stdout)
            matrix = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(4, summary["task_count"])
            self.assertEqual(str(output_path.resolve()), summary["matrix_path"])
            self.assertEqual("robustness-study-matrix-v1", matrix["schema"])
            self.assertNotIn("modal_robustness", result.stderr)

    def test_plan_alignment_study_writes_canonical_matrix(self):
        config = """\
dataset: fixture
data_fingerprint: data-v1
data: {name: fixture}
model_configs:
  dnn_mm: {latent_dim: 4}
models: [dnn_mm]
methods: [cosine, mse]
weights: [0.0, 0.3]
representations: {text: text_projector, image: image_projector}
seeds: [3]
"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "alignment.yaml"
            output_path = root / "matrix.json"
            config_path.write_text(config, encoding="utf-8")

            result = self._run(
                "plan-alignment-study",
                "--config",
                str(config_path),
                "--output",
                str(output_path),
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            summary = json.loads(result.stdout)
            matrix = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(4, summary["task_count"])
            self.assertEqual(str(output_path.resolve()), summary["matrix_path"])
            self.assertEqual("alignment-study-matrix-v1", matrix["schema"])
            self.assertNotIn("alignment_analysis", result.stderr)

    def test_plot_results_renders_only_from_standard_result_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            inputs = []
            for index, model in enumerate(("dnn", "dcn"), start=1):
                path = root / "result-{}.json".format(index)
                path.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "task_id": "task-{}".format(index),
                            "task_key": "key-{}".format(index),
                            "run_id": "run-{}".format(index),
                            "status": "completed",
                            "dataset": "fixture",
                            "model": model,
                            "seed": 3,
                            "device": "cpu",
                            "data_fingerprint": "data-v1",
                            "metrics": {"val_auc": 0.6 + index / 10.0},
                            "artifact_dir": None,
                            "error": None,
                            "metadata": {},
                        }
                    ),
                    encoding="utf-8",
                )
                inputs.append(path)
            output_path = root / "metric.png"

            result = self._run(
                "plot-results",
                "--inputs",
                *(str(path) for path in inputs),
                "--output",
                str(output_path),
                "--metric",
                "val_auc",
                "--group-by",
                "model",
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            summary = json.loads(result.stdout)
            self.assertTrue(output_path.is_file())
            self.assertTrue(Path(summary["provenance_path"]).is_file())
            self.assertEqual(2, summary["input_count"])

    def test_plan_cold_start_study_requires_verified_audit(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            audit_body = {
                "schema": "mmctr-cold-start-audit-v1",
                "audit": {
                    "target": "item",
                    "regime": "zero_shot",
                    "target_count": 1,
                    "train_events": 1,
                    "evaluation_events": 1,
                    "maximum_support_interactions": 0,
                    "support_counts": {"11": 0},
                    "protocol_fingerprint": "a" * 64,
                },
            }
            audit = dict(audit_body)
            audit["manifest_fingerprint"] = hashlib.sha256(
                json.dumps(audit_body, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            audit_path = root / "audit.json"
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            config_path = root / "cold-start.yaml"
            config_path.write_text(
                """\
dataset: fixture
data_fingerprint: data-v1
data: {name: fixture}
model_configs:
  dnn: {latent_dim: 4}
models: [dnn]
seeds: [3]
audit_manifest: audit.json
""",
                encoding="utf-8",
            )
            output_path = root / "matrix.json"

            result = self._run(
                "plan-cold-start-study",
                "--config",
                str(config_path),
                "--output",
                str(output_path),
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            summary = json.loads(result.stdout)
            matrix = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(1, summary["task_count"])
            self.assertEqual("cold-start-study-matrix-v1", matrix["schema"])

    def test_cli_and_catalog_imports_do_not_load_training_frameworks(self):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(SOURCE_ROOT)
        source = (
            "import sys\n"
            "import mmctr.cli\n"
            "import mmctr.models\n"
            "import mmctr.data\n"
            "import mmctr.quantization\n"
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

    def test_trainer_imports_do_not_parse_process_arguments(self):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(SOURCE_ROOT)
        source = (
            "import sys\n"
            "sys.argv = ['host-process', '--unrelated-host-argument']\n"
            "from mmctr.training.entrypoint import Trainer\n"
            "from mmctr.quantization.rq_entrypoint import train as train_rq\n"
            "from mmctr.quantization.psrq_entrypoint import train as train_psrq\n"
            "assert Trainer.__name__ == 'Trainer'\n"
            "assert train_rq.__name__ == 'train'\n"
            "assert train_psrq.__name__ == 'train'\n"
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

    def test_legacy_top_level_runtime_packages_are_removed(self):
        for legacy_package in ("data", "trainers", "utils"):
            self.assertFalse((SOURCE_ROOT / legacy_package).exists())
        self.assertEqual([], list((SOURCE_ROOT / "analysis").rglob("*.py")))


if __name__ == "__main__":
    unittest.main()

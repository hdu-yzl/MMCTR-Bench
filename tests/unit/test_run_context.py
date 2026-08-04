import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import yaml

from mmctr.utils.run_context import config_fingerprint, create_run_context


FIXED_TIME = datetime(2026, 7, 31, 6, 30, 45, 123456, tzinfo=timezone.utc)
CONFIG = {
    "model": {"name": "dnn", "dropout": 0.1},
    "train": {"seed": 2025, "lr": 0.001},
}


class RunContextTest(unittest.TestCase):
    def test_config_fingerprint_is_stable_across_mapping_order(self):
        reordered = {
            "train": {"lr": 0.001, "seed": 2025},
            "model": {"dropout": 0.1, "name": "dnn"},
        }

        self.assertEqual(config_fingerprint(CONFIG), config_fingerprint(reordered))

    def test_context_materializes_provenance_and_final_status(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            context = create_run_context(
                output_root=temporary_directory,
                experiment_name="training",
                dataset="antm2c",
                model="dnn",
                resolved_config=CONFIG,
                now=FIXED_TIME,
                entropy="abcd1234",
                repository_root=temporary_directory,
                metadata={"seed": 2025, "device": "cpu"},
            )

            expected_prefix = "20260731T063045123456Z-{}-".format(config_fingerprint(CONFIG))
            self.assertTrue(context.run_id.startswith(expected_prefix))
            self.assertTrue(context.checkpoints_dir.is_dir())
            self.assertTrue(context.metrics_path.is_file())
            self.assertEqual(context.metrics_path.read_text(encoding="utf-8"), "")
            self.assertEqual(
                yaml.safe_load(context.resolved_config_path.read_text(encoding="utf-8")),
                CONFIG,
            )

            context.finalize("completed", summary={"best_val_auc": 0.75})

            metadata = json.loads(context.metadata_path.read_text(encoding="utf-8"))
            summary = json.loads(context.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "completed")
            self.assertIsNotNone(metadata["ended_at"])
            self.assertEqual(metadata["seed"], 2025)
            self.assertEqual(metadata["device"], "cpu")
            self.assertEqual(summary, {"best_val_auc": 0.75})
            self.assertEqual(list(context.root_dir.glob("*.tmp")), [])

    def test_parallel_identical_configs_get_distinct_directories(self):
        with tempfile.TemporaryDirectory() as temporary_directory:

            def create_one(_index):
                return create_run_context(
                    output_root=temporary_directory,
                    experiment_name="training",
                    dataset="antm2c",
                    model="dnn",
                    resolved_config=CONFIG,
                    now=FIXED_TIME,
                    repository_root=temporary_directory,
                )

            with ThreadPoolExecutor(max_workers=8) as executor:
                contexts = list(executor.map(create_one, range(32)))

            run_ids = {context.run_id for context in contexts}
            run_dirs = {context.root_dir for context in contexts}
            self.assertEqual(len(run_ids), 32)
            self.assertEqual(len(run_dirs), 32)
            self.assertTrue(all(path.is_dir() for path in run_dirs))

    def test_atomic_creation_rejects_an_exact_run_id_collision(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            arguments = {
                "output_root": temporary_directory,
                "experiment_name": "training",
                "dataset": "antm2c",
                "model": "dnn",
                "resolved_config": CONFIG,
                "now": FIXED_TIME,
                "entropy": "same-id",
                "repository_root": temporary_directory,
            }
            create_run_context(**arguments)

            with self.assertRaises(FileExistsError):
                create_run_context(**arguments)

    def test_path_traversal_component_is_rejected(self):
        """Keep untrusted registry labels inside the configured output root."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaises(ValueError):
                create_run_context(
                    output_root=temporary_directory,
                    experiment_name="training",
                    dataset="../private",
                    model="dnn",
                    resolved_config=CONFIG,
                    repository_root=temporary_directory,
                )


if __name__ == "__main__":
    unittest.main()

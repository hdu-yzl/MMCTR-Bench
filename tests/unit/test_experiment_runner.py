import json
import queue
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mmctr.experiments import ExperimentRunner, ExperimentTask


class ExperimentRunnerTests(unittest.TestCase):
    def test_task_identity_is_frozen_when_the_source_config_changes(self):
        config = {"train": {"seed": 1}, "data_fingerprint": "abc"}
        task = ExperimentTask("seed-1", "antm2c", "dnn", 1, config)
        original_key = task.key

        config["train"]["seed"] = 99

        self.assertEqual(1, task.resolved_config["train"]["seed"])
        self.assertEqual(original_key, task.key)

    def test_matrix_isolates_failures_assigns_devices_and_resumes_completed_tasks(self):
        tasks = (
            ExperimentTask(
                task_id="seed-1",
                dataset="antm2c",
                model="dnn",
                seed=1,
                resolved_config={"train": {"seed": 1}, "data_fingerprint": "abc"},
            ),
            ExperimentTask(
                task_id="seed-2",
                dataset="antm2c",
                model="dnn",
                seed=2,
                resolved_config={"train": {"seed": 2}, "data_fingerprint": "abc"},
            ),
        )
        calls = []

        def first_executor(task, context, device):
            calls.append((task.task_id, device))
            if task.seed == 2:
                raise RuntimeError("synthetic failure")
            context.append_metrics({"split": "val", "auc": 0.8})
            return {"val_auc": 0.8, "val_log_loss": 0.5}

        with tempfile.TemporaryDirectory() as directory:
            runner = ExperimentRunner(
                output_root=Path(directory),
                executor=first_executor,
                devices=("cuda:0", "cuda:1"),
                max_workers=2,
                repository_root=Path(directory),
            )
            first = runner.run("sweep", tasks)

            self.assertEqual({"completed", "failed"}, {result.status for result in first})
            self.assertEqual({"cuda:0", "cuda:1"}, {device for _, device in calls})
            self.assertTrue(all(result.artifact_dir.is_dir() for result in first))
            completed = next(result for result in first if result.status == "completed")
            failed = next(result for result in first if result.status == "failed")
            self.assertIn("synthetic failure", failed.error)
            result_payload = json.loads(
                (completed.artifact_dir / "result.json").read_text(encoding="utf-8")
            )
            self.assertEqual(1, result_payload["schema_version"])
            self.assertEqual("abc", result_payload["data_fingerprint"])

            retry_calls = []

            def retry_executor(task, _context, device):
                retry_calls.append((task.task_id, device))
                return {"val_auc": 0.7}

            resumed = ExperimentRunner(
                output_root=Path(directory),
                executor=retry_executor,
                devices=("cuda:0", "cuda:1"),
                max_workers=2,
                repository_root=Path(directory),
            ).run("sweep", tasks)

        self.assertEqual([("seed-2", "cuda:0")], retry_calls)
        self.assertEqual(["completed", "completed"], [result.status for result in resumed])
        resumed_seed_one = next(
            result for result in resumed if result.metadata["task_id"] == "seed-1"
        )
        self.assertEqual(completed.run_id, resumed_seed_one.run_id)

    def test_device_is_returned_if_run_context_creation_fails(self):
        """Return the queue token on setup failure to prevent later workers deadlocking."""
        task = ExperimentTask(
            task_id="context-failure",
            dataset="antm2c",
            model="dnn",
            seed=1,
            resolved_config={"data_fingerprint": "abc"},
        )
        devices = queue.Queue()
        devices.put("cuda:0")
        with tempfile.TemporaryDirectory() as directory:
            runner = ExperimentRunner(Path(directory), lambda *_args: {})
            with mock.patch(
                "mmctr.experiments.runner.create_run_context",
                side_effect=RuntimeError("context failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "context failed"):
                    runner._execute_one("sweep", task, {"tasks": {}}, devices)
        self.assertEqual(1, devices.qsize())
        self.assertEqual("cuda:0", devices.get_nowait())


if __name__ == "__main__":
    unittest.main()

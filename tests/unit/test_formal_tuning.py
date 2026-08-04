import json
import tempfile
import unittest
from pathlib import Path

from mmctr.core import ContractError, RunResult
from mmctr.experiments import TuningTrial, ValidationOnlyTuner


class FormalTuningTests(unittest.TestCase):
    @staticmethod
    def _result(run_id, **metrics):
        return RunResult(run_id=run_id, status="completed", metrics=metrics)

    def test_selection_rejects_test_metrics_and_freezes_provenance_before_final_test(self):
        with tempfile.TemporaryDirectory() as directory:
            tuner = ValidationOnlyTuner(
                output_root=Path(directory),
                study_id="dnn-search",
                experiment_id="sweep-20260803",
                data_fingerprint="dataset-sha",
                seeds=(1, 2),
            )
            contaminated = TuningTrial(
                "trial-bad",
                {"lr": 0.1},
                self._result("bad", val_auc=0.9, test_auc=0.99),
            )
            with self.assertRaisesRegex(ContractError, "test"):
                tuner.freeze((contaminated,))

            trials = (
                TuningTrial(
                    "trial-1",
                    {"lr": 0.01, "dropout": 0.0},
                    self._result("run-1", val_auc=0.70, val_log_loss=0.55),
                ),
                TuningTrial(
                    "trial-2",
                    {"lr": 0.001, "dropout": 0.1},
                    self._result("run-2", val_auc=0.75, val_log_loss=0.60),
                ),
            )
            frozen = tuner.freeze(trials)
            final_task = tuner.final_test_task(
                frozen,
                task_id="final-test",
                dataset="antm2c",
                model="dnn",
                seed=1,
            )
            final_result = self._result("final-run", test_auc=0.74, test_log_loss=0.58)
            result_path = tuner.record_final_test_result(frozen, final_result)

            selection_payload = json.loads(frozen.path.read_text(encoding="utf-8"))
            test_payload = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertEqual("trial-2", frozen.trial_id)
        self.assertEqual({"lr": 0.001, "dropout": 0.1}, dict(frozen.config))
        self.assertEqual((1, 2), frozen.seeds)
        self.assertEqual("dataset-sha", frozen.data_fingerprint)
        self.assertEqual("final_test", final_task.resolved_config["stage"])
        self.assertEqual("trial-2", final_task.resolved_config["selection"]["trial_id"])
        self.assertEqual("trial-2", selection_payload["selected_trial_id"])
        self.assertNotIn("test", json.dumps(selection_payload).lower())
        self.assertEqual("final-run", test_payload["run_id"])


if __name__ == "__main__":
    unittest.main()

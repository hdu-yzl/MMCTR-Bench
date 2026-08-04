import unittest
import json
import tempfile
from pathlib import Path

from mmctr.analysis import (
    ColdStartProtocol,
    load_cold_start_audit,
    load_cold_start_study_config,
    load_cold_start_study_matrix,
    save_cold_start_audit,
    save_cold_start_study_matrix,
)
from mmctr.core import ContractError


class ColdStartProtocolTests(unittest.TestCase):
    def test_verified_audit_builds_versioned_experiment_tasks(self):
        audit = ColdStartProtocol(target="item", regime="zero_shot").audit(
            train_event_ids=("tr-1",),
            train_user_ids=(1,),
            train_item_ids=(10,),
            evaluation_event_ids=("te-1",),
            evaluation_user_ids=(2,),
            evaluation_item_ids=(11,),
        )
        config = """\
dataset: fixture
data_fingerprint: data-v1
data: {name: fixture}
model_configs:
  dnn: {latent_dim: 4}
models: [dnn]
seeds: [3]
audit_manifest: audit.json
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit_path = root / "audit.json"
            config_path = root / "cold-start.yaml"
            matrix_path = root / "matrix.json"
            save_cold_start_audit(audit_path, audit)
            config_path.write_text(config, encoding="utf-8")

            tasks = load_cold_start_study_config(config_path)
            save_cold_start_study_matrix(tasks, matrix_path)
            restored = load_cold_start_study_matrix(matrix_path)

            self.assertEqual([task.key for task in tasks], [task.key for task in restored])
            analysis = tasks[0].resolved_config["analysis"]
            self.assertEqual("cold-start-evaluation-v1", analysis["protocol"])
            self.assertEqual(audit.fingerprint, analysis["audit_fingerprint"])
            self.assertEqual("zero_shot", analysis["regime"])
            payload = json.loads(matrix_path.read_text(encoding="utf-8"))
            self.assertEqual("cold-start-study-matrix-v1", payload["schema"])

    def test_legacy_cold_start_trainer_and_tfrecord_scripts_are_removed(self):
        root = Path(__file__).resolve().parents[2]
        legacy_paths = (
            root / "src/analysis/Trainers_fenxi.py",
            root / "src/analysis/run_all.py",
            root / "src/analysis/cold_start/build_fewshot.py",
            root / "src/analysis/cold_start/build_zeroshot.py",
            root / "src/analysis/cold_start/run_all_fewshot.py",
            root / "src/analysis/cold_start/run_all_zeroshot.py",
            root / "src/scripts/stat_fewshot.py",
            root / "src/scripts/stat_zeroshot.py",
        )
        self.assertEqual([], [path for path in legacy_paths if path.exists()])

    def test_zero_and_few_shot_constraints_are_audited_from_event_sets(self):
        zero = ColdStartProtocol(target="item", regime="zero_shot")
        audit = zero.audit(
            train_event_ids=("tr-1", "tr-2"),
            train_user_ids=(1, 2),
            train_item_ids=(10, 11),
            evaluation_event_ids=("te-1", "te-2"),
            evaluation_user_ids=(1, 3),
            evaluation_item_ids=(12, 13),
        )
        self.assertEqual(2, audit.target_count)
        self.assertEqual(0, audit.maximum_support_interactions)
        self.assertEqual(64, len(audit.fingerprint))
        with self.assertRaisesRegex(ContractError, "zero-shot"):
            zero.audit(
                train_event_ids=("tr-1",),
                train_user_ids=(1,),
                train_item_ids=(10,),
                evaluation_event_ids=("te-1",),
                evaluation_user_ids=(2,),
                evaluation_item_ids=(10,),
            )

        few = ColdStartProtocol(target="user", regime="few_shot", max_support_interactions=2)
        few_audit = few.audit(
            train_event_ids=("tr-1", "tr-2", "tr-3"),
            train_user_ids=(7, 7, 9),
            train_item_ids=(10, 11, 12),
            evaluation_event_ids=("te-1", "te-2"),
            evaluation_user_ids=(7, 9),
            evaluation_item_ids=(13, 14),
        )
        self.assertEqual({7: 2, 9: 1}, dict(few_audit.support_counts))
        with self.assertRaisesRegex(ContractError, "few-shot"):
            ColdStartProtocol(target="user", regime="few_shot", max_support_interactions=1).audit(
                train_event_ids=("tr-1", "tr-2"),
                train_user_ids=(7, 7),
                train_item_ids=(10, 11),
                evaluation_event_ids=("te-1",),
                evaluation_user_ids=(7,),
                evaluation_item_ids=(12,),
            )

    def test_versioned_audit_manifest_round_trips_and_rejects_tampering(self):
        audit = ColdStartProtocol(target="item", regime="zero_shot").audit(
            train_event_ids=("tr-1",),
            train_user_ids=(1,),
            train_item_ids=(10,),
            evaluation_event_ids=("te-1",),
            evaluation_user_ids=(2,),
            evaluation_item_ids=(11,),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cold-start-audit.json"
            save_cold_start_audit(path, audit)
            loaded = load_cold_start_audit(path)
            self.assertEqual(audit, loaded)

            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["audit"]["target_count"] = 99
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "manifest fingerprint"):
                load_cold_start_audit(path)


if __name__ == "__main__":
    unittest.main()

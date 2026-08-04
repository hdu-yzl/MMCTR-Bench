import json
import tempfile
import unittest
from pathlib import Path

import torch

from mmctr.analysis import (
    build_fusion_study_tasks,
    load_fusion_study_config,
    load_fusion_study_matrix,
    save_fusion_study_matrix,
)
from mmctr.core import Batch, ContractError
from mmctr.models import create_model


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class FusionStudyProtocolTests(unittest.TestCase):
    def test_legacy_parallel_model_tree_is_removed(self):
        legacy_tree = REPOSITORY_ROOT / "src/analysis/fusion_analysis"
        self.assertFalse(legacy_tree.is_dir() and any(legacy_tree.rglob("*")))
        self.assertFalse((REPOSITORY_ROOT / "src/analysis/fusion_analysis.py").exists())

    def test_versioned_matrix_round_trip_from_yaml_config(self):
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
            matrix_path = root / "matrix.json"
            config_path.write_text(config, encoding="utf-8")

            tasks = load_fusion_study_config(config_path)
            save_fusion_study_matrix(tasks, matrix_path)
            restored = load_fusion_study_matrix(matrix_path)

            self.assertEqual([task.key for task in tasks], [task.key for task in restored])
            payload = json.loads(matrix_path.read_text(encoding="utf-8"))
            self.assertEqual("fusion-study-matrix-v1", payload["schema"])
            self.assertEqual(4, payload["task_count"])
            self.assertEqual(64, len(payload["fingerprint"]))

    def test_tasks_vary_only_registered_fusion_on_supported_production_models(self):
        model_config = {
            "latent_dim": 4,
            "projection_dim": 4,
            "mlp_dims": [8],
            "dropout": 0.0,
            "batch_norm": False,
        }
        data_config = {
            "name": "fixture",
            "id_feature_num": 30,
            "use_mm_features": ["id", "text"],
            "use_mm_seq_features": ["id", "text"],
            "mm_dims": {"id": 0, "text": 3},
            "mm_seq_dims": {"id": 0, "text": 3},
            "user_features": ["id"],
            "user_features_dim": {"id": 0},
        }
        tasks = build_fusion_study_tasks(
            dataset="fixture",
            data_fingerprint="data-v1",
            data_config=data_config,
            model_configs={"dnn_mm": model_config},
            models=("dnn_mm",),
            fusions=("cat", "mean"),
            seeds=(3, 5),
        )
        self.assertEqual(4, len(tasks))
        self.assertNotIn("modal_fusion_method", model_config)
        self.assertEqual(
            {"concatenate", "mean"},
            {task.resolved_config["analysis"]["fusion"] for task in tasks},
        )

        task = tasks[0]
        model = create_model(
            task.model,
            task.resolved_config["model"],
            task.resolved_config["data"],
        )
        batch = Batch(
            user_features={"id": torch.tensor([[1], [2]])},
            item_features={"id": torch.tensor([[3], [4]]), "text": torch.randn(2, 3)},
            history_features={
                "id": torch.tensor([[0, 3], [3, 4]]),
                "text": torch.randn(2, 2, 3),
            },
            history_mask=torch.tensor([[False, True], [True, True]]),
            labels=torch.tensor([1.0, 0.0]),
        )
        output = model(batch)
        output.logits.sum().backward()
        self.assertEqual((2,), tuple(output.logits.shape))

    def test_paper_private_models_cannot_be_forced_into_approximate_fusion_sweeps(self):
        with self.assertRaisesRegex(ContractError, "supported production models"):
            build_fusion_study_tasks(
                dataset="fixture",
                data_fingerprint="data-v1",
                data_config={},
                model_configs={"marn": {}},
                models=("marn",),
                fusions=("mean",),
                seeds=(1,),
            )


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

import torch

from mmctr.analysis import (
    ActivationCapture,
    AlignmentAuxiliary,
    load_alignment_study_config,
    load_alignment_study_matrix,
    save_alignment_study_matrix,
)
from mmctr.core import ContractError, ModelOutput


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.text = torch.nn.Linear(3, 2, bias=False)
        self.image = torch.nn.Linear(3, 2, bias=False)
        self.head = torch.nn.Linear(4, 1)

    def forward(self, values):
        text = self.text(values)
        image = self.image(values)
        return ModelOutput(logits=self.head(torch.cat((text, image), dim=-1)).squeeze(-1))


class AlignmentProtocolTests(unittest.TestCase):
    def test_study_config_builds_versioned_experiment_tasks(self):
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
            matrix_path = root / "matrix.json"
            config_path.write_text(config, encoding="utf-8")

            tasks = load_alignment_study_config(config_path)
            save_alignment_study_matrix(tasks, matrix_path)
            restored = load_alignment_study_matrix(matrix_path)

            self.assertEqual([task.key for task in tasks], [task.key for task in restored])
            self.assertEqual(4, len(tasks))
            analysis = tasks[-1].resolved_config["analysis"]
            self.assertEqual("representation-alignment-v1", analysis["protocol"])
            self.assertEqual("mse", analysis["method"])
            self.assertEqual(0.3, analysis["weight"])
            self.assertEqual(
                {"image": "image_projector", "text": "text_projector"},
                dict(analysis["representations"]),
            )
            payload = json.loads(matrix_path.read_text(encoding="utf-8"))
            self.assertEqual("alignment-study-matrix-v1", payload["schema"])

    def test_legacy_alignment_trainer_tree_is_removed(self):
        legacy_tree = REPOSITORY_ROOT / "src/analysis/alignment_analysis"
        self.assertFalse(legacy_tree.is_dir() and any(legacy_tree.rglob("*")))

    def test_hooks_capture_existing_modules_and_add_named_auxiliary_loss(self):
        torch.manual_seed(4)
        model = _ToyModel()
        values = torch.randn(5, 3)

        with ActivationCapture(model, {"text": "text", "image": "image"}) as capture:
            original = model(values)
        aligned = AlignmentAuxiliary("cosine", weight=0.3).attach(original, capture.values)

        self.assertTrue(torch.equal(original.logits, aligned.logits))
        self.assertEqual({"alignment_cosine"}, set(aligned.auxiliary_losses))
        self.assertGreaterEqual(float(aligned.auxiliary_losses["alignment_cosine"]), 0.0)
        (aligned.logits.mean() + aligned.auxiliary_loss()).backward()
        self.assertIsNotNone(model.text.weight.grad)
        self.assertIsNotNone(model.image.weight.grad)
        self.assertEqual(0, len(model.text._forward_hooks))

    def test_representation_contract_rejects_missing_or_misaligned_modalities(self):
        output = ModelOutput(logits=torch.zeros(2))
        objective = AlignmentAuxiliary("mse", weight=1.0)
        with self.assertRaisesRegex(ContractError, "at least two"):
            objective.attach(output, {"text": torch.zeros(2, 3)})
        with self.assertRaisesRegex(ContractError, "same shape"):
            objective.attach(
                output,
                {"text": torch.zeros(2, 3), "image": torch.zeros(2, 4)},
            )


if __name__ == "__main__":
    unittest.main()

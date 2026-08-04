import ast
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRAINER_PATH = REPOSITORY_ROOT / "src" / "mmctr" / "training" / "entrypoint.py"


class TrainerRunIsolationGuardTest(unittest.TestCase):
    def test_primary_trainer_creates_run_context_before_runtime_objects(self):
        """Ensure loader or model setup failures still belong to an isolated run record."""
        source = TRAINER_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(TRAINER_PATH))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]

        context_lines = [
            node.lineno
            for node in calls
            if isinstance(node.func, ast.Name) and node.func.id == "create_run_context"
        ]
        runtime_lines = [
            node.lineno
            for node in calls
            if (isinstance(node.func, ast.Name) and node.func.id == "get_data_loader")
            or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in {"get_data_loader", "create_model"}
            )
        ]

        self.assertEqual(len(context_lines), 1)
        self.assertTrue(runtime_lines)
        self.assertLess(context_lines[0], min(runtime_lines))
        self.assertIn("load_training_config", source)
        self.assertIn("PROJECT_ROOT = Path(__file__).resolve().parents[3]", source)
        self.assertNotIn("local_data.yaml", source)
        self.assertNotIn("local_seq_data.yaml", source)
        self.assertIn("self.run_context.checkpoints_dir", source)
        self.assertIn('filename="run.log"', source)
        self.assertIn('self.run_context.finalize("completed"', source)
        self.assertGreaterEqual(source.count('self.run_context.finalize("failed"'), 2)


if __name__ == "__main__":
    unittest.main()

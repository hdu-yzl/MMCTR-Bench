import ast
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TUNER_PATHS = (
    REPOSITORY_ROOT / "src" / "scripts" / "Tuner.py",
    REPOSITORY_ROOT / "src" / "scripts" / "Codebook_Tuner.py",
)


class TunerSplitGuardTest(unittest.TestCase):
    def test_tuners_delegate_selection_and_do_not_evaluate_test_directly(self):
        for path in TUNER_PATHS:
            with self.subTest(path=path.name):
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(path))
                direct_evaluation_lines = [
                    node.lineno
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "evalate"
                ]

                self.assertIn("evaluate_for_selection", source)
                self.assertEqual(direct_evaluation_lines, [])
                self.assertNotIn("test_auc", source)
                self.assertNotIn("test_loss", source)


if __name__ == "__main__":
    unittest.main()

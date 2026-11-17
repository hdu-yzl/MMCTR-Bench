import unittest
from pathlib import Path

import tomli


class LinuxCIWorkflowTest(unittest.TestCase):
    def test_ci_dependency_group_covers_collected_optional_tests(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        pyproject = tomli.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))
        ci_dependencies = pyproject["project"]["optional-dependencies"]["ci"]

        for dependency in ("Pillow", "pandas", "pyarrow", "tensorflow"):
            self.assertTrue(
                any(value.startswith(dependency) for value in ci_dependencies),
                "CI dependency group is missing {}".format(dependency),
            )

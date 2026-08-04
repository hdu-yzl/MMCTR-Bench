import unittest
from pathlib import Path
from typing import Any, Dict, List, Mapping, cast

import yaml


class LinuxCIWorkflowTest(unittest.TestCase):
    def test_linux_cpu_workflow_runs_every_stage_gate(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        workflow_path = project_root / ".github" / "workflows" / "linux-ci.yml"
        self.assertTrue(workflow_path.is_file(), "Linux CI workflow is missing")

        document = cast(
            Dict[str, Any],
            yaml.load(workflow_path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader),
        )
        triggers = cast(Mapping[str, Any], document["on"])
        self.assertEqual(set(triggers), {"pull_request", "push", "workflow_dispatch"})

        jobs = cast(Mapping[str, Any], document["jobs"])
        self.assertEqual(set(jobs), {"quality"})
        quality = cast(Mapping[str, Any], jobs["quality"])
        self.assertEqual(quality["runs-on"], "ubuntu-22.04")

        steps = cast(List[Mapping[str, Any]], quality["steps"])
        actions = {step["uses"] for step in steps if "uses" in step}
        self.assertIn("actions/checkout@v6", actions)
        self.assertIn("actions/setup-python@v6", actions)

        setup_python = next(step for step in steps if step.get("uses") == "actions/setup-python@v6")
        self.assertEqual(cast(Mapping[str, Any], setup_python["with"])["python-version"], "3.8")

        commands = "\n".join(str(step["run"]) for step in steps if "run" in step)
        expected_fragments = (
            "torch==1.13.1+cpu",
            ".[training,ci,dev]",
            "-m pip check",
            "-m ruff format --check .",
            "-m ruff check .",
            "-m mypy src/mmctr",
            "-m pytest tests/unit tests/smoke",
            "-m pytest --cov=mmctr --cov-report=term-missing",
            "-m build",
        )
        for fragment in expected_fragments:
            self.assertIn(fragment, commands)

        environment = cast(Mapping[str, str], document["env"])
        for variable in (
            "PIP_CACHE_DIR",
            "RUFF_CACHE_DIR",
            "MYPY_CACHE_DIR",
            "COVERAGE_FILE",
            "PYTHONPYCACHEPREFIX",
            "TMPDIR",
        ):
            self.assertIn("${{ github.workspace }}", environment[variable])

    def test_mypy_accepts_uninstalled_optional_runtime_dependencies(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        pyproject = (project_root / "pyproject.toml").read_text(encoding="utf-8")

        self.assertIn("[[tool.mypy.overrides]]", pyproject)
        self.assertIn("ignore_missing_imports = true", pyproject)
        for module in ("PIL.*", "matplotlib.*", "pyarrow.*", "tensorflow.*", "transformers.*"):
            self.assertIn('"{}"'.format(module), pyproject)

    def test_ci_dependency_group_covers_collected_optional_tests(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        pyproject = (project_root / "pyproject.toml").read_text(encoding="utf-8")

        self.assertIn("ci = [", pyproject)
        for dependency in ("Pillow", "matplotlib", "pyarrow", "tensorflow"):
            self.assertIn('"{}>'.format(dependency), pyproject)

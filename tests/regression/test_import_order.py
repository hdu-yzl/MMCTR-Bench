import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"


class ImportOrderRegressionTest(unittest.TestCase):
    def _run_in_clean_process(self, source):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(SOURCE_ROOT)
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = subprocess.run(
                [sys.executable, "-c", source],
                cwd=temporary_directory,
                env=environment,
                capture_output=True,
                text=True,
                timeout=60,
            )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)

    def test_helper_import_does_not_eagerly_import_model_packages(self):
        self._run_in_clean_process(
            "import sys\n"
            "import mmctr.utils.helper\n"
            "assert 'models.ctr_models' not in sys.modules\n"
            "assert 'models.mm_ctr_models' not in sys.modules\n"
        )

    def test_helper_and_public_registry_resolve_the_same_canonical_class(self):
        self._run_in_clean_process(
            "from mmctr import __version__\n"
            "from mmctr.models import DNN\n"
            "from mmctr.models.common.registry import resolve_model_class\n"
            "assert __version__ == '0.1.0'\n"
            "assert DNN.__module__ == 'mmctr.models.baseline.dnn'\n"
            "assert resolve_model_class('dnn') is DNN\n"
        )


if __name__ == "__main__":
    unittest.main()

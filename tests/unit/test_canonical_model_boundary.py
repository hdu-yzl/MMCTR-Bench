import ast
import unittest
from pathlib import Path

from mmctr.models.registry import MODEL_REGISTRY
from mmctr.quantization.registry import QUANTIZER_REGISTRY


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"


class CanonicalModelBoundaryTests(unittest.TestCase):
    def test_legacy_model_package_and_tuners_are_not_shipped(self):
        self.assertFalse((SOURCE_ROOT / "models").exists())
        for name in (
            "Tuner.py",
            "Codebook_Tuner.py",
            "Pre_Tuner.py",
            "run_din_autoint.sh",
            "run_four_models_cuda0_3.sh",
        ):
            self.assertFalse((SOURCE_ROOT / "scripts" / name).exists())

    def test_registries_do_not_publish_legacy_resolution_metadata(self):
        for registry in (MODEL_REGISTRY, QUANTIZER_REGISTRY):
            for name in registry.names():
                with self.subTest(registry=registry.kind, name=name):
                    metadata = registry.spec(name).metadata
                    self.assertNotIn("legacy_module", metadata)
                    self.assertNotIn("legacy_symbol", metadata)

    def test_production_source_does_not_import_legacy_models(self):
        offenders = []
        for path in (SOURCE_ROOT / "mmctr").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("models"):
                    offenders.append("{}:{}".format(path.relative_to(SOURCE_ROOT), node.lineno))
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "models" or alias.name.startswith("models."):
                            offenders.append(
                                "{}:{}".format(path.relative_to(SOURCE_ROOT), node.lineno)
                            )
        self.assertEqual([], offenders)

    def test_public_models_export_only_the_canonical_base(self):
        import mmctr.models as public_models

        self.assertNotIn("BaseModel", public_models.__all__)
        self.assertNotIn("LegacyModelAdapter", public_models.__all__)
        self.assertFalse(hasattr(public_models, "BaseModel"))
        self.assertFalse(hasattr(public_models, "LegacyModelAdapter"))


if __name__ == "__main__":
    unittest.main()

import ast
import importlib
import inspect
import unittest
from pathlib import Path

from mmctr.models.common.base import BaseSeqModel
from mmctr.models.common.registry import MODEL_REGISTRY
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

    def test_each_registered_model_has_one_family_module(self):
        id_only = {"autoint", "dcn", "deepfm", "din", "dnn"}
        modules = []
        for name in MODEL_REGISTRY.names():
            with self.subTest(name=name):
                specification = MODEL_REGISTRY.spec(name)
                family = "baseline" if name in id_only else "mm_models"
                expected_module = "mmctr.models.{}.{}".format(family, name)
                self.assertEqual(expected_module, specification.module)
                module = importlib.import_module(specification.module)
                model_classes = [
                    value
                    for _, value in inspect.getmembers(module, inspect.isclass)
                    if value.__module__ == specification.module and issubclass(value, BaseSeqModel)
                ]
                self.assertEqual(
                    [specification.symbol], [value.__name__ for value in model_classes]
                )
                modules.append(specification.module)
        self.assertEqual(len(modules), len(set(modules)))

    def test_models_directory_contains_only_family_and_common_packages(self):
        model_root = SOURCE_ROOT / "mmctr" / "models"
        self.assertEqual(
            {"baseline", "common", "mm_models"},
            {
                path.name
                for path in model_root.iterdir()
                if path.is_dir() and path.name != "__pycache__"
            },
        )
        self.assertEqual(
            {"__init__.py"},
            {path.name for path in model_root.glob("*.py")},
        )


if __name__ == "__main__":
    unittest.main()

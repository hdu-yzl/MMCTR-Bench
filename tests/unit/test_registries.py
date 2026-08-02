import sys
import unittest

from mmctr.core.registry import ComponentRegistry, ComponentSpec, RegistryError
from mmctr.data.registry import DATASET_REGISTRY
from mmctr.models.registry import MODEL_REGISTRY


class RegistryTests(unittest.TestCase):
    def test_public_registries_have_unique_canonical_names(self):
        self.assertEqual(3, len(DATASET_REGISTRY.names()))
        self.assertEqual(25, len(MODEL_REGISTRY.names()))
        self.assertEqual(len(set(MODEL_REGISTRY.names())), len(MODEL_REGISTRY.names()))

    def test_model_alias_resolves_to_one_canonical_name(self):
        self.assertEqual("dnn_mm_seq", MODEL_REGISTRY.canonical_name("dnn_seq"))

    def test_registry_listing_does_not_import_heavy_implementations(self):
        before = set(sys.modules)
        MODEL_REGISTRY.names()
        DATASET_REGISTRY.names()
        imported = set(sys.modules).difference(before)
        self.assertNotIn("models.ctr_models", imported)
        self.assertNotIn("data.dataloaders", imported)

    def test_duplicate_alias_is_rejected(self):
        registry = ComponentRegistry("fixture")
        registry.register(ComponentSpec("first", "builtins", "str", aliases=("old",)))
        with self.assertRaisesRegex(RegistryError, "duplicate"):
            registry.register(ComponentSpec("second", "builtins", "int", aliases=("old",)))


if __name__ == "__main__":
    unittest.main()

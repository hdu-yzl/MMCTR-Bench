import sys
import unittest

from mmctr.core.registry import ComponentRegistry, ComponentSpec, RegistryError
from mmctr.data.registry import DATASET_REGISTRY
from mmctr.models.components.fusion_registry import FUSION_REGISTRY
from mmctr.models.components.pooling_registry import POOLING_REGISTRY
from mmctr.models.registry import MODEL_REGISTRY
from mmctr.quantization.registry import QUANTIZER_REGISTRY


class RegistryTests(unittest.TestCase):
    def test_public_registries_have_unique_canonical_names(self):
        self.assertEqual(3, len(DATASET_REGISTRY.names()))
        self.assertEqual(23, len(MODEL_REGISTRY.names()))
        self.assertEqual(2, len(QUANTIZER_REGISTRY.names()))
        self.assertEqual(6, len(FUSION_REGISTRY.names()))
        self.assertEqual(6, len(POOLING_REGISTRY.names()))
        self.assertEqual(len(set(MODEL_REGISTRY.names())), len(MODEL_REGISTRY.names()))
        self.assertEqual(len(set(QUANTIZER_REGISTRY.names())), len(QUANTIZER_REGISTRY.names()))

    def test_model_alias_resolves_to_one_canonical_name(self):
        self.assertEqual("dnn_mm_seq", MODEL_REGISTRY.canonical_name("dnn_seq"))

    def test_all_datasets_resolve_to_canonical_dataset_loaders(self):
        self.assertEqual(
            "mmctr.data.datasets.antm2c.canonical",
            DATASET_REGISTRY.spec("antm2c").module,
        )
        self.assertEqual(
            "mmctr.data.datasets.microlens.loader",
            DATASET_REGISTRY.spec("microlens").module,
        )
        self.assertEqual(
            "mmctr.data.datasets.tiktok.loader",
            DATASET_REGISTRY.spec("tiktok").module,
        )

    def test_registry_listing_does_not_import_heavy_implementations(self):
        before = set(sys.modules)
        MODEL_REGISTRY.names()
        FUSION_REGISTRY.names()
        QUANTIZER_REGISTRY.names()
        DATASET_REGISTRY.names()
        imported = set(sys.modules).difference(before)
        self.assertNotIn("models.ctr_models", imported)
        self.assertNotIn("data.dataloaders", imported)
        self.assertNotIn("mmctr.quantization.residual", imported)
        self.assertNotIn("mmctr.quantization.psrq", imported)
        self.assertNotIn("mmctr.models.components.fusion", imported)

    def test_duplicate_alias_is_rejected(self):
        registry = ComponentRegistry("fixture")
        registry.register(ComponentSpec("first", "builtins", "str", aliases=("old",)))
        with self.assertRaisesRegex(RegistryError, "duplicate"):
            registry.register(ComponentSpec("second", "builtins", "int", aliases=("old",)))


if __name__ == "__main__":
    unittest.main()

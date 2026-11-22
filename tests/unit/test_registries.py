import unittest

from mmctr.core.registry import ComponentRegistry, ComponentSpec, RegistryError
from mmctr.models.common.components.fusion_registry import FUSION_REGISTRY
from mmctr.models.common.components.pooling_registry import POOLING_REGISTRY
from mmctr.models.common.registry import MODEL_REGISTRY
from mmctr.quantization.registry import QUANTIZER_REGISTRY


class RegistryTests(unittest.TestCase):
    def test_component_registries_expose_expected_catalog_sizes(self):
        self.assertEqual(2, len(QUANTIZER_REGISTRY.names()))
        self.assertEqual(7, len(FUSION_REGISTRY.names()))
        self.assertEqual(6, len(POOLING_REGISTRY.names()))

    def test_model_aliases_resolve_to_one_canonical_name(self):
        self.assertEqual("dnn_mm_seq", MODEL_REGISTRY.canonical_name("dnn_seq"))
        self.assertEqual("psrq", MODEL_REGISTRY.canonical_name("mcca"))
        self.assertEqual("psrq", MODEL_REGISTRY.spec("mcca").name)

    def test_model_and_quantizer_psrq_namespaces_coexist(self):
        self.assertEqual("MCCA", MODEL_REGISTRY.spec("psrq").symbol)
        self.assertEqual("PSRQPretrainer", QUANTIZER_REGISTRY.spec("psrq").symbol)

    def test_duplicate_alias_is_rejected(self):
        registry = ComponentRegistry("fixture")
        registry.register(ComponentSpec("first", "builtins", "str", aliases=("old",)))
        with self.assertRaisesRegex(RegistryError, "duplicate"):
            registry.register(ComponentSpec("second", "builtins", "int", aliases=("old",)))


if __name__ == "__main__":
    unittest.main()

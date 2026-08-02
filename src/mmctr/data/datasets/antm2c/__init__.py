"""AntM2C dataset contract."""

from .schema import (
    ANTM2C_FIELDS,
    AntM2CProtocol,
    FeatureOwner,
    FieldSpec,
    OwnershipAudit,
    audit_item_candidate,
    field_spec,
)
from .history import Interaction, InteractionHistory, assert_no_future_leakage, build_histories
from .item_store import (
    FeatureStoreAudit,
    ItemFeatureStore,
    ItemIndex,
    build_feature_store,
    build_item_index,
)
from .array_store import (
    ARRAY_STORE_SCHEMA_VERSION,
    AntM2CArrayLoader,
    InteractionTable,
    load_array_store,
    write_array_store,
)
from .extraction import (
    BatchEncoder,
    ExtractionInput,
    ExtractionManifest,
    ExtractionShard,
    iter_extracted_features,
    run_batch_extraction,
)


__all__ = [
    "ANTM2C_FIELDS",
    "ARRAY_STORE_SCHEMA_VERSION",
    "AntM2CProtocol",
    "AntM2CArrayLoader",
    "BatchEncoder",
    "ExtractionInput",
    "ExtractionManifest",
    "ExtractionShard",
    "FeatureOwner",
    "FeatureStoreAudit",
    "FieldSpec",
    "Interaction",
    "InteractionHistory",
    "InteractionTable",
    "ItemFeatureStore",
    "ItemIndex",
    "OwnershipAudit",
    "audit_item_candidate",
    "assert_no_future_leakage",
    "build_feature_store",
    "build_histories",
    "build_item_index",
    "field_spec",
    "iter_extracted_features",
    "load_array_store",
    "run_batch_extraction",
    "write_array_store",
]

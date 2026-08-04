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
from .raw import (
    RAW_REQUIRED_FIELDS,
    RawEvent,
    RawReplay,
    iter_raw_events,
    replay_raw_events,
    split_for_timestamp,
)
from .encoders import (
    BertPoolerEncoder,
    CheckpointIdentity,
    ChineseClipImageEncoder,
    fingerprint_checkpoint,
    load_tar_images,
)


__all__ = [
    "ANTM2C_FIELDS",
    "ARRAY_STORE_SCHEMA_VERSION",
    "AntM2CProtocol",
    "AntM2CArrayLoader",
    "BatchEncoder",
    "BertPoolerEncoder",
    "CheckpointIdentity",
    "ChineseClipImageEncoder",
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
    "RAW_REQUIRED_FIELDS",
    "RawEvent",
    "RawReplay",
    "audit_item_candidate",
    "assert_no_future_leakage",
    "build_feature_store",
    "build_histories",
    "build_item_index",
    "field_spec",
    "fingerprint_checkpoint",
    "load_tar_images",
    "iter_extracted_features",
    "iter_raw_events",
    "load_array_store",
    "run_batch_extraction",
    "replay_raw_events",
    "split_for_timestamp",
    "write_array_store",
]

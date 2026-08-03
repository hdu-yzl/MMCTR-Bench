"""Public model components with strict tensor contracts."""

from mmctr.models.components.masking import (
    apply_feature_mask,
    apply_sequence_mask,
    feature_presence,
    masked_softmax,
    validate_sequence_mask,
)
from mmctr.models.components.pooling import (
    AttentionPooling,
    CrossAttentionPooling,
    DinPooling,
    MaxPooling,
    MeanPooling,
    PoolingCapability,
    SequencePooling,
    SumPooling,
)
from mmctr.models.components.pooling_registry import (
    POOLING_REGISTRY,
    available_pooling,
    create_pooling,
    pooling_capabilities,
)
from mmctr.models.components.projection import DimensionAdapter, NamedFeatureProjector


__all__ = [
    "DimensionAdapter",
    "NamedFeatureProjector",
    "AttentionPooling",
    "CrossAttentionPooling",
    "DinPooling",
    "MaxPooling",
    "MeanPooling",
    "POOLING_REGISTRY",
    "PoolingCapability",
    "SequencePooling",
    "SumPooling",
    "apply_feature_mask",
    "apply_sequence_mask",
    "available_pooling",
    "create_pooling",
    "feature_presence",
    "masked_softmax",
    "pooling_capabilities",
    "validate_sequence_mask",
]

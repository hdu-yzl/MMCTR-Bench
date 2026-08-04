"""Public model components with strict tensor contracts."""

from mmctr.models.common.components.fusion import (
    ConcatenateFusion,
    FusionCapability,
    FusionOutput,
    LowRankFusion,
    MAFFusion,
    MTFNFusion,
    MeanFusion,
    ModalityFusion,
    SumFusion,
)
from mmctr.models.common.components.fusion_registry import (
    FUSION_REGISTRY,
    available_fusion,
    create_fusion,
    fusion_capabilities,
)
from mmctr.models.common.components.masking import (
    apply_feature_mask,
    apply_sequence_mask,
    feature_presence,
    masked_softmax,
    validate_sequence_mask,
)
from mmctr.models.common.components.pooling import (
    AttentionPooling,
    CrossAttentionPooling,
    DinPooling,
    MaxPooling,
    MeanPooling,
    PoolingCapability,
    SequencePooling,
    SumPooling,
)
from mmctr.models.common.components.pooling_registry import (
    POOLING_REGISTRY,
    available_pooling,
    create_pooling,
    pooling_capabilities,
)
from mmctr.models.common.components.pipeline import (
    ComponentConfig,
    ModalPipeline,
    ModalPipelineConfig,
    ModalPipelineOutput,
    ModalPipelineSet,
)
from mmctr.models.common.components.projection import DimensionAdapter, NamedFeatureProjector


__all__ = [
    "ConcatenateFusion",
    "ComponentConfig",
    "DimensionAdapter",
    "FUSION_REGISTRY",
    "FusionCapability",
    "FusionOutput",
    "LowRankFusion",
    "MAFFusion",
    "MTFNFusion",
    "ModalPipeline",
    "ModalPipelineConfig",
    "ModalPipelineOutput",
    "ModalPipelineSet",
    "NamedFeatureProjector",
    "MeanFusion",
    "ModalityFusion",
    "AttentionPooling",
    "CrossAttentionPooling",
    "DinPooling",
    "MaxPooling",
    "MeanPooling",
    "POOLING_REGISTRY",
    "PoolingCapability",
    "SequencePooling",
    "SumPooling",
    "SumFusion",
    "apply_feature_mask",
    "apply_sequence_mask",
    "available_pooling",
    "available_fusion",
    "create_fusion",
    "create_pooling",
    "feature_presence",
    "masked_softmax",
    "fusion_capabilities",
    "pooling_capabilities",
    "validate_sequence_mask",
]

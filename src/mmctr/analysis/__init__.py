"""Canonical analysis protocols that operate on production batches and results."""

from .alignment import (
    ALIGNMENT_STUDY_MATRIX_SCHEMA,
    ActivationCapture,
    AlignmentAuxiliary,
    build_alignment_study_tasks,
    load_alignment_study_config,
    load_alignment_study_matrix,
    save_alignment_study_matrix,
)
from .cold_start import (
    COLD_START_STUDY_MATRIX_SCHEMA,
    ColdStartAudit,
    ColdStartProtocol,
    build_cold_start_study_tasks,
    load_cold_start_audit,
    load_cold_start_study_config,
    load_cold_start_study_matrix,
    save_cold_start_audit,
    save_cold_start_study_matrix,
)
from .efficiency import (
    EfficiencyProtocol,
    EfficiencyReport,
    load_efficiency_report,
    save_efficiency_report,
)
from .fusion import (
    FUSION_STUDY_MATRIX_SCHEMA,
    FUSION_STUDY_MODELS,
    build_fusion_study_tasks,
    load_fusion_study_config,
    load_fusion_study_matrix,
    save_fusion_study_matrix,
)
from .plotting import (
    StandardResult,
    load_standard_results,
    render_metric_figure,
    save_figure_provenance,
)
from .robustness import (
    ROBUSTNESS_STUDY_MATRIX_SCHEMA,
    ModalityDropout,
    TransformedDataLoader,
    build_robustness_study_tasks,
    load_robustness_study_config,
    load_robustness_study_matrix,
    save_robustness_study_matrix,
)


__all__ = [
    "ALIGNMENT_STUDY_MATRIX_SCHEMA",
    "ActivationCapture",
    "AlignmentAuxiliary",
    "COLD_START_STUDY_MATRIX_SCHEMA",
    "ColdStartAudit",
    "ColdStartProtocol",
    "EfficiencyProtocol",
    "EfficiencyReport",
    "FUSION_STUDY_MATRIX_SCHEMA",
    "FUSION_STUDY_MODELS",
    "load_cold_start_audit",
    "load_efficiency_report",
    "build_fusion_study_tasks",
    "build_alignment_study_tasks",
    "build_cold_start_study_tasks",
    "load_alignment_study_config",
    "load_alignment_study_matrix",
    "load_fusion_study_config",
    "load_fusion_study_matrix",
    "load_cold_start_study_config",
    "load_cold_start_study_matrix",
    "ModalityDropout",
    "ROBUSTNESS_STUDY_MATRIX_SCHEMA",
    "save_cold_start_audit",
    "save_cold_start_study_matrix",
    "save_alignment_study_matrix",
    "save_efficiency_report",
    "save_fusion_study_matrix",
    "save_figure_provenance",
    "StandardResult",
    "load_standard_results",
    "render_metric_figure",
    "TransformedDataLoader",
    "build_robustness_study_tasks",
    "load_robustness_study_config",
    "load_robustness_study_matrix",
    "save_robustness_study_matrix",
]

"""Single-setting diagnostics and efficiency reporting."""

from .efficiency import (
    EfficiencyProtocol,
    EfficiencyReport,
    load_efficiency_report,
    save_efficiency_report,
)


__all__ = [
    "EfficiencyProtocol",
    "EfficiencyReport",
    "load_efficiency_report",
    "save_efficiency_report",
]

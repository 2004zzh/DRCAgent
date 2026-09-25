"""Phase 4-R repair-kernel adapters.

This package consumes frozen Phase 2/P4-A truth infrastructure and never
mutates the failed P4-B provenance package.
"""

from .current_context import CurrentSemanticContextBuilder
from .identity import build_debt_marker_groups
from .models import (
    CurrentDebtBinding,
    CurrentSemanticContext,
    CurrentViolationIdentity,
    DebtMarkerGroup,
    RebindingMode,
    SemanticFidelity,
    SourceLineageRecord,
)
from .rebinding import CurrentDebtRebinder

__all__ = [
    "CurrentDebtBinding",
    "CurrentDebtRebinder",
    "CurrentSemanticContext",
    "CurrentSemanticContextBuilder",
    "CurrentViolationIdentity",
    "DebtMarkerGroup",
    "RebindingMode",
    "SemanticFidelity",
    "SourceLineageRecord",
    "build_debt_marker_groups",
]

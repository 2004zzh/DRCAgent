"""Formal research-runtime integration for the verified repair kernel.

The package is intentionally an adapter layer.  It does not alter the
Phase-3/Phase-4 repair implementations or the Phase-2 evaluator; it turns a
formal Region context and a symbolic planner decision into the existing
``RepairCandidate`` IR consumed by the candidate graph and coordinator.
"""

from .audit import (
    IntegrationAudit,
    integration_code_digest,
    formal_wiring_report,
    legacy_path_audit,
    production_hardcode_audit,
)
from .candidate_adapter import (
    KERNEL_GENERATOR, candidate_physical_fingerprint, ensure_noop,
    is_kernel_candidate,
)
from .context import FormalKernelContext, build_formal_kernel_context
from .executable_model_builder import (
    ExecutableRepairModelBuilder,
    model_matches_current_snapshot,
    source_snapshot_identity,
)
from .executable_solver import (
    ExecutableRootRealization,
    ExecutableRootSolver,
    ExecutableSolveAudit,
    ExecutableSolveResult,
)
from .facade import FormalRepairKernel
from .gate import (
    FormalIntegrationGateError,
    gate_contract_path,
    verify_formal_integration_gate,
)
from .llm_planner import KernelRepairPlan, SymbolicKernelPlanner
from .models import KernelRegionResult, KernelSupportReport, SupportStatus
from .trajectory_adapter import (
    CondensedTrajectory,
    TrajectoryStep,
    condense_trajectory,
)

__all__ = [
    "CondensedTrajectory",
    "ExecutableRepairModelBuilder",
    "ExecutableRootRealization",
    "ExecutableRootSolver",
    "ExecutableSolveAudit",
    "ExecutableSolveResult",
    "FormalIntegrationGateError",
    "FormalKernelContext",
    "FormalRepairKernel",
    "IntegrationAudit",
    "integration_code_digest",
    "KernelRegionResult",
    "KernelRepairPlan",
    "KernelSupportReport",
    "SupportStatus",
    "SymbolicKernelPlanner",
    "TrajectoryStep",
    "build_formal_kernel_context",
    "KERNEL_GENERATOR",
    "candidate_physical_fingerprint",
    "is_kernel_candidate",
    "condense_trajectory",
    "ensure_noop",
    "gate_contract_path",
    "model_matches_current_snapshot",
    "source_snapshot_identity",
    "formal_wiring_report",
    "legacy_path_audit",
    "production_hardcode_audit",
    "verify_formal_integration_gate",
]

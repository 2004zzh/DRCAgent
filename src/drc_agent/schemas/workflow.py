from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, TypedDict

from pydantic import Field

from .action import Transaction, VerificationResult
from .common import ArtifactRef, StrictModel


class DesignSnapshotRef(StrictModel):
    """Immutable design state plus its immediate-parent provenance edge.

    ``baseline`` is an oracle/root concept, not the implicit parent of every
    later snapshot.  A non-root snapshot that carries a compiler receipt also
    carries the exact parent source artifact against which that receipt was
    produced.  The optional defaults keep historical manifests readable; P5
    production consumers validate the complete contract before use.
    """

    snapshot_id: str
    script_ref: ArtifactRef
    gds_ref: ArtifactRef | None = None
    drc_ref: ArtifactRef | None = None
    connectivity_ref: ArtifactRef | None = None
    lineage_receipt_ref: ArtifactRef | None = None
    parent_snapshot_id: str | None = None
    parent_script_ref: ArtifactRef | None = None
    provenance_run_id: str | None = None
    provenance_relation_source: str | None = None
    score: tuple[int, int, int] | None = None


class AgentSubgraphRef(StrictModel):
    subgraph_id: str
    artifact_ref: ArtifactRef


class HelperRegionAssignment(StrictModel):
    """Snapshot-bound reason that a helper may assist a current target."""

    helper_region_id: str
    assisted_region_ids: list[str] = Field(min_length=1)
    assisted_target_violation_ids: list[str] = Field(min_length=1)
    dependency_evidence_ids: list[str] = Field(min_length=1)
    helper_potential_dof_capability_ids: list[str] = Field(default_factory=list)
    helper_potential_carrier_ids: list[str] = Field(default_factory=list)
    helper_protected_relation_ids: list[str] = Field(default_factory=list)
    snapshot_id: str
    status: Literal["DEPENDENCY_ASSIST", "UNSUPPORTED_WITH_EVIDENCE"] = (
        "DEPENDENCY_ASSIST"
    )


class ActiveWindow(StrictModel):
    window_id: str
    iteration: int
    window_index: int
    base_snapshot_id: str
    subgraph_ids: list[str]
    region_ids: list[str]
    frontier_violation_ids: list[str]
    claimed_frontier_violation_ids: list[str] = Field(default_factory=list)
    decision_region_ids: list[str] = Field(default_factory=list)
    helper_region_ids: list[str] = Field(default_factory=list)
    helper_assignments: list[HelperRegionAssignment] = Field(
        default_factory=list
    )
    context_only_region_ids: list[str] = Field(default_factory=list)
    boundary_dependency_ids: list[str] = Field(default_factory=list)
    hard_dependency_closure: bool = True


class FrontierState(StrEnum):
    PENDING_NOT_YET_ADMITTED = "PENDING_NOT_YET_ADMITTED"
    ATTEMPTED_THIS_ITERATION = "ATTEMPTED_THIS_ITERATION"
    DEFERRED_BUDGET = "DEFERRED_BUDGET"
    UNSUPPORTED_WITH_EVIDENCE = "UNSUPPORTED_WITH_EVIDENCE"
    REMOVED_BY_COMMIT = "REMOVED_BY_COMMIT"


class WindowResult(StrictModel):
    window_id: str
    iteration: int
    window_index: int
    base_snapshot_id: str
    final_snapshot_id: str
    status: Literal[
        "COMMIT", "ROLLBACK", "NO_SAFE_BUNDLE", "NO_OP", "FAILED"
    ]
    claimed_frontier_violation_ids: list[str]
    selected_candidate_ids: list[str] = Field(default_factory=list)
    admitted_frontier_violation_ids: list[str] = Field(default_factory=list)
    llm_started_frontier_violation_ids: list[str] = Field(default_factory=list)
    plan_validated_frontier_violation_ids: list[str] = Field(default_factory=list)
    kernel_attempted_frontier_violation_ids: list[str] = Field(default_factory=list)
    physical_evaluated_frontier_violation_ids: list[str] = Field(default_factory=list)
    attempted_frontier_violation_ids: list[str] = Field(default_factory=list)
    deferred_frontier_violation_ids: list[str] = Field(default_factory=list)
    unsupported_frontier_violation_ids: list[str] = Field(default_factory=list)
    removed_by_commit_violation_ids: list[str] = Field(default_factory=list)
    frontier_state_counts: dict[str, int] = Field(default_factory=dict)
    frontier_states: dict[str, FrontierState] = Field(default_factory=dict)
    decision_region_ids: list[str] = Field(default_factory=list)
    helper_region_ids: list[str] = Field(default_factory=list)
    context_only_region_ids: list[str] = Field(default_factory=list)
    total_region_count: int = 0
    llm_invocation_count: int = 0

    removed_original: int | None = 0
    new_violations: int | None = 0
    connectivity_preserved: bool | None = None
    joint_sandbox_rounds: int = 0
    no_good_count: int = 0


class IterationFinalEvidencePointer(StrictModel):
    """Unambiguous final state; legacy iteration aliases remain first-window."""

    schema_version: Literal["1.0"] = "1.0"
    iteration: int
    window_count: int
    final_window_id: str | None = None
    final_window_index: int | None = None
    final_window_status: str | None = None
    final_snapshot: DesignSnapshotRef
    final_total_drv: int
    last_committed_window_id: str | None = None
    last_committed_window_index: int | None = None
    final_master_attempt_pointer_ref: ArtifactRef | None = None
    legacy_iteration_alias_semantics: Literal[
        "FIRST_WINDOW_COMPATIBILITY_ONLY"
    ] = "FIRST_WINDOW_COMPATIBILITY_ONLY"


class SubgraphPlanResult(StrictModel):
    subgraph_id: str
    bundle_ref: ArtifactRef | None = None
    status: str
    errors: list[str] = Field(default_factory=list)


class TransactionBatch(StrictModel):
    batch_id: str
    base_snapshot_id: str | None = None
    candidate_ids: list[str] = Field(default_factory=list)
    non_noop_action_count: int = 0
    modified_object_count: int = 0
    requires_replan_after_commit: bool = True
    bundle_ids: list[str]


class StoppingStatus(StrEnum):
    CONTINUE = "CONTINUE"
    DRC_CLEAN = "DRC_CLEAN"
    MAX_ITERATIONS = "MAX_ITERATIONS"
    NO_PROGRESS = "NO_PROGRESS"
    ALL_NOOP = "ALL_NOOP"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    INTEGRITY_FAILURE = "INTEGRITY_FAILURE"


class IterationTransactionStatus(StrEnum):
    """Canonical status shared by runtime and iteration artifacts.

    This is intentionally distinct from the internal TransactionStatus:
    iterations can finish before a transaction exists (NO_OP/NO_PROGRESS),
    while transactions have lifecycle states such as PREPARED and VERIFIED.
    """

    BASELINE = "BASELINE"
    COMMIT = "COMMIT"
    ROLLBACK = "ROLLBACK"
    NO_OP = "NO_OP"
    NO_PROGRESS = "NO_PROGRESS"
    FAILED = "FAILED"


class WorkflowError(StrictModel):
    error_id: str
    node: str
    code: str
    message: str
    created_at: datetime


def merge_by_subgraph_id(left: list[SubgraphPlanResult], right: list[SubgraphPlanResult]) -> list[SubgraphPlanResult]:
    merged = {item.subgraph_id: item for item in [*left, *right]}
    return [merged[key] for key in sorted(merged)]


def append_dedup(left: list, right: list) -> list:
    result = []
    seen = set()
    for item in [*left, *right]:
        identity = getattr(item, "verification_id", None) or getattr(item, "error_id", None) or repr(item)
        if identity not in seen:
            result.append(item)
            seen.add(identity)
    return result


class GlobalState(TypedDict, total=False):
    run_id: str
    experiment_id: str
    case_id: str
    backend_type: str
    iteration: int
    config_ref: ArtifactRef
    manifest_ref: ArtifactRef
    baseline_snapshot: DesignSnapshotRef
    current_snapshot: DesignSnapshotRef
    best_verified_snapshot: DesignSnapshotRef
    design_state_ref: ArtifactRef
    violations_ref: ArtifactRef
    regions_ref: ArtifactRef
    agent_graph_ref: ArtifactRef
    active_subgraphs: list[AgentSubgraphRef]
    active_window: ActiveWindow | None
    window_result: WindowResult | None
    subgraph_results: Annotated[list[SubgraphPlanResult], merge_by_subgraph_id]
    selected_batches: list[TransactionBatch]
    current_transaction: Transaction | None
    verification_results: Annotated[list[VerificationResult], append_dedup]
    iteration_history_ref: ArtifactRef
    metrics_ref: ArtifactRef
    stopping_status: StoppingStatus
    errors: Annotated[list[WorkflowError], append_dedup]

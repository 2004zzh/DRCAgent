from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import Field

from drc_agent.schemas.action import DesignState, PatchPlan, RepairCandidate
from drc_agent.schemas.common import ArtifactRef, StrictModel, stable_hash
from drc_agent.schemas.experience import RepairBlueprint
from drc_agent.schemas.state import (
    AgentSubgraph, LayoutObject, NeighborMessage, RegionState, RuleCatalog,
    ViolationRecord,
)
from drc_agent.schemas.workflow import DesignSnapshotRef


class SupportStatus(StrEnum):
    SUPPORTED_DIRECT = "SUPPORTED_DIRECT"
    SUPPORTED_TRAJECTORY = "SUPPORTED_TRAJECTORY"
    SUPPORTED_REVIEWED_RULE = "SUPPORTED_REVIEWED_RULE"
    SUPPORTED_EXACT = "SUPPORTED_EXACT"
    SUPPORTED_REVIEWED = "SUPPORTED_REVIEWED"
    INSUFFICIENT_FIDELITY = "INSUFFICIENT_FIDELITY"
    RULE_NOT_LIVE_QUALIFIED = "RULE_NOT_LIVE_QUALIFIED"
    UNSUPPORTED_REPAIR_FAMILY = "UNSUPPORTED_REPAIR_FAMILY"


class FormalKernelExecutionContext(StrictModel):
    """Complete immutable root state supplied by the formal runtime."""

    run_id: str
    case_id: str
    iteration: int
    baseline_snapshot: DesignSnapshotRef
    current_snapshot: DesignSnapshotRef
    rule_deck_path: str
    evaluator_hash: str | None = None
    subgraph: AgentSubgraph
    region: RegionState
    violations: list[ViolationRecord]
    objects: list[LayoutObject]
    design: DesignState
    catalog: RuleCatalog
    script: str
    neighbor_messages: list[NeighborMessage] = Field(default_factory=list)
    blueprint: RepairBlueprint | None = None
    failure_memory: list[dict[str, Any]] = Field(default_factory=list)
    planning_target_violation_ids: list[str] | None = None
    planning_role: Literal["TARGET_REPAIR", "DEPENDENCY_ASSIST"] = "TARGET_REPAIR"
    assisted_region_ids: list[str] = Field(default_factory=list)
    assisted_target_violation_ids: list[str] = Field(default_factory=list)
    helper_dependency_evidence_ids: list[str] = Field(default_factory=list)
    helper_potential_dof_capability_ids: list[str] = Field(default_factory=list)
    helper_potential_carrier_ids: list[str] = Field(default_factory=list)
    helper_protected_relation_ids: list[str] = Field(default_factory=list)
    context_fingerprint: str
    execution_epoch: str = "epoch_initial"
    window_id: str | None = None
    window_index: int | None = None
    artifact_prefix: str | None = None


class FormalRootProposal(StrictModel):
    proposal_id: str
    executable_binding: dict[str, Any] = Field(default_factory=dict)
    rule_id: str
    conceptual_family: str
    target_violation_id: str
    candidate: RepairCandidate
    patch_plan: PatchPlan
    predicted_target_effect: str
    physical_effect_fingerprint: str
    authority_modes: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)


class KernelSupportReport(StrictModel):
    """A per-region, auditable support decision.

    Support is derived from the reviewed ``RulePredicateIR`` relation type and
    witness fidelity.  Catalog prose is deliberately not used as a routing
    authority.
    """

    rule_id: str
    violation_ids: list[str] = Field(default_factory=list)
    repair_family: str
    status: SupportStatus
    predicate_fidelity: str = "UNAVAILABLE"
    witness_fidelity: str = "UNAVAILABLE"
    reason_codes: list[str] = Field(default_factory=list)


class KernelInvocationAudit(StrictModel):
    mode: str = "formal_v1"
    version: str = "formal-repair-kernel-v1"
    integration_code_digest: str = ""
    invocation_count: int = 0
    supported_invocation_count: int = 0
    unsupported_noop_count: int = 0
    symbolic_llm_plan_count: int = 0
    llm_plan_requested_count: int = 0
    llm_plan_received_count: int = 0
    llm_plan_validated_count: int = 0
    deterministic_plan_count: int = 0
    llm_plan_failed_count: int = 0
    llm_final_plan_count: int = 0
    llm_plan_binding_pass_count: int = 0
    llm_plan_binding_fail_count: int = 0
    llm_plan_revision_count: int = 0
    execution_feedback_count: int = 0
    llm_execution_feedback_revision_count: int = 0
    dof_execution_revision_count: int = 0
    depth_extension_approval_count: int = 0
    llm_bound_candidate_count: int = 0
    trajectory_search_count: int = 0
    condensed_candidate_count: int = 0
    legacy_candidate_intent_call_count: int = 0
    root_proposal_count: int = 0
    trajectory_live_call_count: int = 0
    temporary_debt_state_count: int = 0
    current_debt_rebind_count: int = 0
    final_clean_trajectory_count: int = 0
    master_kernel_commit_count: int = 0
    condensation_attempt_count: int = 0
    condensation_equivalence_pass_count: int = 0
    legacy_lowerer_call_count: int = 0
    legacy_repair_programmer_call_count: int = 0
    candidate_graph_kernel_candidate_count: int = 0
    cp_sat_selected_kernel_candidate_count: int = 0
    master_transaction_kernel_candidate_count: int = 0
    events: list[dict[str, Any]] = Field(default_factory=list)

    def record(self, event: str, **details: Any) -> None:
        self.events.append({"event": event, **details})


class KernelRegionResult(StrictModel):
    region_id: str
    subgraph_id: str
    planning_path: Literal["REPAIR_KERNEL"] = "REPAIR_KERNEL"
    diagnosis: dict[str, Any] | None = None
    repair_focus_refs: list[str] = Field(default_factory=list)
    repair_scene_refs: list[str] = Field(default_factory=list)
    repair_dof_refs: list[str] = Field(default_factory=list)
    symbolic_plan_refs: list[str] = Field(default_factory=list)
    final_plan_id: str | None = None
    plan_origin: Literal[
        "REAL_LLM",
        "FROZEN_REAL_LLM_REPLAY",
        "FAKE_LLM",
        "DETERMINISTIC_TEST",
    ] | None = None
    plan_binding_report: dict[str, Any] | None = None
    candidates: list[RepairCandidate] = Field(default_factory=list)
    noop_candidate: RepairCandidate | None = None
    trajectory_refs: list[str] = Field(default_factory=list)
    kernel_evidence_refs: list[ArtifactRef] = Field(default_factory=list)
    failures: list[dict[str, Any]] = Field(default_factory=list)
    supported_rule_ids: list[str] = Field(default_factory=list)
    unsupported_rule_ids: list[str] = Field(default_factory=list)
    support_reports: list[KernelSupportReport] = Field(default_factory=list)
    scenes: list[dict[str, Any]] = Field(default_factory=list)
    dofs: list[dict[str, Any]] = Field(default_factory=list)
    plans: list[dict[str, Any]] = Field(default_factory=list)
    execution_feedback: list[dict[str, Any]] = Field(default_factory=list)
    planning_context_hash: str

    @property
    def all_candidates(self) -> list[RepairCandidate]:
        return [*self.candidates, *([self.noop_candidate] if self.noop_candidate else [])]


class FormalIntegrationSummary(StrictModel):
    """Stable manifest payload shared by planner and runtime."""

    mode: str
    version: str
    integration_code_digest: str
    phase4r2_contract_sha256: str
    phase4r2_policy_hash: str | None = None
    supported_families: list[str] = Field(default_factory=list)
    legacy_fallback_for_supported: bool = False

    @property
    def fingerprint(self) -> str:
        return stable_hash(self.model_dump(mode="json"))

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field

from drc_agent.repair_kernel_multistep.models import SearchSnapshot
from drc_agent.schemas.action import PatchPlan, RepairCandidate
from drc_agent.schemas.common import Box, StrictModel


class ObligationKind(StrEnum):
    ROOT_TARGET = "ROOT_TARGET"
    ROOT_TARGET_PERSISTENT_VARIANT = "ROOT_TARGET_PERSISTENT_VARIANT"
    TEMPORARY_DEBT = "TEMPORARY_DEBT"


class Attribution(StrEnum):
    ROOT = "ROOT"
    DIRECT_EDIT_COLLATERAL = "DIRECT_EDIT_COLLATERAL"
    LIKELY_LOCAL_COLLATERAL = "LIKELY_LOCAL_COLLATERAL"
    UNKNOWN = "UNKNOWN"


class DebtRelationType(StrEnum):
    ROOT_RELATION = "ROOT_RELATION"
    MINIMUM_WIDTH = "MINIMUM_WIDTH"
    SAME_LAYER_SPACING = "SAME_LAYER_SPACING"
    OPPOSITE_SIDE_CLEARANCE = "OPPOSITE_SIDE_CLEARANCE"
    VIA_METAL_INSIDE = "VIA_METAL_INSIDE"
    VIA_ENCLOSURE = "VIA_ENCLOSURE"
    UNKNOWN = "UNKNOWN"


class RepairObligation(StrictModel):
    obligation_id: str
    kind: ObligationKind
    origin_step: int = Field(ge=0)
    rule_id: str
    rule_family: str
    relation_type: DebtRelationType
    violation_id: str
    violation_fingerprint: str
    marker_bbox: Box
    attribution: Attribution
    priority: int = Field(ge=0)
    supported_solver_family: bool
    reason_codes: list[str] = Field(default_factory=list)


class ProtectedRelation(StrictModel):
    relation_id: str
    rule_id: str
    root_target: bool = False
    established_depth: int = Field(ge=0)
    violation_fingerprint: str


class TemporaryDebt(StrictModel):
    debt_id: str
    obligation_id: str
    violation_fingerprint: str
    rule_id: str
    marker_bbox: Box
    relation_type: DebtRelationType
    supported: bool
    within_trajectory_halo: bool


class RepairState(StrictModel):
    state_id: str
    snapshot_id: str
    root_target_present_or_persistent: bool
    obligations: list[RepairObligation]
    protected_relations: list[ProtectedRelation] = Field(default_factory=list)
    temporary_debts: list[TemporaryDebt] = Field(default_factory=list)
    unrelated_baseline_violation_count: int = Field(ge=0)
    oracle_status: str
    oracle_reason_codes: list[str] = Field(default_factory=list)


class PlanStrategy(StrEnum):
    SATISFY_ROOT_RELATION = "SATISFY_ROOT_RELATION"
    EXPAND_OPPOSITE_EDGE = "EXPAND_OPPOSITE_EDGE"
    MOVE_SECONDARY_PARTICIPANT = "MOVE_SECONDARY_PARTICIPANT"
    MOVE_WHOLE_VIA_STACK = "MOVE_WHOLE_VIA_STACK"
    EXTEND_ENCLOSER = "EXTEND_ENCLOSER"
    SPECIALIZE_TARGET_INSTANCE_LAYER = "SPECIALIZE_TARGET_INSTANCE_LAYER"
    SPECIALIZE_TARGET_INSTANCE_EDGE = "SPECIALIZE_TARGET_INSTANCE_EDGE"
    LOCAL_FRAGMENT_IF_TARGET_OWNED = "LOCAL_FRAGMENT_IF_TARGET_OWNED"


class SymbolicPlan(StrictModel):
    plan_id: str
    active_obligation_id: str
    strategy: PlanStrategy
    goal_relation: DebtRelationType
    allowed_carrier_types: list[str]
    forbidden_carrier_types: list[str] = Field(default_factory=list)
    protected_relation_ids: list[str] = Field(default_factory=list)
    max_disturbance_dbu: int = Field(ge=0)
    expected_effect: str


class SolverProposal(StrictModel):
    proposal_id: str
    plan_id: str
    active_obligation_id: str
    candidate: RepairCandidate
    patch_plan: PatchPlan
    predicted_target_effect: Literal["NO_EFFECT", "PROGRESS", "SATISFIED"]
    predicted_temporary_debt_rule_ids: list[str] = Field(default_factory=list)
    protected_relation_break_ids: list[str] = Field(default_factory=list)
    physical_effect_fingerprint: str
    touched_instance_anchor_ids: list[str] = Field(default_factory=list)
    authority_legal: bool = True
    source_fresh: bool = True
    within_halo: bool = True


class PredicateProbeResult(StrictModel):
    proposal_id: str
    status: str
    passed: bool
    reason_codes: list[str]
    predicted_debt_count: int = Field(ge=0)


class SearchBounds(StrictModel):
    max_depth: int = Field(default=4, ge=1, le=4)
    beam_width: int = Field(default=4, ge=1, le=4)
    max_children_per_node: int = Field(default=4, ge=1, le=4)
    max_live_sandbox_evals_per_root: int = Field(default=32, ge=1, le=32)
    max_root_relative_new_drv: int = Field(default=6, ge=0)
    max_new_drv_growth_per_step: int = Field(default=4, ge=0)
    max_touched_source_anchors: int = Field(default=4, ge=1)
    max_modified_instances: int = Field(default=2, ge=1)
    max_duplicate_states: int = 1
    max_no_progress_expansions: int = Field(default=4, ge=0)


class SearchEdge(StrictModel):
    expansion_id: str
    parent_node_id: str
    child_node_id: str | None = None
    depth: int
    plan_id: str | None = None
    proposal_id: str | None = None
    physical_effect_fingerprint: str | None = None
    funnel_record_ids: list[str] = Field(default_factory=list)
    phase2_classification: str | None = None
    phase4_classification: str | None = None
    status: str
    reason_codes: list[str] = Field(default_factory=list)


class SearchNode(StrictModel):
    node_id: str
    snapshot: SearchSnapshot
    score: tuple[int, int, int, int, int, int, str]
    status: str


class SearchOutcome(StrictModel):
    root_sample_id: str
    policy_revision: str
    status: str
    final_snapshot: SearchSnapshot | None = None
    selected_edge_ids: list[str] = Field(default_factory=list)
    nodes: list[SearchNode] = Field(default_factory=list)
    edges: list[SearchEdge] = Field(default_factory=list)
    funnel_event_count: int = Field(ge=0)
    sandbox_calls: int = Field(ge=0)
    deepest_failure_stage: str | None = None
    failure_codes: list[str] = Field(default_factory=list)

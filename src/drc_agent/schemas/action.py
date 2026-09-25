from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from .common import ArtifactRef, Box, EditabilityClass, Point, Predicate, StrictModel, Vector
from .state import ResourceClaim
from .tools import AttemptTerminalStatus, EvidenceValidity


class ActionType(StrEnum):
    NO_OP = "NO_OP"
    MOVE_SHAPE = "MOVE_SHAPE"
    RESIZE_SHAPE = "RESIZE_SHAPE"
    ADJUST_ENDPOINT = "ADJUST_ENDPOINT"
    ADD_POLYGON = "ADD_POLYGON"
    DELETE_POLYGON = "DELETE_POLYGON"
    MOVE_VIA_STACK = "MOVE_VIA_STACK"
    ADD_VIA_STACK = "ADD_VIA_STACK"
    DELETE_VIA_STACK = "DELETE_VIA_STACK"
    REPLACE_VIA_STACK = "REPLACE_VIA_STACK"
    CHANGE_LAYER = "CHANGE_LAYER"
    LOCAL_DETOUR = "LOCAL_DETOUR"
    LOCAL_REROUTE = "LOCAL_REROUTE"
    INSTANCE_SPECIALIZATION = "INSTANCE_SPECIALIZATION"


class EndpointSelector(StrictModel):
    endpoint: Literal["START", "END"]
    axis: Literal["X", "Y"] | None = None


class EditProvenance(StrictModel):
    generator: str
    intent_id: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class InstanceGeometrySpecialization(StrictModel):
    """Bounded copy-on-write replacement for one cell occurrence."""

    instance_anchor_id: str
    parent_cell: str
    source_cell: str
    specialized_cell_name: str
    layer: str
    rotation: int = Field(ge=0, le=3)
    mirror: bool
    dx_dbu: int
    dy_dbu: int
    original_polygons_local_dbu: list[list[Point]] = Field(min_length=1)
    replacement_polygons_local_dbu: list[list[Point]] = Field(min_length=1)
    original_polygons_by_layer_local_dbu: dict[str, list[list[Point]]] = Field(
        default_factory=dict,
    )
    complete_polygons_by_layer_local_dbu: dict[str, list[list[Point]]] = Field(
        default_factory=dict,
    )
    replaced_layers: list[str] = Field(default_factory=list, max_length=4)
    occurrence_local_fragment: bool = False


class LayoutEdit(StrictModel):
    edit_id: str
    op: ActionType
    target_object_ids: list[str]
    source_anchor_ids: list[str]
    layer_before: str | None = None
    layer_after: str | None = None
    geometry_before_dbu: list[list[Point]] | None = None
    geometry_after_dbu: list[list[Point]] | None = None
    replacement_group_id: str | None = None
    delta_dbu: Vector | None = None
    endpoint_selector: EndpointSelector | None = None
    net_id: str | None = None
    preconditions: list[Predicate] = Field(default_factory=list)
    postconditions: list[Predicate] = Field(default_factory=list)
    editability_required: set[EditabilityClass] = Field(default_factory=set)
    instance_specializations: list[InstanceGeometrySpecialization] = Field(
        default_factory=list, max_length=4,
    )
    provenance: EditProvenance


class SourceTarget(StrictModel):
    source_anchor_id: str
    source_hash: str
    object_id: str


class TimingDelta(StrictModel):
    setup_ps: int | None = None
    hold_ps: int | None = None
    fidelity: Literal["STA", "PROXY", "UNAVAILABLE"] = "UNAVAILABLE"


class SideEffect(StrictModel):
    kind: str
    severity_milli: int = 0
    evidence_ids: list[str] = Field(default_factory=list)


class DisturbanceEstimate(StrictModel):
    modified_objects: int = 0
    area_dbu2: int = 0
    via_operations: int = 0
    explicit_risk_milli: int = 0

    @property
    def objective_value(self) -> int:
        return (self.modified_objects * 1_000_000 + self.area_dbu2 +
                self.via_operations * 100_000 + self.explicit_risk_milli)


class CostEstimate(StrictModel):
    patch_operations: int = 0
    preview_calls: int = 0
    runtime_millis: int = 0


class CandidateValidationStatus(StrEnum):
    VALID = "VALID"
    VALID_REQUIRES_SANDBOX = "VALID_REQUIRES_SANDBOX"
    INVALID_SCHEMA = "INVALID_SCHEMA"
    INVALID_TARGET = "INVALID_TARGET"
    INVALID_EDITABILITY = "INVALID_EDITABILITY"
    INVALID_GEOMETRY = "INVALID_GEOMETRY"
    INVALID_RESOURCE = "INVALID_RESOURCE"
    INVALID_BENEFIT_PROOF = "INVALID_BENEFIT_PROOF"
    STALE_TARGET = "STALE_TARGET"
    UNPATCHABLE = "UNPATCHABLE"


class ValidationIssue(StrictModel):
    level: int
    code: str
    message: str
    evidence: dict[str, Any] = Field(default_factory=dict)

class SandboxStatus(StrEnum):
    NOT_RUN = "NOT_RUN"
    CLEAN_PROGRESS = "CLEAN_PROGRESS"
    NO_PROGRESS = "NO_PROGRESS"
    OFF_TARGET_EFFECT = "OFF_TARGET_EFFECT"
    REGRESSION = "REGRESSION"
    CONNECTIVITY_FAIL = "CONNECTIVITY_FAIL"
    EXECUTION_FAIL = "EXECUTION_FAIL"
    TIMEOUT = "TIMEOUT"


class SandboxScope(StrEnum):
    NONE = "NONE"
    ISOLATED_CANDIDATE = "ISOLATED_CANDIDATE"
    JOINT_BUNDLE = "JOINT_BUNDLE"


class SandboxEvidence(StrictModel):
    status: SandboxStatus = SandboxStatus.NOT_RUN
    scope: SandboxScope = SandboxScope.NONE
    base_snapshot_id: str | None = None
    removed_original_violation_ids: list[str] = Field(default_factory=list)
    removed_original_marker_fingerprints: list[str] = Field(default_factory=list)
    target_violation_removed_ids: list[str] = Field(default_factory=list)
    new_violation_ids: list[str] = Field(default_factory=list)
    new_marker_fingerprints: list[str] = Field(default_factory=list)
    removed_original_count: int | None = Field(default=None, ge=0)
    new_violation_count: int | None = Field(default=None, ge=0)
    connectivity_preserved: bool | None = None
    verification_ref: ArtifactRef | None = None
    new_violation_records: list[dict] = Field(default_factory=list)
    attempted_script_sha256: str | None = None
    attempted_gds_sha256: str | None = None
    attempted_drc_sha256: str | None = None
    candidate_ids: list[str] = Field(default_factory=list)
    base_script_sha256: str | None = None
    physical_effect_fingerprint: str | None = None
    patch_plan_sha256: str | None = None
    rule_deck_sha256: str | None = None
    evaluator_sha256: str | None = None
    connectivity_reference_sha256: str | None = None
    failure_code: str | None = None
    failure_message: str | None = None
    execution_protocol_version: str = "legacy-v1"
    attempt_id: str | None = None
    logical_evidence_key: str | None = None
    terminal_status: AttemptTerminalStatus | None = None
    drc_validity: EvidenceValidity = EvidenceValidity.NOT_EVALUATED
    sanity_validity: EvidenceValidity = EvidenceValidity.NOT_EVALUATED
    connectivity_validity: EvidenceValidity = EvidenceValidity.NOT_EVALUATED
    fresh_evidence_valid: bool = False

    @field_validator(
        "attempted_script_sha256", "attempted_gds_sha256",
        "attempted_drc_sha256", "base_script_sha256",
        "physical_effect_fingerprint", "patch_plan_sha256",
        "rule_deck_sha256", "evaluator_sha256",
        "connectivity_reference_sha256",
    )
    @classmethod
    def validate_optional_sha256(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.lower()
        if len(normalized) != 64 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("sandbox artifact hash must be SHA-256")
        return normalized

    @model_validator(mode="before")
    @classmethod
    def preserve_legacy_count_defaults(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        raw = dict(value)
        if not str(
            raw.get("execution_protocol_version", "legacy-v1")
        ).startswith("p3.2"):
            raw.setdefault("removed_original_count", 0)
            raw.setdefault("new_violation_count", 0)
        return raw

    @model_validator(mode="after")
    def validate_tool_evidence(self) -> "SandboxEvidence":
        if self.removed_original_violation_ids:
            object.__setattr__(self, "removed_original_count", max(
                self.removed_original_count or 0,
                len(set(self.removed_original_violation_ids)),
            ))
        if self.new_violation_ids:
            object.__setattr__(self, "new_violation_count", max(
                self.new_violation_count or 0,
                len(set(self.new_violation_ids)),
            ))
        if (
            self.removed_original_count is not None
            and len(set(self.removed_original_marker_fingerprints))
            > self.removed_original_count
        ):
            raise ValueError("removed marker identity exceeds official removed count")
        if (
            self.new_violation_count is not None
            and len(set(self.new_marker_fingerprints)) > self.new_violation_count
        ):
            raise ValueError("new marker identity exceeds official new count")
        if self.status == SandboxStatus.CLEAN_PROGRESS:
            if (
                self.verification_ref is None
                or not self.base_snapshot_id
                or self.removed_original_count is None
                or self.removed_original_count <= 0
                or self.new_violation_count != 0
                or self.connectivity_preserved is not True
                or not all((
                    self.attempted_script_sha256,
                    self.attempted_gds_sha256,
                    self.attempted_drc_sha256,
                ))
            ):
                raise ValueError(
                    "clean sandbox progress requires snapshot-bound tool evidence"
                )
            if self.execution_protocol_version.startswith("p3.2") and (
                self.drc_validity != EvidenceValidity.VALID
                or self.sanity_validity != EvidenceValidity.VALID
                or self.connectivity_validity != EvidenceValidity.VALID
                or not self.fresh_evidence_valid
                or not self.attempt_id
                or not self.logical_evidence_key
                or self.terminal_status != AttemptTerminalStatus.EVALUATED
            ):
                raise ValueError(
                    "P3.2 clean progress requires complete attempt-owned evidence"
                )
        return self


class RuleBenefitProofRecord(StrictModel):
    violation_id: str
    status: Literal["PROVEN_REMOVED", "NOT_PROVEN", "CONTRADICTED"]
    proof_code: str
    input_hash: str
    proof_version: str = "rule-proof-v2"


class BenefitEvidence(StrictModel):
    proof_status: Literal["NONE", "PROVEN"] = "NONE"
    proof_codes: list[str] = Field(default_factory=list)
    rule_proofs: list[RuleBenefitProofRecord] = Field(default_factory=list)
    proven_removed_violation_ids: list[str] = Field(default_factory=list)
    proven_drc_benefit_lb: int = Field(default=0, ge=0)
    heuristic_removed_violation_ids: list[str] = Field(default_factory=list)
    heuristic_drc_benefit_est: int = Field(default=0, ge=0)
    sandbox: SandboxEvidence = Field(default_factory=SandboxEvidence)

    @model_validator(mode="after")
    def validate_proof(self) -> "BenefitEvidence":
        unique = sorted(set(self.proven_removed_violation_ids))
        if self.proven_drc_benefit_lb > len(unique):
            raise ValueError("proven benefit cannot exceed uniquely proven violations")
        if self.proven_drc_benefit_lb and self.proof_status != "PROVEN":
            raise ValueError("positive proven benefit requires proof_status=PROVEN")
        if self.proof_status == "NONE" and self.proven_removed_violation_ids:
            raise ValueError("unproven violation IDs cannot enter trusted coverage")
        return self

class RepairCandidate(StrictModel):
    schema_version: Literal["2.0"] = "2.0"
    candidate_id: str
    region_id: str
    subgraph_id: str
    rank_from_agent: int
    action_family: str
    edits: list[LayoutEdit]
    semantic_objective: str = "AUTO"
    semantic_postconditions: list[Predicate] = Field(default_factory=list)
    target_violation_ids: list[str] = Field(default_factory=list)
    target_object_ids: list[str] = Field(default_factory=list)
    affected_object_ids: list[str] = Field(default_factory=list)
    affected_net_ids: list[str] = Field(default_factory=list)
    affected_connectivity_component_ids: list[str] = Field(
        default_factory=list
    )
    affected_physical_geometry_ids: list[str] = Field(default_factory=list)
    source_targets: list[SourceTarget] = Field(default_factory=list)
    edit_footprint_dbu: Box
    required_resources: list[ResourceClaim] = Field(default_factory=list)
    released_resources: list[ResourceClaim] = Field(default_factory=list)
    benefit_evidence: BenefitEvidence = Field(default_factory=BenefitEvidence)
    predicted_introduced_rule_ids: list[str] = Field(default_factory=list)
    predicted_timing_delta: TimingDelta | None = None
    predicted_side_effects: list[SideEffect] = Field(default_factory=list)
    preconditions: list[Predicate] = Field(default_factory=list)
    postconditions: list[Predicate] = Field(default_factory=list)
    requires_candidate_ids: list[str] = Field(default_factory=list)
    excludes_candidate_ids: list[str] = Field(default_factory=list)
    cooperates_with_hints: list[str] = Field(default_factory=list)
    experience_support_ids: list[str] = Field(default_factory=list)
    agent_confidence_milli: int = 0
    estimated_disturbance: DisturbanceEstimate = Field(default_factory=DisturbanceEstimate)
    estimated_cost: CostEstimate = Field(default_factory=CostEstimate)
    validation_status: CandidateValidationStatus
    validation_issues: list[ValidationIssue] = Field(default_factory=list)
    semantic_fingerprint: str

    @model_validator(mode="before")
    @classmethod
    def migrate_v1_benefit(cls, value: Any) -> Any:
        """Read v1 artifacts without promoting heuristic lower bounds to fact."""
        if not isinstance(value, dict):
            return value
        raw = dict(value)
        old_ids = list(raw.pop("predicted_removed_violation_ids", []) or [])
        old_lb = int(raw.pop("predicted_drc_benefit_lb", 0) or 0)
        old_est = int(raw.pop("predicted_drc_benefit_est", 0) or 0)
        if "benefit_evidence" not in raw:
            raw["benefit_evidence"] = {
                "proof_status": "NONE",
                "proof_codes": ["MIGRATED_V1_HEURISTIC"] if old_ids or old_lb or old_est else [],
                "proven_removed_violation_ids": [],
                "proven_drc_benefit_lb": 0,
                "heuristic_removed_violation_ids": old_ids,
                "heuristic_drc_benefit_est": max(old_lb, old_est, len(old_ids)),
                "sandbox": {"status": "NOT_RUN", "scope": "NONE"},
            }
        raw["schema_version"] = "2.0"
        return raw

    @field_validator("agent_confidence_milli")
    @classmethod
    def confidence_range(cls, value: int) -> int:
        if not 0 <= value <= 1000:
            raise ValueError("agent_confidence_milli must be in [0, 1000]")
        return value

    @property
    def is_noop(self) -> bool:
        return self.action_family == ActionType.NO_OP and not self.edits


    @property
    def predicted_removed_violation_ids(self) -> list[str]:
        """Deprecated read-only compatibility view for pre-v2 callers."""
        return self.benefit_evidence.heuristic_removed_violation_ids

    @property
    def predicted_drc_benefit_lb(self) -> int:
        return self.benefit_evidence.proven_drc_benefit_lb

    @property
    def predicted_drc_benefit_est(self) -> int:
        return self.benefit_evidence.heuristic_drc_benefit_est


class CandidateEdgeEvidence(StrictModel):
    evidence_id: str
    kind: str
    details: dict[str, Any] = Field(default_factory=dict)


class CandidateEdge(StrictModel):
    edge_id: str
    a: str
    b: str
    relation: Literal["CONFLICTS_WITH", "SHARES_OBJECT_WITH", "COMPETES_RESOURCE_WITH",
                      "REQUIRES", "COOPERATES_WITH", "TIMING_COUPLED_WITH"]
    hard: bool
    directed: bool
    score_milli: int
    evidence: list[CandidateEdgeEvidence]
class CandidateGraphAudit(StrictModel):
    candidate_pair_count: int = 0
    relevant_pair_count: int = 0
    covered_relevant_pair_count: int = 0
    completeness_milli: int = 1000
    missing_relevant_pairs: list[tuple[str, str]] = Field(default_factory=list)
    spatially_relevant_pair_count: int = 0
    source_overlap_pair_count: int = 0
    object_overlap_pair_count: int = 0
    instance_occurrence_overlap_pair_count: int = 0
    net_overlap_pair_count: int = 0
    connectivity_component_overlap_pair_count: int = 0
    resource_overlap_pair_count: int = 0
    hard_conflict_edge_count: int = 0
    soft_relation_edge_count: int = 0
    unresolved_pair_count: int = 0
    unresolved_pairs: list[tuple[str, str]] = Field(default_factory=list)


class CandidateGraph(StrictModel):
    subgraph_id: str
    candidates: list[RepairCandidate]
    edges: list[CandidateEdge]
    audit: CandidateGraphAudit = Field(default_factory=CandidateGraphAudit)
    resource_capacities: dict[str, int] = Field(default_factory=dict)
    no_good_candidate_sets: list[list[str]] = Field(default_factory=list)
    bundle_atoms: list[list[str]] = Field(default_factory=list)


class JointRepairBundle(StrictModel):
    bundle_id: str
    subgraph_id: str
    selected_candidate_ids: list[str]
    solver_status: str
    benefit: int
    disturbance: int
    evidence_quality: int
    sandbox_status: SandboxStatus = SandboxStatus.NOT_RUN
    sandbox_verification_ref: ArtifactRef | None = None
    solver_metrics: dict[str, int | float | str] = Field(default_factory=dict)
    candidate_dispositions: list[dict[str, Any]] = Field(default_factory=list)
    all_noop_reason: str | None = None


class SourceEdit(StrictModel):
    source_anchor_id: str
    original_source_hash: str
    replacement_kind: Literal[
        "replace_expr", "replace_stmt", "replace_instance_stmt",
        "replace_blank_line", "replace_top_cell_stmt", "insert_after",
        "delete_stmt",
    ]
    generated_source: str
    contributing_candidate_ids: list[str]


class CompilerDirectMutationReceipt(StrictModel):
    """Reparsed parent-to-child identity for one direct source rewrite."""

    receipt_id: str
    old_source_object_id: str
    child_source_object_id: str
    old_source_anchor_id: str
    child_source_anchor_id: str
    parent_snapshot_id: str | None = None
    child_snapshot_id: str | None = None
    compiler_operation: str
    before_source_identity: str
    after_source_identity: str
    before_physical_geometry_identity: str
    after_physical_geometry_identity: str
    verification_status: Literal[
        "SOURCE_REPARSE_VERIFIED_PHYSICAL_EDA_PENDING", "FRESH_VERIFIED",
        "SOURCE_REPARSE_VERIFIED_PHYSICAL_EDA_INCOMPLETE",
    ]


class CompilerOccurrenceReceipt(StrictModel):
    old_instance_anchor_id: str
    source_sha256: str
    statement_ast_hash: str
    parent_cell: str
    old_source_cell: str
    new_source_cell: str
    transform: tuple[int, bool, int, int]
    replaced_layers: list[str]
    replacement_layer_hashes: dict[str, str]


class PatchPlan(StrictModel):
    base_script_sha256: str
    occurrence_receipts: list[CompilerOccurrenceReceipt] = Field(default_factory=list)
    source_edits: list[SourceEdit]
    touched_anchor_ids: list[str]
    expected_object_after_hashes: dict[str, str]


class TransactionStatus(StrEnum):
    PREPARED = "PREPARED"
    RUNNING = "RUNNING"
    VERIFIED = "VERIFIED"
    COMMITTED = "COMMITTED"
    ROLLED_BACK = "ROLLED_BACK"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class Transaction(StrictModel):
    transaction_id: str
    run_id: str
    case_id: str
    iteration: int
    base_snapshot_id: str
    bundle_ids: list[str]
    patch_plan_ref: ArtifactRef
    job_dir: str
    status: TransactionStatus
    started_at: datetime
    finished_at: datetime | None = None
    tool_results: list[Any] = Field(default_factory=list)
    verification_result_ref: ArtifactRef | None = None
    decision_reasons: list[str] = Field(default_factory=list)
    rollback_reason: str | None = None
    execution_protocol_version: str = "legacy-v1"
    attempt_id: str | None = None
    logical_evidence_key: str | None = None
    execution_epoch: str | None = None
    purpose: str | None = None
    terminal_status: AttemptTerminalStatus | None = None


class VerificationOutcome(StrEnum):
    WORKSPACE_PREPARATION_FAILURE = "WORKSPACE_PREPARATION_FAILURE"
    PATCH_APPLICATION_FAILURE = "PATCH_APPLICATION_FAILURE"
    SYNTAX_FAILURE = "SYNTAX_FAILURE"
    LAYOUT_GENERATION_FAILURE = "LAYOUT_GENERATION_FAILURE"
    DRC_EXECUTION_FAILURE = "DRC_EXECUTION_FAILURE"
    EVIDENCE_PUBLICATION_FAILURE = "EVIDENCE_PUBLICATION_FAILURE"
    CANCELLED = "CANCELLED"
    GDS_SANITY_FAILURE = "GDS_SANITY_FAILURE"
    DRC_NO_PROGRESS = "DRC_NO_PROGRESS"
    NEW_DRC_INTRODUCED = "NEW_DRC_INTRODUCED"
    CONNECTIVITY_FAILURE = "CONNECTIVITY_FAILURE"
    TIMING_FAILURE = "TIMING_FAILURE"
    SUCCESS = "SUCCESS"


class VerificationResult(StrictModel):
    verification_id: str
    transaction_id: str
    outcome: VerificationOutcome
    script_valid: bool | None
    gds_sanity_pass: bool | None
    connectivity_preserved: bool | None
    timing_gate_pass: bool | None = None
    baseline_residual_count: int
    residual_violation_count: int | None
    new_violation_count: int | None
    removed_original_count: int | None
    removed_original_violation_ids: list[str] = Field(default_factory=list)
    removed_original_marker_fingerprints: list[str] = Field(default_factory=list)
    new_violation_ids: list[str] = Field(default_factory=list)
    new_marker_fingerprints: list[str] = Field(default_factory=list)
    evidence_refs: list[ArtifactRef] = Field(default_factory=list)
    failure_codes: list[str] = Field(default_factory=list)
    execution_protocol_version: str = "legacy-v1"
    attempt_id: str | None = None
    terminal_status: AttemptTerminalStatus | None = None
    script_validity: EvidenceValidity = EvidenceValidity.NOT_EVALUATED
    layout_validity: EvidenceValidity = EvidenceValidity.NOT_EVALUATED
    drc_validity: EvidenceValidity = EvidenceValidity.NOT_EVALUATED
    sanity_validity: EvidenceValidity = EvidenceValidity.NOT_EVALUATED
    connectivity_validity: EvidenceValidity = EvidenceValidity.NOT_EVALUATED
    fresh_evidence_valid: bool = False
    failure_domain: str | None = None
    failure_stage: str | None = None

    @model_validator(mode="after")
    def validate_marker_identity_counts(self) -> "VerificationResult":
        if (
            self.removed_original_count is not None
            and len(set(self.removed_original_marker_fingerprints))
            > self.removed_original_count
        ):
            raise ValueError("removed marker identity exceeds official removed count")
        if (
            self.new_violation_count is not None
            and len(set(self.new_marker_fingerprints)) > self.new_violation_count
        ):
            raise ValueError("new marker identity exceeds official new count")
        if self.execution_protocol_version.startswith("p3.2"):
            complete = all(
                value == EvidenceValidity.VALID
                for value in (
                    self.script_validity,
                    self.layout_validity,
                    self.drc_validity,
                    self.sanity_validity,
                    self.connectivity_validity,
                )
            )
            if self.fresh_evidence_valid != complete:
                raise ValueError(
                    "fresh_evidence_valid must equal complete P3.2 verification validity"
                )
            if self.outcome == VerificationOutcome.SUCCESS and not complete:
                raise ValueError("P3.2 SUCCESS requires complete fresh evidence")
        return self


class DesignState(StrictModel):
    case_id: str
    die_boundary: Box | None = None
    legal_layers: set[str] = Field(default_factory=set)
    objects: dict[str, Any] = Field(default_factory=dict)
    violations: dict[str, Any] = Field(default_factory=dict)
    local_topology: dict[str, Any] = Field(default_factory=dict)
    physical_geometries: dict[str, Any] = Field(default_factory=dict)
    rule_predicates: dict[str, Any] = Field(default_factory=dict)
    rule_witnesses: dict[str, Any] = Field(default_factory=dict)
    rule_knowledge_packs: dict[str, Any] = Field(default_factory=dict)
    source_hashes: dict[str, str] = Field(default_factory=dict)
    resource_capacities: dict[str, int] = Field(default_factory=dict)
    manufacturing_grid_dbu: int = Field(default=1, ge=1)
    dbu_per_um: int = Field(default=4000, ge=1)

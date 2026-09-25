from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import Field

from .common import ArtifactRef, Box, Edge, EditabilityClass, Point, SourceSpan, StrictModel


class RuleFamily(StrEnum):
    SPACING = "spacing"
    WIDTH = "width"
    ENCLOSURE = "enclosure"
    AREA = "area"
    ALIGNMENT = "alignment"
    SHORT = "short"
    VIA = "via"
    UNKNOWN = "unknown"


class ViolationRecord(StrictModel):
    violation_id: str
    case_id: str
    report_iteration: int
    rule_id: str
    rule_family: RuleFamily
    description: str | None = None
    marker_type: Literal["edge_pair", "polygon", "bbox"]
    marker_bbox_dbu: Box
    marker_geometry_dbu: list[Point] | list[Edge]
    layers: tuple[str, ...] = ()
    severity: int = 1
    source_report_path: str
    source_index: int
    associated_object_ids: list[str] = Field(default_factory=list)
    association_quality: Literal["exact", "geometric", "unknown"] = "unknown"
    fingerprint: str


class LayoutObject(StrictModel):
    object_id: str
    source_anchor_id: str | None = None
    kind: Literal["polygon", "path", "via_stack", "cell_instance", "pin", "unknown"]
    layer: str
    bbox_dbu: Box
    geometry_dbu: list[Point] | None = None
    net_id: str | None = None
    net_mapping_quality: Literal[
        "exact", "connectivity_component", "unavailable"
    ] = "unavailable"
    connectivity_component_id: str | None = None
    routing_type: str | None = None
    editability: EditabilityClass
    touches_region_boundary: bool = False
    source_span: SourceSpan | None = None
    insertion_source_span: SourceSpan | None = None
    source_variable: str | None = None
    source_cell: str | None = None
    layer_number: int | None = None
    geometry_hash: str


class RuleSpec(StrictModel):
    description: str | None = None
    repair_semantics: Literal[
        "METAL_SPACING", "PARALLEL_RUN", "GRID_ALIGNMENT",
        "TRACK_ALIGNMENT", "VIA_METAL_WIDTH", "ENCLOSURE", "UNKNOWN",
    ] = "UNKNOWN"
    family: RuleFamily = RuleFamily.UNKNOWN
    primary_layers: list[str] = Field(default_factory=list)
    related_layers: list[str] = Field(default_factory=list)
    min_influence_nm: int | None = None
    halo_multiplier: float = 2.0
    allowed_action_families: list[str] = Field(default_factory=lambda: ["NO_OP", "MOVE_SHAPE"])
    compound_with: list[str] = Field(default_factory=list)
    provenance: str


class RuleCatalog(StrictModel):
    schema_version: str = "1.0"
    rules: dict[str, RuleSpec] = Field(default_factory=dict)
    fallback: RuleSpec
    content_hash: str

    def lookup(self, rule_id: str) -> tuple[RuleSpec, bool]:
        return (self.rules[rule_id], True) if rule_id in self.rules else (self.fallback, False)

    def repair_semantics_for(
        self, rule_id: str, violation_description: str | None = None,
    ) -> str:
        spec, exact = self.lookup(rule_id)
        if spec.repair_semantics != "UNKNOWN":
            return spec.repair_semantics
        if not exact:
            return "UNKNOWN"
        description = " ".join(
            part for part in [spec.description, violation_description] if part
        ).lower()
        layers = set(spec.primary_layers + spec.related_layers)
        if spec.family == RuleFamily.ALIGNMENT:
            return "TRACK_ALIGNMENT" if "track" in description else "GRID_ALIGNMENT"
        if spec.family == RuleFamily.WIDTH and any(
            layer.startswith("V") for layer in layers
        ) and any(layer.startswith("M") for layer in layers):
            return "VIA_METAL_WIDTH"
        if spec.family == RuleFamily.ENCLOSURE:
            return "ENCLOSURE"
        if rule_id == "M2.S.7":
            return "PARALLEL_RUN"
        if spec.family == RuleFamily.SPACING:
            return "METAL_SPACING"
        return "UNKNOWN"


class AffinityEvidence(StrictModel):
    halo_overlap_ratio: float = 0.0
    normalized_gap_score: float = 0.0
    same_rule: bool = False
    same_rule_family: bool = False
    layer_relation: float = 0.0
    shared_object_context: float = 0.0
    shared_net: float | None = None
    resource_overlap_proxy: float | None = None
    same_standard_cell_row: bool | None = None
    local_density_similarity: float | None = None


class ResourceContext(StrictModel):
    source: Literal["true_routing_grid", "geometry_proxy", "unavailable"] = "unavailable"
    occupied_track_bins: list[str] = Field(default_factory=list)
    free_track_bins: list[str] = Field(default_factory=list)
    via_site_bins: list[str] = Field(default_factory=list)
    local_whitespace_ratio: float | None = None
    local_density: float | None = None
    congestion_score: float | None = None
    confidence: float = 0.0


class TimingContext(StrictModel):
    source: Literal["sta", "proxy", "unavailable"]
    affected_path_ids: list[str] = Field(default_factory=list)
    setup_slack_budget_ps: int | None = None
    hold_slack_budget_ps: int | None = None
    critical_net_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class SourceSlice(StrictModel):
    source_anchor_id: str
    source_span: SourceSpan
    text: str


class CandidateAttemptSummary(StrictModel):
    candidate_id: str
    semantic_fingerprint: str
    outcome: str
    changed_precondition_ids: list[str] = Field(default_factory=list)


class RollbackSummary(StrictModel):
    transaction_id: str
    reason: str
    candidate_ids: list[str] = Field(default_factory=list)


class RegionFeatures(StrictModel):
    marker_count: int
    object_count: int
    token_estimate: int
    density_proxy: float | None = None
    geometry_signature: str


class RegionStatus(StrEnum):
    ACTIVE = "ACTIVE"
    INACTIVE_RESOLVED = "INACTIVE_RESOLVED"
    INACTIVE_MERGED = "INACTIVE_MERGED"
    INACTIVE_SPLIT = "INACTIVE_SPLIT"
    REACTIVATED = "REACTIVATED"
    OVERSIZED = "OVERSIZED"
    INVALID = "INVALID"


class RegionState(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    region_id: str
    lineage_id: str
    parent_region_ids: list[str] = Field(default_factory=list)
    status: RegionStatus = RegionStatus.ACTIVE
    iteration: int
    bbox_dbu: Box
    edit_halo_dbu: Box
    layers: list[str]
    violation_ids: list[str]
    rule_ids: list[str]
    rule_family_histogram: dict[str, int]
    object_ids: list[str]
    editable_object_ids: list[str]
    frozen_object_ids: list[str]
    segment_ids: list[str] = Field(default_factory=list)
    via_ids: list[str] = Field(default_factory=list)
    cell_instance_ids: list[str] = Field(default_factory=list)
    net_ids: list[str] = Field(default_factory=list)
    net_mapping_quality: Literal[
        "exact", "connectivity_component", "partial", "unavailable"
    ] = "unavailable"
    resource_context: ResourceContext = Field(default_factory=ResourceContext)
    timing_context: TimingContext | None = None
    source_slices: list[SourceSlice] = Field(default_factory=list)
    neighbor_edge_refs: list[str] = Field(default_factory=list)
    candidate_history: list[CandidateAttemptSummary] = Field(default_factory=list)
    rollback_history: list[RollbackSummary] = Field(default_factory=list)
    derived_features: RegionFeatures
    quality_flags: set[str] = Field(default_factory=set)
    content_hash: str


class PotentialAccessEvidence(StrictModel):
    """One pre-plan access fact with an explicit proof tier.

    EXACT_AUTHORIZED_POTENTIAL_ACCESS is source/occurrence authority, not
    proof that an edit will be selected. PROVEN_PHYSICAL_RELATION requires
    exact current polygons or a reviewed relation. Broad-phase and unknown
    facts remain useful context but can never create a hard dependency.
    """

    evidence_id: str
    category: Literal[
        "EXACT_AUTHORIZED_POTENTIAL_ACCESS",
        "PROVEN_PHYSICAL_RELATION",
        "BROAD_PHASE",
        "UNKNOWN",
    ]
    relation_kind: str
    access_mode: Literal["READ", "WRITE", "READ_WRITE", "CONTEXT"]
    identity_ids: list[str] = Field(default_factory=list)
    source: str
    details: dict[str, Any] = Field(default_factory=dict)


class PotentialPhysicalAccessSummary(StrictModel):
    """Conservative pre-plan physical read/write evidence for one Region."""

    schema_version: Literal["1.0", "2.0"] = "1.0"
    summary_id: str
    snapshot_id: str
    region_id: str
    writable_occurrence_ids: list[str] = Field(default_factory=list)
    writable_source_target_ids: list[str] = Field(default_factory=list)
    potential_write_geometry_ids: list[str] = Field(default_factory=list)
    source_contributor_ids: list[str] = Field(default_factory=list)
    source_definition_ids: list[str] = Field(default_factory=list)
    potential_write_layers: list[str] = Field(default_factory=list)
    locality_bbox: Box
    protected_relation_ids: list[str] = Field(default_factory=list)
    protected_relation_write_ids: list[str] = Field(default_factory=list)
    protected_read_geometry_ids: list[str] = Field(default_factory=list)
    merged_boundary_contributor_ids: list[str] = Field(default_factory=list)
    landing_contact_relation_ids: list[str] = Field(default_factory=list)
    connectivity_component_ids: list[str] = Field(default_factory=list)
    connectivity_evidence_quality: Literal[
        "exact", "inferred", "unavailable"
    ] = "unavailable"
    exact_authorized_access_ids: list[str] = Field(default_factory=list)
    proven_physical_relation_ids: list[str] = Field(default_factory=list)
    broad_phase_relation_ids: list[str] = Field(default_factory=list)
    unknown_access_ids: list[str] = Field(default_factory=list)
    potential_dof_capability_ids: list[str] = Field(default_factory=list)
    potential_carrier_ids: list[str] = Field(default_factory=list)
    access_evidence: list[PotentialAccessEvidence] = Field(default_factory=list)
    unknown_reasons: list[str] = Field(default_factory=list)
    content_hash: str


class EdgeEvidence(StrictModel):
    evidence_id: str
    kind: str
    details: dict[str, Any] = Field(default_factory=dict)


class AgentEdge(StrictModel):
    edge_id: str
    u: str
    v: str
    relation: Literal["geometry", "shared_net", "resource", "timing"]
    score: float
    hard: bool
    directed: bool = False
    evidence: list[EdgeEvidence]
    confidence: float
    created_iteration: int
    last_validated_iteration: int


class StructuredConstraint(StrictModel):
    kind: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class IntentSummary(StrictModel):
    action_family: str
    physical_intent: dict = Field(default_factory=dict)
    plan_id: str | None = None
    target_violation_ids: list[str] = Field(default_factory=list)
    target_relation: str = "AUTO"
    preferred_participant_ids: list[str] = Field(default_factory=list)
    preferred_dof_ids: list[str] = Field(default_factory=list)
    coordination_request: str | None = None
    target_object_ids: list[str] = Field(default_factory=list)
    footprint_dbu: Box | None = None
    fingerprint: str


class ResourceClaim(StrictModel):
    resource_id: str
    amount: int = 1
    capacity: int | None = None


class TimingClaim(StrictModel):
    path_id: str
    setup_delta_ps: int | None = None
    hold_delta_ps: int | None = None


class BoundaryDependency(StrictModel):
    dependency_id: str
    u_scope_id: str
    v_scope_id: str
    relation_types: list[str]
    evidence_ids: list[str]
    hard: bool
    requires_transaction_recheck: bool = True


class AgentPlanningView(StrictModel):
    view_id: str
    scope_id: str
    region_ids: list[str]
    internal_edge_ids: list[str]
    boundary_dependency_ids: list[str]
    factor_summary: dict[str, Any]
    factor_summary_ref: ArtifactRef | None = None
    context_token_estimate: int


class HierarchicalPlanningScope(StrictModel):
    scope_id: str
    hard_component_region_ids: list[str]
    factor_node_ids: list[str]
    views: list[AgentPlanningView]
    view_refs: list[ArtifactRef] = Field(default_factory=list)
    cross_view_constraints: list[dict[str, Any]] = Field(default_factory=list)
    cross_view_constraints_ref: ArtifactRef | None = None
    solver_scope: Literal["GLOBAL_HARD_COMPONENT", "FACTORIZED"] = "GLOBAL_HARD_COMPONENT"


class NeighborMessage(StrictModel):
    message_id: str
    round: int
    sender_region_id: str
    receiver_region_id: str
    relation: str
    evidence_ids: list[str] = Field(default_factory=list)
    constraints: list[StructuredConstraint] = Field(default_factory=list)
    proposed_intents: list[IntentSummary] = Field(default_factory=list)
    resource_claims: list[ResourceClaim] = Field(default_factory=list)
    timing_claims: list[TimingClaim] = Field(default_factory=list)
    requested_response: list[str] = Field(default_factory=list)
    content_hash: str


class AgentSubgraph(StrictModel):
    subgraph_id: str
    region_ids: list[str]
    edge_ids: list[str]
    hierarchical: bool = False
    scope_kind: Literal["FLAT", "PLANNING_VIEW"] = "FLAT"
    physical_scope_id: str | None = None
    solver_scope_id: str | None = None
    planning_view_id: str | None = None
    boundary_dependency_ids: list[str] = Field(default_factory=list)
    priority: int = 0


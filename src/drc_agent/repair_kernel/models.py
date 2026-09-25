from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, model_validator

from drc_agent.schemas.common import Box, Edge, StrictModel
from drc_agent.schemas.repair_program import RepairOperation


class RepairFamily(StrEnum):
    SPACING = "SPACING"
    ENCLOSURE = "ENCLOSURE"
    VIA_METAL_CONTEXTUAL = "VIA_METAL_CONTEXTUAL"
    ALIGNMENT = "ALIGNMENT"
    UNSUPPORTED = "UNSUPPORTED_REPAIR_FAMILY"


class Observability(StrEnum):
    EXACT = "EXACT"
    APPROXIMATE = "APPROXIMATE"
    PROXY = "PROXY"
    UNAVAILABLE = "UNAVAILABLE"


class ParticipantRole(StrEnum):
    PRIMARY_OFFENDER = "PRIMARY_OFFENDER"
    SECONDARY_OFFENDER = "SECONDARY_OFFENDER"
    ENCLOSER = "ENCLOSER"
    ENCLOSED_VIA = "ENCLOSED_VIA"
    VIA = "VIA"
    LANDING_METAL = "LANDING_METAL"
    MERGED_METAL_COMPONENT = "MERGED_METAL_COMPONENT"
    OBSTACLE = "OBSTACLE"
    CONTEXT = "CONTEXT"


class RepairFocus(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    focus_id: str
    primary_violation_id: str
    primary_rule_id: str
    repair_family: RepairFamily
    coupled_violation_ids: list[str] = Field(default_factory=list)
    region_id: str
    local_bbox: Box
    target_predicate_id: str
    target_witness_id: str
    focus_reason: str


class BoundaryContributor(StrictModel):
    edge: Literal["LEFT", "RIGHT", "BOTTOM", "TOP"]
    physical_geometry_ids: list[str]
    source_object_ids: list[str]
    source_anchor_ids: list[str]
    edit_authority: bool


class HierarchyProjection(StrictModel):
    projection_id: str
    physical_geometry_ids: list[str]
    source_object_ids: list[str]
    source_anchor_ids: list[str]
    instance_anchor_ids: list[str]
    merged_component_id: str | None = None
    boundary_contributors: list[BoundaryContributor] = Field(default_factory=list)
    quality: Observability
    unavailable_reason: str | None = None


class SceneParticipant(StrictModel):
    participant_id: str
    role: ParticipantRole
    layer: str
    geometry_kind: str
    source_object_ids: list[str] = Field(default_factory=list)
    source_anchor_ids: list[str] = Field(default_factory=list)
    instance_anchor_ids: list[str] = Field(default_factory=list)
    physical_geometry_ids: list[str] = Field(default_factory=list)
    merged_component_id: str | None = None
    bbox: Box
    boundary_edges: list[Edge] = Field(default_factory=list)
    editable: bool
    edit_authority: str
    connectivity_component_ids: list[str] = Field(default_factory=list)
    connectivity_quality: Observability
    frozen_reason: str | None = None


class SpacingRelation(StrictModel):
    relation_kind: Literal["SPACING"] = "SPACING"
    measurement_kind: Literal[
        "PROJECTED", "TIP_TO_TIP_CONTEXTUAL", "CORNER_TO_CORNER",
        "CONDITIONAL_PRL", "MINIMUM_PARALLEL_RUN",
    ] = "PROJECTED"
    participant_a_id: str
    participant_b_id: str
    orientation: Literal["HORIZONTAL_EDGES", "VERTICAL_EDGES"]
    separation_axis: Literal["X", "Y"]
    current_gap_dbu: int
    required_gap_dbu: int
    deficit_dbu: int
    projection_overlap_dbu: int
    orthogonal_gap_dbu: int = 0
    distance_squared_dbu2: int = 0
    constraint_kind: Literal[
        "MINIMUM_GAP", "MAXIMUM_PARALLEL_RUN", "MINIMUM_PARALLEL_RUN",
    ] = "MINIMUM_GAP"
    participant_a_marker_edge: Edge | None = None
    participant_b_marker_edge: Edge | None = None
    fidelity: Observability


class EnclosureRelation(StrictModel):
    relation_kind: Literal["ENCLOSURE"] = "ENCLOSURE"
    inner_participant_id: str
    outer_participant_id: str
    left_enclosure_dbu: int
    right_enclosure_dbu: int
    bottom_enclosure_dbu: int
    top_enclosure_dbu: int
    minimum_side_enclosure_dbu: int
    strengthened_side_enclosure_dbu: int
    left_deficit_dbu: int
    right_deficit_dbu: int
    bottom_deficit_dbu: int
    top_deficit_dbu: int
    fidelity: Observability


class CornerRepairOption(StrictModel):
    corner: Literal["BOTTOM_LEFT", "BOTTOM_RIGHT", "TOP_LEFT", "TOP_RIGHT"]
    metal_edge: Literal["LEFT", "RIGHT", "BOTTOM", "TOP"]
    target_coordinate_dbu: int
    current_coordinate_dbu: int | None = None
    physical_geometry_ids: list[str]
    source_object_ids: list[str]
    source_anchor_ids: list[str]
    edit_authority: bool


class ViaLandingConstraint(StrictModel):
    """Exact current occurrence facts for one metal adjacent to a via cut."""

    side: Literal["LOWER", "UPPER"]
    participant_id: str
    via_layer: str
    metal_layer: str
    via_inside_landing: bool
    enclosure_dbu: dict[str, int]
    accepted_opposite_side_minima_dbu: list[tuple[int, int]] = Field(
        default_factory=list,
    )
    containment_rule_ids: list[str] = Field(default_factory=list)
    enclosure_rule_ids: list[str] = Field(default_factory=list)


class ViaMetalRelation(StrictModel):
    relation_kind: Literal["VIA_METAL_CONTEXTUAL"] = "VIA_METAL_CONTEXTUAL"
    predicate_kind: Literal[
        "INSIDE_COINCIDENT_EDGE_COUNT",
        "NO_NONCOINCIDENT_CONVEX_CORNER",
    ] = "INSIDE_COINCIDENT_EDGE_COUNT"
    via_participant_id: str
    metal_participant_id: str
    via_inside_merged_metal: bool
    coincident_edge_map: dict[str, bool]
    current_coincident_edge_count: int
    required_coincident_edge_count: int
    missing_coincident_via_edges: list[str]
    merged_boundary_contributors: list[BoundaryContributor]
    # Every metal polygon that crosses a missing via edge and would therefore
    # keep that edge hidden after a partial edit.  These contributors form the
    # complete atomic carrier for creating a new coincident union boundary.
    coincident_edge_carrier_contributors: list[BoundaryContributor] = Field(
        default_factory=list,
    )
    bad_noncoincident_convex_corners: list[str] = Field(default_factory=list)
    corner_edge_map: dict[str, list[str]] = Field(default_factory=dict)
    corner_repair_options: list[CornerRepairOption] = Field(default_factory=list)
    adjacent_landing_constraints: list[ViaLandingConstraint] = Field(
        default_factory=list,
    )
    via_shape_kind: Literal["RECTANGLE", "OTHER"] = "OTHER"
    via_shape_on_manufacturing_grid: bool = False
    fidelity: Observability


class AlignmentRelation(StrictModel):
    relation_kind: Literal["ALIGNMENT"] = "ALIGNMENT"
    alignment_kind: Literal["EDGE_GRID", "TRACK_CENTERLINE"]
    participant_id: str
    axis: Literal["X", "Y"]
    current_edge_coordinates_dbu: list[int]
    illegal_coordinates_dbu: list[int]
    grid_pitch_dbu: int = Field(gt=0)
    grid_offset_dbu: int = 0
    base_grid_dbu: int | None = None
    current_centerline_dbu: int | None = None
    legal_target_centerlines_dbu: list[int] = Field(default_factory=list)
    preserve_width: bool = True
    preserve_grid_span_residue_modulus: int = Field(default=4, ge=1, le=8)
    fidelity: Observability


class RepairScene(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    scene_id: str
    focus: RepairFocus
    repair_family: RepairFamily
    participants: list[SceneParticipant]
    hierarchy_projections: list[HierarchyProjection]
    relation: (
        SpacingRelation | EnclosureRelation | ViaMetalRelation
        | AlignmentRelation | None
    )
    local_obstacle_ids: list[str] = Field(default_factory=list)
    edit_halo: Box
    manufacturing_grid_dbu: int = Field(gt=0)
    predicate_fidelity: Observability
    witness_fidelity: Observability
    hierarchy_quality: Observability
    connectivity_quality: Observability
    build_status: str = "SCENE_BUILT"
    failure_reason: str | None = None


class DOFType(StrEnum):
    SHIFT_BOUNDARY_EDGE = "SHIFT_BOUNDARY_EDGE"
    TRANSLATE_EDITABLE_PARTICIPANT = "TRANSLATE_EDITABLE_PARTICIPANT"
    EXTEND_ENCLOSER_LEFT = "EXTEND_ENCLOSER_LEFT"
    EXTEND_ENCLOSER_RIGHT = "EXTEND_ENCLOSER_RIGHT"
    EXTEND_ENCLOSER_BOTTOM = "EXTEND_ENCLOSER_BOTTOM"
    EXTEND_ENCLOSER_TOP = "EXTEND_ENCLOSER_TOP"
    ADD_CONNECTED_ENCLOSURE_PATCH = "ADD_CONNECTED_ENCLOSURE_PATCH"
    ALIGN_METAL_EDGE_TO_VIA_EDGE = "ALIGN_METAL_EDGE_TO_VIA_EDGE"
    ALIGN_VIA_EDGE_TO_METAL_EDGE = "ALIGN_VIA_EDGE_TO_METAL_EDGE"
    EXTEND_METAL_EDGE_TO_VIA_EDGE = "EXTEND_METAL_EDGE_TO_VIA_EDGE"
    ADD_BOUNDED_CONNECTED_METAL_PATCH = "ADD_BOUNDED_CONNECTED_METAL_PATCH"


class RepairDOF(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    dof_id: str
    scene_id: str
    participant_id: str
    source_object_ids: list[str]
    dof_type: DOFType
    parameter_name: str
    axis: Literal["X", "Y"] | None = None
    edge: Literal["LEFT", "RIGHT", "BOTTOM", "TOP"] | None = None
    domain_min_dbu: int
    domain_max_dbu: int
    allowed_intervals: list[tuple[int, int]]
    manufacturing_grid_dbu: int
    predicate_effect: str
    expected_direction: str
    locality_bbox: Box
    requires_connectivity_preview: bool
    connectivity_quality: Observability
    risk_flags: list[str] = Field(default_factory=list)
    coupling_group_id: str | None = None
    conflicts_with_dof_ids: list[str] = Field(default_factory=list)
    edit_operation_family: str
    satisfies_relation_keys: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def ordered_domain(self) -> "RepairDOF":
        if self.domain_min_dbu > self.domain_max_dbu:
            raise ValueError("DOF domain is empty")
        return self


class GeometrySolveRequest(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    request_id: str
    target_relation: str
    preferred_participant_ids: list[str] = Field(default_factory=list)
    preferred_dof_ids: list[str] = Field(default_factory=list)
    allowed_dof_ids: list[str] = Field(default_factory=list)
    forbidden_dof_ids: list[str] = Field(default_factory=list)
    max_operation_count: int = Field(default=4, ge=1, le=4)
    # P5 may enumerate a larger symbolic pool before the physical-search child
    # bound is applied. This does not increase paid EDA calls by itself.
    max_solutions: int = Field(default=4, ge=1, le=16)
    strategy_tag: str = "SATISFY_PRIMARY_PREDICATE_MINIMUM_DISTURBANCE"
    disturbance_preference: str = "LEXICOGRAPHIC_MINIMUM"


class SolverStatus(StrEnum):
    GEOMETRY_FEASIBLE = "GEOMETRY_FEASIBLE"
    GEOMETRY_UNSAT = "GEOMETRY_UNSAT"
    UNSUPPORTED_REPAIR_FAMILY = "UNSUPPORTED_REPAIR_FAMILY"
    NO_EDITABLE_DOF = "NO_EDITABLE_DOF"
    INSUFFICIENT_FIDELITY = "INSUFFICIENT_FIDELITY"
    COMPILER_UNEXPRESSIBLE = "COMPILER_UNEXPRESSIBLE"


class GeometrySolution(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    solution_id: str
    scene_id: str
    repair_family: RepairFamily
    selected_dof_ids: list[str]
    solver_status: SolverStatus
    objective_tuple: tuple[int, int, int, int, int, int]
    exact_geometry_parameters: dict[str, Any]
    predicted_before_measurements: dict[str, int | bool | str]
    predicted_after_measurements: dict[str, int | bool | str]
    predicate_constraints_satisfied: bool
    operation_count: int
    operations: list[RepairOperation]
    source_object_ids: list[str]
    physical_effect_fingerprint: str
    risk_flags: list[str] = Field(default_factory=list)
    requires_connectivity_preview: bool
    solver_proof: list[str]


class Phase3SampleResult(StrictModel):
    sample_id: str
    rule_id: str
    repair_family: RepairFamily
    focus_built: bool
    scene_built: bool
    scene_hash: str | None = None
    participant_count: int = 0
    editable_participant_count: int = 0
    hierarchy_quality: Observability = Observability.UNAVAILABLE
    predicate_measurement_available: bool = False
    dof_count: int = 0
    dof_domains_nonempty: bool = False
    solver_feasible_solution_count: int = 0
    unique_physical_effect_count: int = 0
    ir_compilation_count: int = 0
    live_sandbox_attempt_count: int = 0
    target_cleared_count: int = 0
    new_drc_count: int = 0
    connectivity_pass_count: int = 0
    clean_count: int = 0
    deepest_failure_stage: str

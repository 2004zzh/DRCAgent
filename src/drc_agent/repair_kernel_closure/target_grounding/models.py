from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field

from drc_agent.repair_kernel_closure.edit_authority import EditAuthorityClass
from drc_agent.schemas.common import Box, Edge, Point, StrictModel


class EdgeClass(StrEnum):
    TIP = "TIP"
    SIDE = "SIDE"
    UNKNOWN = "UNKNOWN"


class MappingFidelity(StrEnum):
    EXACT = "EXACT"
    GRID_TOLERANT = "GRID_TOLERANT"
    APPROXIMATE = "APPROXIMATE"
    UNAVAILABLE = "UNAVAILABLE"


class TargetParticipantRole(StrEnum):
    FIRST_OFFENDER = "FIRST_OFFENDER"
    SECOND_OFFENDER = "SECOND_OFFENDER"


class GroundedRelevantEdge(StrictModel):
    edge_id: str
    physical_edge: Edge
    reported_marker_edge: Edge
    orientation: Literal["HORIZONTAL", "VERTICAL"]
    physical_edge_length_dbu: int
    reported_projection_length_dbu: int
    edge_class: EdgeClass
    line_distance_dbu: int = Field(ge=0)
    mapping_fidelity: MappingFidelity


class GroundedTargetParticipant(StrictModel):
    participant_id: str
    physical_geometry_id: str
    layer: str
    flattened_bbox: Box
    flattened_geometry: list[Point]
    source_object_id: str
    source_anchor_id: str
    instance_anchor_id: str | None = None
    source_cell_name: str | None = None
    instance_name: str | None = None
    participant_role: TargetParticipantRole
    edge_ids: list[str]
    relevant_edges: list[GroundedRelevantEdge]
    mapping_fidelity: MappingFidelity
    authority_class: EditAuthorityClass
    authority_resolution_ref: str
    is_target_witness_contributor: Literal[True] = True


class GroundedTargetRelation(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    relation_id: str
    rule_id: str
    repair_family: Literal["SPACING"] = "SPACING"
    relation_type: str
    target_violation_id: str
    witness_id: str
    participant_ids: list[str] = Field(min_length=2)
    marker_bbox: Box
    offending_edge_pair: list[GroundedRelevantEdge] = Field(default_factory=list)
    separation_axis: Literal["X", "Y", "UNKNOWN"]
    current_measurement: int
    required_measurement: int
    deficit: int
    predicate_fidelity: str
    geometry_fidelity: MappingFidelity
    source_mapping_fidelity: MappingFidelity
    hierarchy_fidelity: MappingFidelity
    witness_locked: Literal[True] = True
    grounding_reason_codes: list[str]
    rule_deck_sha256: str
    reviewed_rule_source_line: int
    tip_threshold_dbu: int | None = None
    edge_semantics_valid: bool


class TargetGroundingResult(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    status: str
    participants: list[GroundedTargetParticipant]
    relation: GroundedTargetRelation | None = None
    neighbor_context_geometry_ids: list[str] = Field(default_factory=list)
    rejected_editable_object_ids: list[str] = Field(default_factory=list)
    failure_codes: list[str] = Field(default_factory=list)


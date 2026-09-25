from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from drc_agent.schemas.common import Box, Edge, StrictModel


class M4EdgeClass(StrEnum):
    BOUNDARY = "BOUNDARY"
    TIP = "TIP"


class M4RelationFidelity(StrEnum):
    EXACT = "EXACT"
    REVIEWED = "REVIEWED"
    UNAVAILABLE = "UNAVAILABLE"


class ReviewedM4Participant(StrictModel):
    participant_id: str
    physical_geometry_id: str
    source_object_id: str | None = None
    source_anchor_id: str | None = None
    instance_anchor_id: str | None = None
    source_cell: str | None = None
    bbox_dbu: Box
    marker_edge: Edge
    physical_edge: Edge
    edge_class: M4EdgeClass
    merged_component_id: str


class ReviewedM4Relation(StrictModel):
    relation_id: str
    rule_id: str
    violation_id: str
    relation_type: str
    marker_edges: list[Edge] = Field(min_length=2, max_length=2)
    participants: list[ReviewedM4Participant] = Field(min_length=2, max_length=2)
    edge_orientation: str
    separation_axis: str
    projection_overlap_dbu: int = Field(ge=0)
    parallel_run_length_dbu: int = Field(ge=0)
    current_measurement_dbu: int = Field(ge=0)
    required_measurement_dbu: int = Field(ge=0)
    deficit_or_surplus_dbu: int
    merged_component_ids: list[str]
    deck_sha256: str
    deck_lines: list[int]
    fidelity: M4RelationFidelity
    reason_codes: list[str] = Field(default_factory=list)


class DebtCoverageRecord(StrictModel):
    violation_id: str
    rule_id: str
    status: str
    relation_id: str | None = None
    reason_codes: list[str] = Field(default_factory=list)

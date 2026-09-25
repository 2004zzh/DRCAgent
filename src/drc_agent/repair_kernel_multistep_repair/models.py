from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field

from drc_agent.schemas.common import Box, StrictModel
from drc_agent.schemas.rules import FlattenedPhysicalGeometry
from drc_agent.schemas.state import LayoutObject, RegionState, ViolationRecord


class SemanticFidelity(StrEnum):
    EXACT = "EXACT"
    REVIEWED = "REVIEWED"
    APPROXIMATE = "APPROXIMATE"
    UNAVAILABLE = "UNAVAILABLE"


class RebindingMode(StrEnum):
    EXACT_CURRENT_BINDING = "EXACT_CURRENT_BINDING"
    PHYSICAL_RELATION_REBINDING = "PHYSICAL_RELATION_REBINDING"


class CurrentViolationIdentity(StrictModel):
    identity_id: str
    rule_id: str
    normalized_marker_fingerprint: str
    canonical_marker_geometry: list[Any]
    marker_bbox: Box
    multiplicity_index: int = Field(ge=0)
    current_violation_id: str
    snapshot_id: str


class DebtMarkerGroup(StrictModel):
    group_id: str
    rule_id: str
    normalized_fingerprint: str
    member_violation_ids: list[str]
    member_count: int = Field(ge=1)
    representative_marker: Box
    marker_multiset_key: str
    identities: list[CurrentViolationIdentity]


class SourceLineageRecord(StrictModel):
    lineage_id: str
    parent_snapshot_source_object_id: str | None = None
    child_snapshot_source_object_id: str
    old_source_anchor_id: str | None = None
    new_source_anchor_id: str | None = None
    old_instance_anchor_id: str | None = None
    new_instance_anchor_id: str | None = None
    specialized_cell_origin: str | None = None
    current_specialized_cell: str | None = None
    parent_snapshot_id: str | None = None
    child_snapshot_id: str | None = None
    compiler_operation: str | None = None
    before_source_identity: str | None = None
    after_source_identity: str | None = None
    before_physical_geometry_identity: str | None = None
    after_physical_geometry_identity: str | None = None
    verification_status: str | None = None
    lineage_confidence: SemanticFidelity
    reason_codes: list[str] = Field(default_factory=list)


class CurrentRulePredicate(StrictModel):
    predicate_id: str
    rule_id: str
    relation_type: str
    involved_layers: list[str]
    required_dbu: int | None = None
    deck_sha256: str
    deck_lines: list[int]
    reviewed: bool


class CurrentRuleWitness(StrictModel):
    witness_id: str
    violation_id: str
    rule_id: str
    predicate_id: str
    relation_type: str
    participating_physical_geometry_ids: list[str]
    source_anchor_ids: list[str]
    instance_anchor_ids: list[str]
    physical_geometries: list[FlattenedPhysicalGeometry]
    current_measurements: dict[str, Any] = Field(default_factory=dict)
    relation_details: dict[str, Any] = Field(default_factory=dict)
    fidelity: SemanticFidelity
    unresolved_reasons: list[str] = Field(default_factory=list)


class CurrentSemanticContext(StrictModel):
    context_id: str
    snapshot_id: str
    case_id: str
    current_script: str
    current_drc: str
    violations: list[ViolationRecord]
    source_objects: list[LayoutObject]
    physical_geometries: list[FlattenedPhysicalGeometry]
    rule_predicates: dict[str, CurrentRulePredicate]
    rule_witnesses: dict[str, CurrentRuleWitness]
    regions: list[RegionState]
    violation_to_region: dict[str, str]
    local_topology: dict[str, Any]
    source_hashes: dict[str, str]
    manufacturing_grid_dbu: int
    source_lineage_map: dict[str, SourceLineageRecord]
    instance_lineage_map: dict[str, SourceLineageRecord]
    marker_groups: list[DebtMarkerGroup]
    context_fingerprint: str


class CurrentDebtBinding(StrictModel):
    binding_id: str
    snapshot_id: str
    group_id: str
    mode: RebindingMode
    representative_violation_id: str
    member_violation_ids: list[str]
    witness_id: str
    physical_contributor_ids: list[str]
    source_anchor_ids: list[str]
    instance_anchor_ids: list[str]
    relation_type: str
    fidelity: SemanticFidelity
    reason_codes: list[str] = Field(default_factory=list)

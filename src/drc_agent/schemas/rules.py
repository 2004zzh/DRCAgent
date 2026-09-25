from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import Field

from .common import Box, Edge, Point, StrictModel


class RuleRelationType(StrEnum):
    CONDITIONAL_PARALLEL_RUN = "CONDITIONAL_PARALLEL_RUN"
    VIA_INSIDE_METAL_WITH_COINCIDENT_EDGES = (
        "VIA_INSIDE_METAL_WITH_COINCIDENT_EDGES"
    )
    VIA_METAL_NONCOINCIDENT_CONVEX_CORNER = (
        "VIA_METAL_NONCOINCIDENT_CONVEX_CORNER"
    )
    OPPOSITE_SIDE_ENCLOSURE = "OPPOSITE_SIDE_ENCLOSURE"
    EDGE_GRID_ALIGNMENT = "EDGE_GRID_ALIGNMENT"
    TRACK_CENTERLINE_ALIGNMENT = "TRACK_CENTERLINE_ALIGNMENT"
    MINIMUM_SPACING = "MINIMUM_SPACING"
    TIP_TO_TIP_SPACING = "TIP_TO_TIP_SPACING"
    CORNER_TO_CORNER_SPACING = "CORNER_TO_CORNER_SPACING"
    MINIMUM_PARALLEL_RUN = "MINIMUM_PARALLEL_RUN"
    UNKNOWN = "UNKNOWN"


class PhysicalPrimitive(StrEnum):
    POLYGON = "POLYGON"
    EDGE = "EDGE"
    VIA_POLYGON = "VIA_POLYGON"
    METAL_POLYGON = "METAL_POLYGON"
    ROUTE_FRAGMENT = "ROUTE_FRAGMENT"
    HIERARCHICAL_INSTANCE = "HIERARCHICAL_INSTANCE"


class RuleClassification(StrEnum):
    SOUND_SIMPLE = "SOUND_SIMPLE"
    CONTEXTUAL = "CONTEXTUAL"
    UNSUPPORTED = "UNSUPPORTED"


class EvidenceQuality(StrEnum):
    """Provenance quality of the physical geometry used by a witness."""

    EXACT_FLATTENED = "EXACT_FLATTENED"
    SOURCE_EXACT = "SOURCE_EXACT"
    GEOMETRIC_INFERRED = "GEOMETRIC_INFERRED"
    MARKER_PROXY = "MARKER_PROXY"
    UNAVAILABLE = "UNAVAILABLE"


class WitnessFidelity(StrEnum):
    """Fidelity of the analyzer to the frozen signoff rule predicate.

    This is deliberately independent from :class:`EvidenceQuality`: an exact
    flattened polygon can still be evaluated by an approximate predicate.
    """

    EXACT_SIGNOFF_PREDICATE = "EXACT_SIGNOFF_PREDICATE"
    EXACT_GEOMETRY_APPROX_PREDICATE = (
        "EXACT_GEOMETRY_APPROX_PREDICATE"
    )
    SOURCE_GEOMETRY_APPROX_PREDICATE = (
        "SOURCE_GEOMETRY_APPROX_PREDICATE"
    )
    PROXY = "PROXY"
    UNAVAILABLE = "UNAVAILABLE"


class PredicateClause(StrictModel):
    clause_id: str
    expression: str
    operands: list[str]
    comparator: str | None = None
    required_value_dbu: int | None = None
    required_count: int | None = None
    inclusive: bool = True


class MeasurementSemantics(StrictModel):
    quantities: list[str]
    unit: Literal["DBU"] = "DBU"
    projection: str
    orientation: str
    merge_semantics: str
    dbu_per_um: int = 4000
    manufacturing_grid_dbu: int = 4
    alignment_axis: Literal["X", "Y"] | None = None
    grid_pitch_dbu: int | None = None
    grid_offset_dbu: int = 0
    base_grid_dbu: int | None = None


class RuleDeckProvenance(StrictModel):
    path: str
    sha256: str
    start_line: int
    end_line: int
    excerpt_sha256: str
    supporting_ranges: list[str] = Field(default_factory=list)


class RulePredicateIR(StrictModel):
    schema_version: Literal["3.0"] = "3.0"
    rule_id: str
    involved_layers: list[str]
    relation_type: RuleRelationType
    participating_primitives: list[PhysicalPrimitive]
    actual_predicate: list[PredicateClause]
    required_conditions: list[str]
    measurement_semantics: MeasurementSemantics
    provenance: RuleDeckProvenance
    classification: RuleClassification
    analyzer_id: str
    analyzer_version: str = "rule-witness-v1"


class RuleKnowledgePack(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    predicate: RulePredicateIR
    exact_deck_excerpt: str
    reviewed_explanation: str
    repair_notes: list[str] = Field(default_factory=list)
    allowed_program_operations: list[str] = Field(default_factory=list)


class OccurrenceProvenance(StrictModel):
    source_sha256: str
    statement_ast_hash: str
    parent_cell: str
    source_cell: str
    transform: tuple[int, bool, int, int]
    line: int
    column: int


class FlattenedPhysicalGeometry(StrictModel):
    geometry_id: str
    occurrence_provenance: OccurrenceProvenance | None = None
    source_object_id: str | None = None
    source_anchor_id: str | None = None
    source_cell: str | None = None
    instance_anchor_id: str | None = None
    hierarchy_path: list[str] = Field(default_factory=list)
    layer: str
    polygon_dbu: list[Point]
    bbox_dbu: Box
    physical_edges: list[Edge] = Field(default_factory=list)
    ownership_quality: Literal[
        "EXACT_TOP_LEVEL", "EXACT_SHARED_CELL", "INFERRED", "UNAVAILABLE"
    ]
    editable: bool
    source_instance_count: int = 1
    geometry_hash: str


class MeasurementObservation(StrictModel):
    name: str
    value: int | bool | str | None
    required: int | bool | str | None = None
    comparator: str
    satisfied: bool | None = None
    evidence_geometry_ids: list[str] = Field(default_factory=list)


class ConnectivityWitness(StrictModel):
    quality: Literal["exact", "inferred", "unavailable"]
    connectivity_component_ids: list[str] = Field(default_factory=list)
    named_net_ids: list[str] = Field(default_factory=list)
    preserved_by_program: bool | None = None
    evidence: str


class RuleInteraction(StrictModel):
    interaction_id: str
    relation_type: RuleRelationType
    geometry_ids: list[str]
    edge_ids: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class RuleWitness(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    witness_id: str
    violation_id: str
    rule_id: str
    predicate_id: str
    offending_relation: RuleInteraction
    participating_physical_geometry_ids: list[str]
    physical_geometries: list[FlattenedPhysicalGeometry]
    current_measurements: list[MeasurementObservation]
    required_relation: str
    source_contributor_ids: list[str]
    editable_contributor_ids: list[str]
    editable_source_object_ids: list[str] = Field(default_factory=list)
    connectivity_evidence: ConnectivityWitness
    evidence_quality: EvidenceQuality
    predicate_fidelity: WitnessFidelity = WitnessFidelity.UNAVAILABLE
    local_bbox_dbu: Box
    predicate_satisfied: bool
    unresolved_reasons: list[str] = Field(default_factory=list)
    analyzer_id: str

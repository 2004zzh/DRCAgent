from __future__ import annotations

from itertools import permutations

from drc_agent.repair_kernel_closure.target_grounding import (
    M1_S2_TIP_THRESHOLD_DBU,
)
from drc_agent.schemas.common import Edge, StrictModel, stable_hash

from .models import CurrentDebtBinding, CurrentSemanticContext


M1_S2_REQUIRED_DBU = 100


class ReviewedSpacingParticipant(StrictModel):
    physical_geometry_id: str
    source_object_id: str
    source_anchor_id: str
    source_cell: str
    instance_anchor_id: str | None = None
    edge_index: int
    edge_coordinate_dbu: int
    edge_length_dbu: int
    edge_class: str


class ReviewedSpacingRelation(StrictModel):
    relation_id: str
    rule_id: str
    participant_ids: list[str]
    participants: list[ReviewedSpacingParticipant]
    separation_axis: str
    current_measurement_dbu: int
    required_measurement_dbu: int
    deficit_dbu: int
    deck_sha256: str
    fidelity: str
    reason_codes: list[str]


def _orientation(edge: Edge) -> str | None:
    if edge.start.x == edge.end.x and edge.start.y != edge.end.y:
        return "VERTICAL"
    if edge.start.y == edge.end.y and edge.start.x != edge.end.x:
        return "HORIZONTAL"
    return None


def _length(edge: Edge) -> int:
    return abs(edge.end.x - edge.start.x) + abs(edge.end.y - edge.start.y)


def _interval(edge: Edge) -> tuple[int, int]:
    return (
        tuple(sorted((edge.start.y, edge.end.y)))
        if _orientation(edge) == "VERTICAL"
        else tuple(sorted((edge.start.x, edge.end.x)))
    )


def _interval_gap(first: Edge, second: Edge) -> int:
    a1, a2 = _interval(first)
    b1, b2 = _interval(second)
    return max(a1 - b2, b1 - a2, 0)


def _coordinate(edge: Edge) -> int:
    return edge.start.x if _orientation(edge) == "VERTICAL" else edge.start.y


def ground_current_m1_s2(
    context: CurrentSemanticContext,
    binding: CurrentDebtBinding,
) -> ReviewedSpacingRelation | None:
    violation = next(
        (
            item for item in context.violations
            if item.violation_id == binding.representative_violation_id
        ),
        None,
    )
    if violation is None or violation.rule_id != "M1.S.2":
        return None
    marker_edges = [
        item for item in violation.marker_geometry_dbu
        if isinstance(item, Edge)
    ]
    if len(marker_edges) != 2:
        return None
    candidates = [
        item for item in context.physical_geometries
        if item.layer == "M1"
        and item.source_object_id and item.source_anchor_id and item.source_cell
        and item.bbox_dbu.intersects(violation.marker_bbox_dbu.expand(400))
    ]
    assignments = []
    for first, second in permutations(candidates, 2):
        if first.geometry_id == second.geometry_id:
            continue
        matched = []
        score = []
        valid = True
        for marker, geometry in zip(marker_edges, (first, second)):
            edge_values = []
            for index, edge in enumerate(geometry.physical_edges):
                if _orientation(edge) != _orientation(marker):
                    continue
                line_distance = abs(_coordinate(edge) - _coordinate(marker))
                interval_gap = _interval_gap(edge, marker)
                if line_distance > context.manufacturing_grid_dbu:
                    continue
                if interval_gap > context.manufacturing_grid_dbu:
                    continue
                edge_values.append((
                    (
                        line_distance,
                        interval_gap,
                        abs(_length(edge) - _length(marker)),
                        geometry.geometry_id,
                        index,
                    ),
                    index,
                    edge,
                ))
            if not edge_values:
                valid = False
                break
            selected = min(edge_values, key=lambda item: item[0])
            score.extend(selected[0])
            matched.append((geometry, selected[1], selected[2]))
        if valid:
            assignments.append((tuple(score), matched))
    if not assignments:
        return None
    _, matched = min(assignments, key=lambda item: item[0])
    classes = [
        "TIP" if _length(edge) <= M1_S2_TIP_THRESHOLD_DBU else "SIDE"
        for _, _, edge in matched
    ]
    if set(classes) != {"TIP", "SIDE"}:
        return None
    orientation = _orientation(marker_edges[0])
    if orientation != _orientation(marker_edges[1]):
        return None
    current = abs(_coordinate(marker_edges[0]) - _coordinate(marker_edges[1]))
    deficit = max(0, M1_S2_REQUIRED_DBU - current)
    participants = [
        ReviewedSpacingParticipant(
            physical_geometry_id=geometry.geometry_id,
            source_object_id=str(geometry.source_object_id),
            source_anchor_id=str(geometry.source_anchor_id),
            source_cell=str(geometry.source_cell),
            instance_anchor_id=geometry.instance_anchor_id,
            edge_index=index,
            edge_coordinate_dbu=_coordinate(edge),
            edge_length_dbu=_length(edge),
            edge_class=edge_class,
        )
        for (geometry, index, edge), edge_class in zip(matched, classes)
    ]
    predicate = context.rule_predicates.get("M1.S.2")
    if predicate is None:
        return None
    payload = [
        context.snapshot_id,
        binding.binding_id,
        [item.model_dump(mode="json") for item in participants],
        current,
        M1_S2_REQUIRED_DBU,
        predicate.deck_sha256,
    ]
    return ReviewedSpacingRelation(
        relation_id="reviewed_m1_relation_" + stable_hash(payload)[:20],
        rule_id="M1.S.2",
        participant_ids=[
            item.physical_geometry_id for item in participants
        ],
        participants=participants,
        separation_axis="X" if orientation == "VERTICAL" else "Y",
        current_measurement_dbu=current,
        required_measurement_dbu=M1_S2_REQUIRED_DBU,
        deficit_dbu=deficit,
        deck_sha256=predicate.deck_sha256,
        fidelity="REVIEWED",
        reason_codes=[
            "CURRENT_MARKER_EDGE_LOCKED",
            "CURRENT_PHYSICAL_EDGE_REBOUND",
            "REVIEWED_M1_S2_TIP_TO_SIDE",
            "SOURCE_INSTANCE_IDENTITY_PRESERVED",
        ],
    )

ground_current_projected_tip_side_spacing = ground_current_m1_s2

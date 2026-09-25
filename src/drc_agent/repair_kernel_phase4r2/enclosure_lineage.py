from __future__ import annotations

from pydantic import Field

from drc_agent.repair_kernel_multistep_repair.models import CurrentSemanticContext
from drc_agent.repair_kernel_multistep_repair.rebinding import CurrentDebtRebinder
from drc_agent.repair_kernel_multistep_repair.reviewed_spacing import (
    ReviewedSpacingRelation,
    ground_current_m1_s2,
)
from drc_agent.schemas.common import Box, StrictModel, stable_hash


class BaselineRelationLineage(StrictModel):
    lineage_id: str
    rule_id: str
    root_relation_id: str
    current_relation_id: str
    root_measurement_dbu: int
    current_measurement_dbu: int
    participant_origin_cells: list[str]
    edge_classes: list[str]
    classification: str
    reason_codes: list[str] = Field(default_factory=list)


class LocalRelationLineageReport(StrictModel):
    report_id: str
    root_relation_count: int
    current_relation_count: int
    persistent_variants: list[BaselineRelationLineage]
    unmatched_current_relation_ids: list[str]
    semantically_new_count: int


def _origin_cell(name: str | None) -> str:
    if not name:
        return "UNAVAILABLE"
    value = name.removeprefix("cell_")
    if "_drc_agent_spec_" in value:
        value = value.split("_drc_agent_spec_", 1)[0]
    return value


def _relations(
    context: CurrentSemanticContext,
    local_bbox: Box,
) -> list[ReviewedSpacingRelation]:
    rebinder = CurrentDebtRebinder()
    values = []
    for group in context.marker_groups:
        if group.rule_id != "M1.S.2" or not group.representative_marker.intersects(local_bbox):
            continue
        binding = rebinder.bind(
            context, rule_id=group.rule_id,
            violation_fingerprint=group.normalized_fingerprint,
            marker_bbox=group.representative_marker,
        )
        relation = ground_current_m1_s2(context, binding)
        if relation is not None:
            values.append(relation)
    return sorted(values, key=lambda item: item.relation_id)


def _signature(relation: ReviewedSpacingRelation) -> tuple:
    return tuple(sorted(
        (_origin_cell(item.source_cell), item.edge_class)
        for item in relation.participants
    ))


def match_reviewed_m1_relation_lineage(
    roots: list[ReviewedSpacingRelation],
    currents: list[ReviewedSpacingRelation],
) -> LocalRelationLineageReport:
    available = list(currents)
    persistent = []
    for root in roots:
        candidates = [item for item in available if _signature(item) == _signature(root)]
        if not candidates:
            continue
        current = min(candidates, key=lambda item: (
            abs(item.current_measurement_dbu - root.current_measurement_dbu),
            item.relation_id,
        ))
        available.remove(current)
        origins = sorted(value[0] for value in _signature(root))
        edge_classes = sorted(value[1] for value in _signature(root))
        persistent.append(BaselineRelationLineage(
            lineage_id="baseline_relation_lineage_" + stable_hash([
                root.relation_id, current.relation_id, origins, edge_classes,
            ])[:20],
            rule_id="M1.S.2", root_relation_id=root.relation_id,
            current_relation_id=current.relation_id,
            root_measurement_dbu=root.current_measurement_dbu,
            current_measurement_dbu=current.current_measurement_dbu,
            participant_origin_cells=origins, edge_classes=edge_classes,
            classification="BASELINE_RELATION_PERSISTENT_VARIANT",
            reason_codes=[
                "SAME_REVIEWED_RELATION_TYPE",
                "SAME_PARTICIPANT_CELL_LINEAGE",
                "SAME_TIP_SIDE_ROLES",
                "LOCAL_MARKER_VARIANT_NOT_TEMPORARY_DEBT",
            ],
        ))
    return LocalRelationLineageReport(
        report_id="local_relation_lineage_" + stable_hash([
            [item.relation_id for item in roots],
            [item.relation_id for item in currents],
            [item.lineage_id for item in persistent],
        ])[:20],
        root_relation_count=len(roots), current_relation_count=len(currents),
        persistent_variants=persistent,
        unmatched_current_relation_ids=[item.relation_id for item in available],
        semantically_new_count=len(available),
    )


def bind_local_baseline_relation_lineage(
    root_context: CurrentSemanticContext,
    current_context: CurrentSemanticContext,
    *,
    local_bbox: Box,
) -> LocalRelationLineageReport:
    return match_reviewed_m1_relation_lineage(
        _relations(root_context, local_bbox),
        _relations(current_context, local_bbox),
    )

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, Field

from drc_agent.actions.proof import RuleBenefitProofEngine
from drc_agent.schemas.action import (
    ActionType, BenefitEvidence, CandidateValidationStatus, DesignState, DisturbanceEstimate,
    EditProvenance, RepairCandidate, ValidationIssue,
)
from drc_agent.schemas.common import Box, stable_hash
from drc_agent.schemas.state import (
    LayoutObject, RegionState, RuleCatalog, RuleFamily, ViolationRecord,
)


class ValidationReport(BaseModel):
    candidate_id: str
    status: CandidateValidationStatus
    issues: list[ValidationIssue] = Field(default_factory=list)
    levels_passed: list[int] = Field(default_factory=list)


def make_noop_candidate(region: RegionState, subgraph_id: str) -> RepairCandidate:
    return RepairCandidate(
        candidate_id=f"{region.region_id}:noop", region_id=region.region_id,
        subgraph_id=subgraph_id, rank_from_agent=0, action_family=ActionType.NO_OP,
        edits=[], edit_footprint_dbu=region.bbox_dbu,
        benefit_evidence=BenefitEvidence(proof_status="PROVEN", proof_codes=["NO_OP_ZERO_EFFECT"]),
        estimated_disturbance=DisturbanceEstimate(), validation_status=CandidateValidationStatus.VALID,
        semantic_fingerprint=stable_hash([region.region_id, "NO_OP"]),
    )


def _polygon_area2(points) -> int:
    return abs(sum(points[i].x * points[(i + 1) % len(points)].y -
                   points[(i + 1) % len(points)].x * points[i].y for i in range(len(points))))


def _orientation(a, b, c) -> int:
    value = (b.y - a.y) * (c.x - b.x) - (b.x - a.x) * (c.y - b.y)
    return 0 if value == 0 else (1 if value > 0 else -1)


def _on_segment(a, b, c) -> bool:
    return (
        min(a.x, c.x) <= b.x <= max(a.x, c.x)
        and min(a.y, c.y) <= b.y <= max(a.y, c.y)
    )


def _segments_intersect(a, b, c, d) -> bool:
    o1, o2 = _orientation(a, b, c), _orientation(a, b, d)
    o3, o4 = _orientation(c, d, a), _orientation(c, d, b)
    if o1 != o2 and o3 != o4:
        return True
    return (
        (o1 == 0 and _on_segment(a, c, b))
        or (o2 == 0 and _on_segment(a, d, b))
        or (o3 == 0 and _on_segment(c, a, d))
        or (o4 == 0 and _on_segment(c, b, d))
    )


def _polygon_is_simple(points) -> bool:
    if len({(point.x, point.y) for point in points}) != len(points):
        return False
    count = len(points)
    for left in range(count):
        a, b = points[left], points[(left + 1) % count]
        for right in range(left + 1, count):
            if right in {left, (left + 1) % count}:
                continue
            if left == 0 and right == count - 1:
                continue
            c, d = points[right], points[(right + 1) % count]
            if _segments_intersect(a, b, c, d):
                return False
    return True


def _geometry_box(points) -> Box:
    return Box(
        x1=min(point.x for point in points),
        y1=min(point.y for point in points),
        x2=max(point.x for point in points),
        y2=max(point.y for point in points),
    )


def _after_box(candidate: RepairCandidate, object_id: str) -> Box | None:
    points = [
        point
        for edit in candidate.edits
        if object_id in edit.target_object_ids
        for polygon in (edit.geometry_after_dbu or [])
        for point in polygon
    ]
    return _geometry_box(points) if points else None


def _centroid_distance(left: Box, right: Box) -> float:
    lx, ly = left.centroid
    rx, ry = right.centroid
    return ((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5


def _heuristic_effect(
    candidate: RepairCandidate, violation: ViolationRecord,
    objects: list[LayoutObject],
) -> bool:
    marker = violation.marker_bbox_dbu
    for obj in objects:
        after = _after_box(candidate, obj.object_id)
        if after is None:
            continue
        before = obj.bbox_dbu
        if violation.rule_family in {RuleFamily.SPACING, RuleFamily.SHORT}:
            if after.gap(marker) > before.gap(marker):
                return True
        elif violation.rule_family == RuleFamily.WIDTH:
            if (
                candidate.action_family in {
                    ActionType.RESIZE_SHAPE, ActionType.ADJUST_ENDPOINT,
                }
                and (after.width > before.width or after.height > before.height)
            ):
                return True
        elif violation.rule_family in {
            RuleFamily.AREA, RuleFamily.ENCLOSURE,
        }:
            if (
                candidate.action_family in {
                    ActionType.RESIZE_SHAPE, ActionType.ADJUST_ENDPOINT,
                }
                and after.area > before.area
            ):
                return True
        elif violation.rule_family == RuleFamily.ALIGNMENT:
            if _centroid_distance(after, marker) < _centroid_distance(
                before, marker
            ):
                return True
    return False


def _proven_local_decomposition(
    candidate: RepairCandidate, design: DesignState,
) -> bool:
    predicates = [
        item for item in candidate.semantic_postconditions
        if item.kind == "local_fragment_decomposition"
    ]
    if len(predicates) != 1:
        return False
    parameters = predicates[0].parameters
    fragment_id = parameters.get("fragment_id")
    fragment = (
        (design.local_topology.get("fragments") or {}).get(fragment_id)
        if design.local_topology else None
    )
    if not fragment:
        return False
    parent_id = parameters.get("parent_object_id")
    parent_raw = design.objects.get(parent_id)
    if parent_raw is None:
        return False
    parent = LayoutObject.model_validate(parent_raw)
    anchor = parameters.get("parent_source_anchor_id")
    if (
        fragment.get("parent_object_id") != parent_id
        or fragment.get("parent_source_anchor_id") != anchor
        or parent.source_anchor_id != anchor
        or not parent.source_span
        or not parent.insertion_source_span
    ):
        return False
    if Box.model_validate(
        parameters.get("changed_footprint_dbu")
    ) != candidate.edit_footprint_dbu:
        return False
    fragment_box = Box.model_validate(fragment["bbox_dbu"])
    if Box.model_validate(
        parameters.get("fragment_bbox_dbu")
    ) != fragment_box:
        return False
    deletes = [
        edit for edit in candidate.edits
        if edit.op == ActionType.DELETE_POLYGON
        and edit.target_object_ids == [parent_id]
    ]
    replacements = [
        edit for edit in candidate.edits
        if edit.op == ActionType.ADD_POLYGON
        and edit.target_object_ids == [parent_id]
        and edit.geometry_after_dbu
    ]
    sides = [
        edit for edit in candidate.edits
        if edit.op == ActionType.ADD_POLYGON
        and not edit.target_object_ids and edit.geometry_after_dbu
    ]
    if (
        len(deletes) != 1 or len(replacements) != 1
        or len(sides) != int(parameters.get("side_piece_count", -1))
        or any(edit.source_anchor_ids != [anchor] for edit in candidate.edits)
    ):
        return False
    groups = {edit.replacement_group_id for edit in candidate.edits}
    if len(groups) != 1 or None in groups:
        return False
    replacement_box = _geometry_box(
        replacements[0].geometry_after_dbu[0]
    )
    if not replacement_box.intersects(fragment_box):
        return False
    side_boxes = [
        _geometry_box(edit.geometry_after_dbu[0]) for edit in sides
    ]
    if side_boxes:
        combined = replacement_box
        for box in side_boxes:
            if not box.intersects(fragment_box):
                return False
            combined = combined.union(box)
        parent_box = parent.bbox_dbu
        if parent_box.width >= parent_box.height:
            if combined.x1 != parent_box.x1 or combined.x2 != parent_box.x2:
                return False
        elif combined.y1 != parent_box.y1 or combined.y2 != parent_box.y2:
            return False
    return True


def _local_decomposition_targets_violation(
    candidate: RepairCandidate, design: DesignState, violation_id: str,
) -> bool:
    """Accept only a source-exact fragment explicitly bound to this violation."""
    if not _proven_local_decomposition(candidate, design):
        return False
    predicate = next(
        item for item in candidate.semantic_postconditions
        if item.kind == "local_fragment_decomposition"
    )
    fragment = (
        (design.local_topology.get("fragments") or {}).get(
            predicate.parameters.get("fragment_id")
        )
        if design.local_topology else None
    )
    return bool(
        fragment
        and violation_id in set(fragment.get("violation_ids") or [])
        and fragment.get("parent_object_id")
        == predicate.parameters.get("parent_object_id")
    )


def _semantic_consistency_issue(
    candidate: RepairCandidate, design: DesignState,
) -> tuple[str, str] | None:
    for predicate in candidate.semantic_postconditions:
        parameters = predicate.parameters
        if predicate.kind == "local_fragment_decomposition":
            if not _proven_local_decomposition(candidate, design):
                return (
                    "LOCAL_PATCH_UNREPRESENTABLE",
                    "local route decomposition lacks exact parent/source ownership proof",
                )
        elif predicate.kind == "match_reference_width":
            axis = parameters.get("axis")
            expected = int(parameters.get("reference_width_dbu", -1))
            boxes = [
                _geometry_box(polygon)
                for edit in candidate.edits
                for polygon in (edit.geometry_after_dbu or [])
                if edit.target_object_ids
            ]
            if len(boxes) != 1:
                return (
                    "SEMANTIC_WIDTH_UNPROVEN",
                    "MATCH_REFERENCE_WIDTH requires one authoritative after geometry",
                )
            actual = boxes[0].height if axis == "Y" else boxes[0].width
            if actual != expected:
                return (
                    "SEMANTIC_WIDTH_MISMATCH",
                    f"after perpendicular width {actual} != reference {expected}",
                )
        elif predicate.kind == "ensure_enclosure":
            marker = Box.model_validate(parameters["marker_bbox_dbu"])
            required = int(parameters["minimum_enclosure_dbu"] )
            parent_id = str(parameters["parent_object_id"])
            parent_raw = design.objects.get(parent_id)
            if parent_raw is None:
                return (
                    "SEMANTIC_ENCLOSURE_PARENT_MISSING",
                    "enclosure parent object is unavailable",
                )
            parent = LayoutObject.model_validate(parent_raw)
            patches = [
                _geometry_box(polygon)
                for edit in candidate.edits
                if edit.op == ActionType.ADD_POLYGON
                for polygon in (edit.geometry_after_dbu or [])
            ]
            if not patches:
                return (
                    "SEMANTIC_ENCLOSURE_UNPROVEN",
                    "ENSURE_ENCLOSURE has no bounded metal patch",
                )
            required_box = marker.expand(required)
            if not any(
                patch.x1 <= required_box.x1
                and patch.y1 <= required_box.y1
                and patch.x2 >= required_box.x2
                and patch.y2 >= required_box.y2
                and patch.intersection_area(parent.bbox_dbu) > 0
                for patch in patches
            ):
                return (
                    "SEMANTIC_ENCLOSURE_MISMATCH",
                    "local patch does not both enclose the marker and overlap its parent route",
                )
        elif predicate.kind == "snap_edge_to_legal_alignment":
            axis = str(parameters["axis"])
            period = int(parameters["period_dbu"])
            origin = int(parameters["origin_dbu"])
            target = int(parameters["target_coordinate_dbu"])
            boxes = [
                _geometry_box(polygon)
                for edit in candidate.edits
                for polygon in (edit.geometry_after_dbu or [])
                if edit.target_object_ids
            ]
            if len(boxes) != 1:
                return (
                    "SEMANTIC_ALIGNMENT_UNPROVEN",
                    "alignment requires one authoritative after geometry",
                )
            coordinate = boxes[0].y1 if axis == "Y" else boxes[0].x1
            if coordinate != target or (coordinate - origin) % period:
                return (
                    "SEMANTIC_ALIGNMENT_MISMATCH",
                    "after edge is not at the deterministic legal coordinate",
                )
    return None



def _specialization_targets_marker(
    candidate: RepairCandidate, marker: Box, influence: int,
) -> bool:
    expanded = marker.expand(influence)
    return any(
        edit.instance_specializations
        and any(
            _geometry_box(polygon).intersects(expanded)
            for polygon in (edit.geometry_before_dbu or [])
            + (edit.geometry_after_dbu or [])
            if polygon
        )
        for edit in candidate.edits
    )

def _locality_issue(
    candidate: RepairCandidate, violations: dict[str, ViolationRecord],
    design: DesignState, region: RegionState,
) -> tuple[str, str] | None:
    local_objectives = {
        "ENSURE_ENCLOSURE", "SNAP_EDGE_TO_LEGAL_ALIGNMENT",
        "INCREASE_SPACING", "LOCAL_DETOUR_AROUND_OBSTACLE",
    }
    if candidate.semantic_objective not in local_objectives:
        return None
    markers = [
        violations[identifier].marker_bbox_dbu
        for identifier in candidate.target_violation_ids
        if identifier in violations
    ]
    if not markers:
        return ("LOCALITY_EVIDENCE_MISSING", "local repair has no target marker")
    influence = markers[0]
    for marker in markers[1:]:
        influence = influence.union(marker)
    margin = max(design.manufacturing_grid_dbu * 4, 1)
    influence = influence.expand(margin)
    footprint = candidate.edit_footprint_dbu
    local_scope = region.bbox_dbu.expand(margin)
    if not (
        region.edit_halo_dbu.x1 <= footprint.x1
        and region.edit_halo_dbu.y1 <= footprint.y1
        and footprint.x2 <= region.edit_halo_dbu.x2
        and footprint.y2 <= region.edit_halo_dbu.y2
    ):
        return (
            "EXCESSIVE_COLLATERAL_FOOTPRINT",
            "changed footprint exceeds the bounded region edit halo",
        )
    if footprint.area > max(
        influence.area * 16, local_scope.area * 4, margin * margin * 16,
    ):
        return (
            "EXCESSIVE_COLLATERAL_FOOTPRINT",
            "changed footprint is disproportionate to local fragment scope",
        )
    if not footprint.intersects(influence) and not all(
        _local_decomposition_targets_violation(candidate, design, identifier)
        for identifier in candidate.target_violation_ids
    ):
        return (
            "OFF_TARGET_FOOTPRINT",
            "changed footprint does not intersect target violation influence",
        )
    return None


class CandidateChecker:
    MUTATION_OPS = set(ActionType) - {ActionType.NO_OP}

    def validate(
        self, candidate: RepairCandidate, region: RegionState,
        design: DesignState, allowed_action_families: set[str],
        violations: list[ViolationRecord] | dict[str, ViolationRecord],
        rule_catalog: RuleCatalog | None = None,
    ) -> ValidationReport:
        issues: list[ValidationIssue] = []
        levels = []

        def fail(level: int, code: str, message: str, status: CandidateValidationStatus) -> ValidationReport:
            issue = ValidationIssue(level=level, code=code, message=message)
            issues.append(issue)
            candidate.validation_status = status
            candidate.validation_issues = issues
            return ValidationReport(candidate_id=candidate.candidate_id, status=status,
                                    issues=issues, levels_passed=levels)

        if candidate.region_id != region.region_id or not candidate.candidate_id:
            return fail(0, "IDENTITY_MISMATCH", "candidate identity does not match region",
                        CandidateValidationStatus.INVALID_SCHEMA)
        if candidate.action_family not in allowed_action_families:
            return fail(0, "ACTION_NOT_ALLOWED", "action family is outside the rule whitelist",
                        CandidateValidationStatus.INVALID_SCHEMA)
        if candidate.is_noop:
            levels.extend(range(6))
            candidate.validation_status = CandidateValidationStatus.VALID
            return ValidationReport(candidate_id=candidate.candidate_id,
                                    status=CandidateValidationStatus.VALID, levels_passed=levels)
        if candidate.action_family in {
            ActionType.LOCAL_DETOUR, ActionType.LOCAL_REROUTE,
            ActionType.CHANGE_LAYER,
        }:
            replacement_edits = [
                edit for edit in candidate.edits
                if edit.op in {
                    ActionType.DELETE_POLYGON, ActionType.ADD_POLYGON,
                }
                and edit.replacement_group_id is not None
            ]
            groups = {edit.replacement_group_id for edit in replacement_edits}
            operations = [edit.op for edit in candidate.edits]
            if len(groups) != 1:
                return fail(
                    0, "INVALID_ROUTING_REPLACEMENT_GROUP",
                    "routing macro edits must share one replacement_group_id",
                    CandidateValidationStatus.INVALID_SCHEMA,
                )
            if operations.count(ActionType.DELETE_POLYGON) != 1:
                return fail(
                    0, "INVALID_ROUTING_DELETE_COUNT",
                    "routing macro must delete exactly one old segment",
                    CandidateValidationStatus.INVALID_SCHEMA,
                )
            if not any(
                edit.op == ActionType.ADD_POLYGON
                and edit.target_object_ids
                and edit.geometry_after_dbu
                for edit in candidate.edits
            ):
                return fail(
                    0, "INVALID_ROUTING_REPLACEMENT",
                    "routing macro must add one executable replacement polygon",
                    CandidateValidationStatus.INVALID_SCHEMA,
                )
            if (
                candidate.action_family == ActionType.CHANGE_LAYER
                and operations.count(ActionType.ADD_VIA_STACK) < 2
            ):
                return fail(
                    0, "MISSING_LAYER_CHANGE_VIAS",
                    "CHANGE_LAYER requires deterministic endpoint vias",
                    CandidateValidationStatus.INVALID_SCHEMA,
                )
        if not set(candidate.target_violation_ids).issubset(region.violation_ids):
            return fail(0, "UNKNOWN_VIOLATION", "candidate references a violation outside its region",
                        CandidateValidationStatus.INVALID_TARGET)
        if not set(candidate.target_object_ids).issubset(design.objects):
            return fail(0, "UNKNOWN_OBJECT", "candidate references an unknown object",
                        CandidateValidationStatus.INVALID_TARGET)
        violation_by_id = (
            violations if isinstance(violations, dict)
            else {item.violation_id: item for item in violations}
        )
        missing_violations = (
            set(candidate.target_violation_ids) - set(violation_by_id)
        )
        if missing_violations:
            return fail(
                0, "MISSING_VIOLATION_EVIDENCE",
                "candidate target violation evidence is unavailable",
                CandidateValidationStatus.INVALID_TARGET,
            )
        target_objects = [
            LayoutObject.model_validate(design.objects[object_id])
            for object_id in candidate.target_object_ids
        ]
        related_by_violation: dict[str, list[LayoutObject]] = {}
        max_delta = max([
            max(abs(edit.delta_dbu.dx), abs(edit.delta_dbu.dy))
            for edit in candidate.edits if edit.delta_dbu is not None
        ] or [0])
        influence = max(design.manufacturing_grid_dbu, max_delta)
        for violation_id in candidate.target_violation_ids:
            violation = violation_by_id[violation_id]
            associated = set(violation.associated_object_ids)
            related = [
                obj for obj in target_objects
                if (
                    (not associated or obj.object_id in associated)
                    and (not violation.layers or obj.layer in violation.layers)
                )
            ]
            if not related:
                return fail(
                    0, "UNRELATED_OBJECT_VIOLATION",
                    "target object is not associated with the violation layers",
                    CandidateValidationStatus.INVALID_TARGET,
                )
            if not any(
                obj.bbox_dbu.intersects(
                    violation.marker_bbox_dbu.expand(influence)
                )
                for obj in related
            ) and not _specialization_targets_marker(
                candidate, violation.marker_bbox_dbu, influence,
            ) and not (
                candidate.semantic_objective == "ENSURE_ENCLOSURE"
                and candidate.edit_footprint_dbu.intersects(
                    violation.marker_bbox_dbu
                )
            ) and not _local_decomposition_targets_violation(
                candidate, design, violation_id,
            ):
                return fail(
                    0, "OUTSIDE_VIOLATION_INFLUENCE",
                    "target geometry is outside the violation influence area",
                    CandidateValidationStatus.INVALID_TARGET,
                )
            related_by_violation[violation_id] = related
        levels.append(0)

        current_anchor_hashes = design.source_hashes
        for target in candidate.source_targets:
            if target.source_anchor_id not in current_anchor_hashes:
                return fail(1, "MISSING_SOURCE_ANCHOR", "source anchor is unavailable",
                            CandidateValidationStatus.INVALID_TARGET)
            if current_anchor_hashes[target.source_anchor_id] != target.source_hash:
                return fail(1, "STALE_TARGET", "source geometry changed after candidate generation",
                            CandidateValidationStatus.STALE_TARGET)
        edits_by_anchor = {}
        for edit in candidate.edits:
            for anchor in edit.source_anchor_ids:
                edits_by_anchor.setdefault(anchor, []).append(edit)
        for anchor, anchor_edits in edits_by_anchor.items():
            if len(anchor_edits) <= 1:
                continue
            carrier_specializations = [
                edit for edit in anchor_edits
                if (
                    edit.op == ActionType.INSTANCE_SPECIALIZATION
                    and edit.instance_specializations
                    and not edit.target_object_ids
                )
            ]
            ordinary_edits = [
                edit for edit in anchor_edits
                if edit not in carrier_specializations
            ]
            # A source statement may own both one directly rewritten polygon
            # and one occurrence-specialized child. PatchCompiler compiles
            # these in two phases, so this combination is source-exact and
            # does not represent competing rewrites of the same span.
            valid_direct_plus_specialization = (
                len(carrier_specializations) == 1
                and len(ordinary_edits) == 1
                and bool(ordinary_edits[0].target_object_ids)
                and ordinary_edits[0].op in {
                    ActionType.ADJUST_ENDPOINT,
                    ActionType.RESIZE_SHAPE,
                    ActionType.MOVE_SHAPE,
                }
            )
            replacement_groups = {
                edit.replacement_group_id for edit in anchor_edits
            }
            replacement_targets = [
                edit for edit in anchor_edits
                if edit.geometry_after_dbu and edit.target_object_ids
            ]
            delete_count = sum(
                edit.op in {ActionType.DELETE_POLYGON, ActionType.DELETE_VIA_STACK}
                for edit in anchor_edits
            )
            valid_replacement = (
                len(replacement_groups) == 1
                and None not in replacement_groups
                and delete_count == 1
                and len(replacement_targets) == 1
            )
            if not (valid_replacement or valid_direct_plus_specialization):
                return fail(
                    1, "INTERNAL_SOURCE_CONFLICT",
                    f"edits on {anchor} are not one atomic replacement group",
                    CandidateValidationStatus.INVALID_TARGET,
                )
        levels.append(1)

        for edit in candidate.edits:
            for object_id in edit.target_object_ids:
                obj = LayoutObject.model_validate(design.objects[object_id])
                if obj.editability.value == "D":
                    return fail(2, "FROZEN_OBJECT", "Class D object cannot be mutated",
                                CandidateValidationStatus.INVALID_EDITABILITY)
                if (
                    obj.editability.value == "C"
                    and edit.op != ActionType.ADJUST_ENDPOINT
                    and not _proven_local_decomposition(candidate, design)
                ):
                    return fail(
                        2, "BOUNDARY_OBJECT_OPERATION",
                        "Class C requires endpoint adjustment or an exact "
                        "source-owned local decomposition proof",
                        CandidateValidationStatus.INVALID_EDITABILITY,
                    )
                if obj.editability.value == "A" and "VIA_STACK" not in edit.op.value:
                    return fail(2, "PARTIAL_VIA_EDIT", "Class A via stack must be edited atomically",
                                CandidateValidationStatus.INVALID_EDITABILITY)
        levels.append(2)

        for edit in candidate.edits:
            grid = design.manufacturing_grid_dbu
            for polygon in edit.geometry_after_dbu or []:
                if any(point.x % grid or point.y % grid for point in polygon):
                    return fail(3, "OFF_MANUFACTURING_GRID", "after geometry is off the configured DBU grid",
                                CandidateValidationStatus.INVALID_GEOMETRY)
                if (len(polygon) < 3 or _polygon_area2(polygon) == 0
                        or not _polygon_is_simple(polygon)):
                    return fail(3, "INVALID_POLYGON", "after geometry is degenerate",
                                CandidateValidationStatus.INVALID_GEOMETRY)
                if design.die_boundary and any(not (design.die_boundary.x1 <= p.x <= design.die_boundary.x2 and
                                                     design.die_boundary.y1 <= p.y <= design.die_boundary.y2)
                                               for p in polygon):
                    return fail(3, "OUTSIDE_DIE", "after geometry is outside die boundary",
                                CandidateValidationStatus.INVALID_GEOMETRY)
            if edit.layer_after and design.legal_layers and edit.layer_after not in design.legal_layers:
                return fail(3, "ILLEGAL_LAYER", "target layer is not legal",
                            CandidateValidationStatus.INVALID_GEOMETRY)
        semantic_issue = _semantic_consistency_issue(candidate, design)
        if semantic_issue is not None:
            return fail(
                3, semantic_issue[0], semantic_issue[1],
                CandidateValidationStatus.INVALID_GEOMETRY,
            )
        locality_issue = _locality_issue(
            candidate, violation_by_id, design, region,
        )
        if locality_issue is not None:
            return fail(
                3, locality_issue[0], locality_issue[1],
                CandidateValidationStatus.INVALID_GEOMETRY,
            )
        levels.append(3)

        heuristic = [
            violation_id
            for violation_id, related in related_by_violation.items()
            if _heuristic_effect(
                candidate, violation_by_id[violation_id], related,
            )
        ]
        proof_records = RuleBenefitProofEngine().prove(
            candidate=candidate, violations=violation_by_id,
            related_objects=related_by_violation, catalog=rule_catalog,
        )
        # Marker/bbox trends remain heuristic until a proof record is
        # PROVEN_REMOVED; current DAC26 catalog is intentionally fail-closed.
        candidate.benefit_evidence = BenefitEvidence(
            proof_status="NONE",
            proof_codes=sorted({item.proof_code for item in proof_records}),
            rule_proofs=proof_records,
            proven_removed_violation_ids=[],
            proven_drc_benefit_lb=0,
            heuristic_removed_violation_ids=sorted(heuristic),
            heuristic_drc_benefit_est=len(heuristic),
        )

        resource_demand = Counter()
        for claim in candidate.required_resources:
            resource_demand[claim.resource_id] += claim.amount
            capacity = claim.capacity if claim.capacity is not None else design.resource_capacities.get(claim.resource_id)
            if capacity is not None and resource_demand[claim.resource_id] > capacity:
                return fail(4, "INTERNAL_RESOURCE_OVERFLOW", "candidate exceeds resource capacity",
                            CandidateValidationStatus.INVALID_RESOURCE)
        levels.append(4)
        if any(not edit.source_anchor_ids and edit.op in self.MUTATION_OPS for edit in candidate.edits):
            return fail(5, "NO_PATCH_ANCHOR", "mutation has no source anchor",
                        CandidateValidationStatus.UNPATCHABLE)
        levels.append(5)
        status = CandidateValidationStatus.VALID_REQUIRES_SANDBOX if (
            candidate.benefit_evidence.proven_drc_benefit_lb == 0 or any(
                effect.kind == "CONNECTIVITY_ARTICULATION_RISK"
                for effect in candidate.predicted_side_effects
            )
        ) else CandidateValidationStatus.VALID
        candidate.validation_status = status
        candidate.validation_issues = issues
        return ValidationReport(candidate_id=candidate.candidate_id, status=status,
                                issues=issues, levels_passed=levels)


def validate_candidate_set(candidates: list[RepairCandidate], region_ids: set[str]) -> None:
    grouped: dict[str, list[RepairCandidate]] = {region_id: [] for region_id in region_ids}
    for candidate in candidates:
        if candidate.region_id not in grouped:
            raise ValueError(f"candidate belongs to unknown region: {candidate.region_id}")
        grouped[candidate.region_id].append(candidate)
    for region_id, values in grouped.items():
        noops = [candidate for candidate in values if candidate.is_noop]
        if len(noops) != 1:
            raise ValueError(f"region {region_id} must have exactly one NO_OP")
        fingerprints = [candidate.semantic_fingerprint for candidate in values if not candidate.is_noop]
        if len(fingerprints) != len(set(fingerprints)):
            raise ValueError(f"region {region_id} has duplicate semantic candidates")


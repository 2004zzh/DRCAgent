from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from pydantic import Field

from drc_agent.patching.compiler import PatchCompiler
from drc_agent.patching.repair_program import build_physical_effect_fingerprint
from drc_agent.repair_kernel_closure.edit_authority.compiler_support import specialized_cell_name
from drc_agent.repair_kernel_closure.mutation_carrier.geometry import (
    infer_transform,
    local_delta_for_global,
    transform_point,
)
from drc_agent.repair_kernel_multistep.policy.models import SolverProposal, SymbolicPlan
from drc_agent.repair_kernel_multistep_repair.models import (
    CurrentDebtBinding,
    CurrentSemanticContext,
)
from drc_agent.schemas.action import (
    ActionType,
    CandidateValidationStatus,
    EditProvenance,
    InstanceGeometrySpecialization,
    JointRepairBundle,
    LayoutEdit,
    RepairCandidate,
)
from drc_agent.schemas.common import (
    Box,
    Point,
    StrictModel,
    canonical_polygon,
    stable_hash,
)
from drc_agent.schemas.state import LayoutObject

from .m4_models import ReviewedM4Participant, ReviewedM4Relation


class AtomicOccurrenceMutation(StrictModel):
    mutation_id: str
    instance_anchor_id: str
    source_cell: str
    source_object_id: str
    source_anchor_id: str
    global_dx_dbu: int
    global_dy_dbu: int
    complete_polygons_by_layer_local_dbu: dict[str, list[list[Point]]]
    specialization: InstanceGeometrySpecialization
    affected_layers: list[str]
    footprint_dbu: Box
    protected_relation_effect: str = "MAY_AFFECT"
    reason_codes: list[str] = Field(default_factory=list)


def _contains(outer: Box, inner: Box) -> bool:
    return (
        outer.x1 <= inner.x1 and outer.y1 <= inner.y1
        and inner.x2 <= outer.x2 and inner.y2 <= outer.y2
    )


def _region(context: CurrentSemanticContext, binding: CurrentDebtBinding):
    region_id = context.violation_to_region.get(binding.representative_violation_id)
    return next((item for item in context.regions if item.region_id == region_id), None)


def _compiler_owner(
    context: CurrentSemanticContext,
    binding: CurrentDebtBinding,
    footprint: Box,
) -> LayoutObject | None:
    region = _region(context, binding)
    if region is None:
        return None
    editable = set(region.editable_object_ids)
    values = [
        item for item in context.source_objects
        if item.object_id in editable and item.source_anchor_id
        and item.source_span and item.insertion_source_span
    ]
    return min(
        values,
        key=lambda item: (int(item.bbox_dbu.gap(footprint)), item.object_id),
        default=None,
    )


def build_atomic_occurrence_mutation(
    context: CurrentSemanticContext,
    participant: ReviewedM4Participant,
    *,
    global_dx_dbu: int,
    global_dy_dbu: int,
) -> AtomicOccurrenceMutation | None:
    if (
        not participant.instance_anchor_id or not participant.source_cell
        or not participant.source_object_id or not participant.source_anchor_id
    ):
        return None
    seed = next(
        (item for item in context.source_objects
         if item.object_id == participant.source_object_id),
        None,
    )
    if seed is None or not seed.geometry_dbu:
        return None
    physical = next(
        (item for item in context.physical_geometries
         if item.geometry_id == participant.physical_geometry_id),
        None,
    )
    if physical is None:
        return None
    transform = infer_transform(list(seed.geometry_dbu), list(physical.polygon_dbu))
    if transform is None:
        return None
    rotation, mirror, tx, ty = transform
    local_dx, local_dy = local_delta_for_global(
        rotation=rotation, mirror=mirror, dx=tx, dy=ty,
        global_dx=global_dx_dbu, global_dy=global_dy_dbu,
    )
    legal_layers = {item.layer for item in context.source_objects}
    by_layer: dict[str, list[list[Point]]] = defaultdict(list)
    for item in sorted(context.source_objects, key=lambda value: value.object_id):
        if (
            item.source_cell == participant.source_cell
            and item.geometry_dbu and item.layer in legal_layers
        ):
            before = canonical_polygon(list(item.geometry_dbu))
            by_layer[item.layer].append(canonical_polygon([
                Point(x=point.x + local_dx, y=point.y + local_dy)
                for point in before
            ]))
    if not by_layer or "M4" not in by_layer:
        return None
    original_m4 = sorted(
        [canonical_polygon(list(item.geometry_dbu or []))
         for item in context.source_objects
         if item.source_cell == participant.source_cell
         and item.layer == "M4" and item.geometry_dbu],
        key=stable_hash,
    )
    complete = {
        layer: sorted(polygons, key=stable_hash)
        for layer, polygons in sorted(by_layer.items())
    }
    global_points = [
        transform_point(point, rotation, mirror, tx, ty)
        for polygons in complete.values() for polygon in polygons for point in polygon
    ]
    before_points = [
        transform_point(point, rotation, mirror, tx, ty)
        for item in context.source_objects
        if item.source_cell == participant.source_cell and item.geometry_dbu
        for point in canonical_polygon(list(item.geometry_dbu or []))
    ]
    all_points = before_points + global_points
    if not all_points:
        return None
    footprint = Box(
        x1=min(point.x for point in all_points),
        y1=min(point.y for point in all_points),
        x2=max(point.x for point in all_points),
        y2=max(point.y for point in all_points),
    )
    identity = [
        participant.instance_anchor_id, participant.source_cell,
        global_dx_dbu, global_dy_dbu, complete,
    ]
    specialization = InstanceGeometrySpecialization(
        instance_anchor_id=participant.instance_anchor_id,
        parent_cell=f"cell_{context.case_id}",
        source_cell=participant.source_cell,
        specialized_cell_name=specialized_cell_name(participant.source_cell, identity),
        layer="M4", rotation=rotation, mirror=mirror,
        dx_dbu=tx, dy_dbu=ty,
        original_polygons_local_dbu=original_m4,
        replacement_polygons_local_dbu=complete["M4"],
        complete_polygons_by_layer_local_dbu=complete,
    )
    return AtomicOccurrenceMutation(
        mutation_id="atomic_occurrence_" + stable_hash(identity)[:20],
        instance_anchor_id=participant.instance_anchor_id,
        source_cell=participant.source_cell,
        source_object_id=participant.source_object_id,
        source_anchor_id=participant.source_anchor_id,
        global_dx_dbu=global_dx_dbu, global_dy_dbu=global_dy_dbu,
        complete_polygons_by_layer_local_dbu=complete,
        specialization=specialization,
        affected_layers=sorted(complete), footprint_dbu=footprint,
        reason_codes=[
            "ONE_INSTANCE_ONE_SPECIALIZATION",
            "COMPLETE_CURRENT_CELL_GEOMETRY",
            "ALL_LAYERS_MOVED_ATOMICALLY",
        ],
    )


def compile_atomic_occurrence_proposal(
    context: CurrentSemanticContext,
    binding: CurrentDebtBinding,
    relation: ReviewedM4Relation,
    participant: ReviewedM4Participant,
    plan: SymbolicPlan,
    *,
    global_dx_dbu: int,
    global_dy_dbu: int,
) -> SolverProposal | None:
    mutation = build_atomic_occurrence_mutation(
        context, participant,
        global_dx_dbu=global_dx_dbu, global_dy_dbu=global_dy_dbu,
    )
    if mutation is None:
        return None
    region = _region(context, binding)
    if region is None or not _contains(region.edit_halo_dbu, mutation.footprint_dbu):
        return None
    owner = _compiler_owner(context, binding, mutation.footprint_dbu)
    if owner is None or not owner.source_anchor_id:
        return None
    edit_identity = [mutation.mutation_id, owner.source_anchor_id]
    edit = LayoutEdit(
        edit_id="atomic_edit_" + stable_hash(edit_identity)[:20],
        op=ActionType.INSTANCE_SPECIALIZATION,
        target_object_ids=[participant.source_object_id],
        source_anchor_ids=[owner.source_anchor_id],
        editability_required={owner.editability},
        instance_specializations=[mutation.specialization],
        provenance=EditProvenance(
            generator="phase4r2_atomic_occurrence",
            evidence_ids=[relation.relation_id, binding.witness_id],
        ),
    )
    candidate = RepairCandidate(
        candidate_id="candidate_" + stable_hash(edit_identity)[:20],
        region_id=region.region_id,
        subgraph_id=f"phase4r2_{region.region_id}",
        rank_from_agent=1, action_family="INSTANCE_SPECIALIZATION",
        edits=[edit], target_violation_ids=[binding.representative_violation_id],
        target_object_ids=[participant.source_object_id],
        affected_object_ids=[participant.source_object_id],
        affected_physical_geometry_ids=[participant.physical_geometry_id],
        edit_footprint_dbu=mutation.footprint_dbu,
        validation_status=CandidateValidationStatus.VALID_REQUIRES_SANDBOX,
        semantic_fingerprint=stable_hash([relation.relation_id, mutation.mutation_id]),
    )
    bundle = JointRepairBundle(
        bundle_id="phase4r2_bundle_" + stable_hash(candidate.candidate_id)[:16],
        subgraph_id=candidate.subgraph_id,
        selected_candidate_ids=[candidate.candidate_id],
        solver_status="PHASE4R2_ATOMIC_OCCURRENCE",
        benefit=0, disturbance=0, evidence_quality=0,
    )
    try:
        patch = PatchCompiler().compile(
            bundle, [candidate], context.source_objects,
            Path(context.current_script),
        )
    except Exception:
        return None
    effect = build_physical_effect_fingerprint(candidate).sha256
    return SolverProposal(
        proposal_id="proposal_" + stable_hash([
            plan.plan_id, effect, patch.base_script_sha256,
        ])[:20],
        plan_id=plan.plan_id,
        active_obligation_id=plan.active_obligation_id,
        candidate=candidate, patch_plan=patch,
        predicted_target_effect="SATISFIED",
        physical_effect_fingerprint=effect,
        touched_instance_anchor_ids=[participant.instance_anchor_id],
    )


def displacement_options(relation: ReviewedM4Relation) -> list[tuple[int, int, int]]:
    deficit = max(0, relation.deficit_or_surplus_dbu)
    if deficit == 0:
        return []
    first_x = relation.participants[0].marker_edge.start.x
    second_x = relation.participants[1].marker_edge.start.x
    if first_x <= second_x:
        return [(0, -deficit, 0), (1, deficit, 0)]
    return [(0, deficit, 0), (1, -deficit, 0)]

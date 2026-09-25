from __future__ import annotations

from drc_agent.regions.physical import verified_occurrence_transform

from collections import defaultdict
from itertools import combinations

from drc_agent.repair_kernel.geometry import rectangle_points
from drc_agent.repair_kernel.dof import generate_repair_dofs
from drc_agent.repair_kernel.models import (
    EnclosureRelation, GeometrySolution, ParticipantRole, RepairFamily,
    RepairScene, SpacingRelation, ViaMetalRelation,
)
from drc_agent.schemas.common import (
    Box, Edge, Point, canonical_polygon, stable_hash,
)
from drc_agent.schemas.repair_program import (
    AddBoundedPolygonOp, InstanceLayerReplacement, MoveObjectOp,
    RepairProgram, ResizeEdgeOp, SpecializeInstanceBoundaryOp,
    SpecializeInstanceLayerOp,
)
from drc_agent.schemas.rules import FlattenedPhysicalGeometry
from drc_agent.schemas.state import LayoutObject

from .models import AuthorityResolution
from .enclosure import build_enclosure_specialization_programs


def _program(scene: RepairScene, operation, strategy: str) -> RepairProgram:
    return RepairProgram(
        program_id="phase3r_program_" + stable_hash([
            scene.scene_id, operation.model_dump(mode="json"),
        ])[:20],
        region_id=scene.focus.region_id,
        target_violation_ids=[scene.focus.primary_violation_id],
        witness_ids=[scene.focus.target_witness_id],
        operations=[operation], strategy_code=strategy,
    )


def _enclosure_programs(
    scene: RepairScene, authority: AuthorityResolution,
) -> list[RepairProgram]:
    relation = scene.relation
    assert isinstance(relation, EnclosureRelation)
    participants = {item.participant_id: item for item in scene.participants}
    via = participants[relation.inner_participant_id]
    landing = participants[relation.outer_participant_id]
    owner = authority.insertion_owner_by_layer.get(landing.layer)
    if owner is None:
        return []
    strong = relation.strengthened_side_enclosure_dbu
    minimum = relation.minimum_side_enclosure_dbu
    patch = Box(
        x1=via.bbox.x1 - strong, x2=via.bbox.x2 + strong,
        y1=via.bbox.y1 - minimum, y2=via.bbox.y2 + minimum,
    )
    if not (
        patch.intersects(landing.bbox)
        and scene.edit_halo.x1 <= patch.x1
        and scene.edit_halo.y1 <= patch.y1
        and patch.x2 <= scene.edit_halo.x2
        and patch.y2 <= scene.edit_halo.y2
    ):
        return []
    operation = AddBoundedPolygonOp(
        op="ADD_BOUNDED_POLYGON", source_owner_object_id=owner,
        layer=landing.layer, polygon_dbu=rectangle_points(patch),
    )
    return [_program(scene, operation, "PHASE3R_A_LOCAL_ADDITIVE_OVERRIDE")]


def _transform(point: Point, rotation: int, mirror: bool, dx: int, dy: int) -> Point:
    x, y = point.x, point.y
    if mirror:
        y = -y
    if rotation == 1:
        x, y = -y, x
    elif rotation == 2:
        x, y = -x, -y
    elif rotation == 3:
        x, y = y, -x
    return Point(x=x + dx, y=y + dy)


def _inverse(point: Point, rotation: int, mirror: bool, dx: int, dy: int) -> Point:
    x, y = point.x - dx, point.y - dy
    if rotation == 1:
        x, y = y, -x
    elif rotation == 2:
        x, y = -x, -y
    elif rotation == 3:
        x, y = -y, x
    if mirror:
        y = -y
    return Point(x=x, y=y)


def _infer_transform(
    local: list[Point], flattened: list[Point],
) -> tuple[int, bool, int, int] | None:
    if len(local) != len(flattened) or not local:
        return None
    # Polygon identity is independent of the serialized starting vertex and
    # traversal direction.  Hierarchy flattening may legally canonicalize
    # either while retaining the exact same rigid occurrence transform.
    target_orders = []
    for values in (flattened, list(reversed(flattened))):
        for offset in range(len(values)):
            target_orders.append(values[offset:] + values[:offset])
    for target in target_orders:
        for mirror in (False, True):
            for rotation in range(4):
                origin = _transform(local[0], rotation, mirror, 0, 0)
                dx, dy = target[0].x - origin.x, target[0].y - origin.y
                if [
                    _transform(point, rotation, mirror, dx, dy)
                    for point in local
                ] == target:
                    return rotation, mirror, dx, dy
    return None


def _occurrence_transform(local, geometry):
    if getattr(geometry, "occurrence_provenance", None) is not None:
        return verified_occurrence_transform(local, geometry)
    # Archived fixtures lack provenance; compilation still checks the source.
    return _infer_transform(local, list(geometry.polygon_dbu))


def _resize_box(box: Box, edge: str, coordinate: int) -> Box | None:
    values = {
        "LEFT": [coordinate, box.y1, box.x2, box.y2],
        "RIGHT": [box.x1, box.y1, coordinate, box.y2],
        "BOTTOM": [box.x1, coordinate, box.x2, box.y2],
        "TOP": [box.x1, box.y1, box.x2, coordinate],
    }[edge]
    try:
        result = Box.from_sequence(values)
    except ValueError:
        return None
    return result if result.width > 0 and result.height > 0 else None


def _blocks(box: Box, via: Box, edge: str) -> bool:
    return {
        "LEFT": box.x1 < via.x1,
        "RIGHT": box.x2 > via.x2,
        "BOTTOM": box.y1 < via.y1,
        "TOP": box.y2 > via.y2,
    }[edge]


def _coordinate(via: Box, edge: str) -> int:
    return {
        "LEFT": via.x1, "RIGHT": via.x2,
        "BOTTOM": via.y1, "TOP": via.y2,
    }[edge]


def _via_programs(scene: RepairScene, context: dict) -> list[RepairProgram]:
    relation = scene.relation
    assert isinstance(relation, ViaMetalRelation)
    participants = {item.participant_id: item for item in scene.participants}
    via = participants[relation.via_participant_id]
    metal = participants[relation.metal_participant_id]
    physical = [
        FlattenedPhysicalGeometry.model_validate(item)
        for item in context.get("phase3_physical_geometries", [])
        if item.get("geometry_id") in set(metal.physical_geometry_ids)
    ]
    objects = [
        LayoutObject.model_validate(item)
        for item in context.get("phase3_objects", [])
    ]
    editable_ids = {
        str(item["object_id"]) for item in context.get("editable_objects", [])
    }
    proposals = []
    for edge in relation.missing_coincident_via_edges:
        coordinate = _coordinate(via.bbox, edge)
        blockers = [item for item in physical if _blocks(item.bbox_dbu, via.bbox, edge)]
        direct = [
            item for item in blockers
            if item.instance_anchor_id is None
            and item.source_object_id in editable_ids
        ]
        shared = [item for item in blockers if item.instance_anchor_id is not None]
        if len({item.source_object_id for item in direct}) != 1 or not shared:
            continue
        owner_id = direct[0].source_object_id
        owner = next((item for item in objects if item.object_id == owner_id), None)
        if owner is None or _resize_box(owner.bbox_dbu, edge, coordinate) is None:
            continue
        by_instance = defaultdict(list)
        for item in shared:
            by_instance[item.instance_anchor_id].append(item)
        replacements = []
        displacement = abs({
            "LEFT": owner.bbox_dbu.x1, "RIGHT": owner.bbox_dbu.x2,
            "BOTTOM": owner.bbox_dbu.y1, "TOP": owner.bbox_dbu.y2,
        }[edge] - coordinate)
        failed = False
        for instance_anchor, occurrence_group in sorted(by_instance.items()):
            representative = occurrence_group[0]
            cell_objects = sorted(
                (
                    item for item in objects
                    if item.source_cell == representative.source_cell
                    and item.layer == representative.layer
                    and item.geometry_dbu
                ),
                key=lambda item: item.object_id,
            )
            by_source = {item.source_object_id: item for item in occurrence_group}
            seed_obj = next(
                (item for item in cell_objects if item.object_id in by_source), None,
            )
            seed_flat = by_source.get(seed_obj.object_id) if seed_obj else None
            transform = (
                _occurrence_transform(
                    list(seed_obj.geometry_dbu or []), seed_flat,
                ) if seed_obj and seed_flat else None
            )
            if transform is None or not cell_objects:
                failed = True
                break
            rotation, mirror, dx, dy = transform
            original_local = [
                canonical_polygon(list(item.geometry_dbu or []))
                for item in cell_objects
            ]
            replacement_local = []
            for polygon in original_local:
                global_points = [
                    _transform(point, rotation, mirror, dx, dy)
                    for point in polygon
                ]
                global_box = Box(
                    x1=min(item.x for item in global_points),
                    y1=min(item.y for item in global_points),
                    x2=max(item.x for item in global_points),
                    y2=max(item.y for item in global_points),
                )
                resized = (
                    _resize_box(global_box, edge, coordinate)
                    if _blocks(global_box, via.bbox, edge) else global_box
                )
                if resized is None:
                    failed = True
                    break
                displacement += abs({
                    "LEFT": global_box.x1, "RIGHT": global_box.x2,
                    "BOTTOM": global_box.y1, "TOP": global_box.y2,
                }[edge] - {
                    "LEFT": resized.x1, "RIGHT": resized.x2,
                    "BOTTOM": resized.y1, "TOP": resized.y2,
                }[edge])
                replacement_local.append(canonical_polygon([
                    _inverse(point, rotation, mirror, dx, dy)
                    for point in rectangle_points(resized)
                ]))
            if failed:
                break
            replacements.append(InstanceLayerReplacement(
                instance_anchor_id=str(instance_anchor),
                parent_cell=f"cell_{context['sample']['case_id']}",
                source_cell=str(representative.source_cell),
                layer=representative.layer, rotation=rotation, mirror=mirror,
                dx_dbu=dx, dy_dbu=dy,
                original_polygons_local_dbu=original_local,
                replacement_polygons_local_dbu=replacement_local,
            ))
        if failed or not replacements:
            continue
        operation = SpecializeInstanceBoundaryOp(
            op="SPECIALIZE_INSTANCE_BOUNDARY",
            source_owner_object_id=str(owner_id), edge=edge,
            target_coordinate_dbu=coordinate,
            instance_layer_replacements=replacements,
        )
        proposals.append((displacement, edge, _program(
            scene, operation, "PHASE3R_A_TARGET_INSTANCE_SPECIALIZATION",
        )))
    return [item[2] for item in sorted(proposals, key=lambda item: (item[0], item[1]))]



def _apply_rectilinear_operation(
    box: Box, operation: MoveObjectOp | ResizeEdgeOp,
) -> Box | None:
    if isinstance(operation, MoveObjectOp):
        return Box(
            x1=box.x1 + operation.dx_dbu,
            y1=box.y1 + operation.dy_dbu,
            x2=box.x2 + operation.dx_dbu,
            y2=box.y2 + operation.dy_dbu,
        )
    return _resize_box(box, operation.edge, operation.target_coordinate_dbu)


def _positive_area_overlap(left: Box, right: Box) -> bool:
    return (
        min(left.x2, right.x2) > max(left.x1, right.x1)
        and min(left.y2, right.y2) > max(left.y1, right.y1)
    )


def _closed_direct_metal_occurrences(
    direct: LayoutObject,
    physical: list[FlattenedPhysicalGeometry],
    *,
    max_occurrences: int = 4,
) -> list[FlattenedPhysicalGeometry] | None:
    """Resolve the bounded occurrence-local closure of a moved route shape.

    An overlapping same-layer occurrence contributes to the same merged
    physical metal.  Leaving that contributor behind while moving the direct
    route deterministically creates a bend or a split merged boundary.  The
    result contains one transform-bearing representative per exact occurrence.
    More than four occurrences is outside the existing bounded local authority
    and therefore fails closed.
    """

    by_instance: dict[str, FlattenedPhysicalGeometry] = {}
    for item in physical:
        if (
            item.layer != direct.layer
            or item.instance_anchor_id is None
            or item.source_object_id is None
            or item.source_cell is None
            or not _positive_area_overlap(item.bbox_dbu, direct.bbox_dbu)
        ):
            continue
        anchor = str(item.instance_anchor_id)
        current = by_instance.get(anchor)
        if current is None or item.geometry_id < current.geometry_id:
            by_instance[anchor] = item
    if len(by_instance) > max_occurrences:
        return None
    return [by_instance[key] for key in sorted(by_instance)]


def _closed_same_layer_component(
    seeds: list[FlattenedPhysicalGeometry],
    physical: list[FlattenedPhysicalGeometry],
    *,
    max_members: int = 8,
) -> list[FlattenedPhysicalGeometry] | None:
    """Return the bounded merged-metal component containing the seeds."""

    if not seeds or len({item.layer for item in seeds}) != 1:
        return None
    layer = seeds[0].layer
    candidates = {
        item.geometry_id: item for item in physical
        if item.layer == layer
    }
    selected = {
        item.geometry_id: item for item in seeds
    }
    changed = True
    while changed:
        changed = False
        for geometry_id, item in sorted(candidates.items()):
            if geometry_id in selected:
                continue
            if any(
                _positive_area_overlap(
                    item.bbox_dbu, member.bbox_dbu,
                )
                for member in selected.values()
            ):
                selected[geometry_id] = item
                changed = True
                if len(selected) > max_members:
                    return None
    return [selected[key] for key in sorted(selected)]


def _occurrence_translation_keeps_contact_candidates(
    seeds: list[FlattenedPhysicalGeometry],
    physical: list[FlattenedPhysicalGeometry],
    *,
    global_dx: int,
    global_dy: int,
) -> bool:
    """Return whether a whole-occurrence move keeps every known metal join.

    This is a bounded eligibility preview, not a connectivity proof.  When a
    complete occurrence (for example a via stack and both landings) moves as
    one unit, dragging every overlapping external contributor is unnecessary
    if each existing positive-area metal contact still has a contact candidate
    after the move.  Exact connectivity and all DRC relations remain subject
    to the isolated live verification path.  Missing lineage or a lost known
    contact fails closed into the existing merged-component closure.
    """

    anchors = {
        str(item.instance_anchor_id) for item in seeds
        if item.instance_anchor_id is not None
    }
    if not anchors or len(anchors) != len({
        item.instance_anchor_id for item in seeds
    }):
        return False
    moved = [
        item for item in physical
        if item.instance_anchor_id is not None
        and str(item.instance_anchor_id) in anchors
        and item.layer.startswith("M")
    ]
    if not moved:
        return False

    known_contact_count = 0
    for item in moved:
        shifted = Box(
            x1=item.bbox_dbu.x1 + global_dx,
            y1=item.bbox_dbu.y1 + global_dy,
            x2=item.bbox_dbu.x2 + global_dx,
            y2=item.bbox_dbu.y2 + global_dy,
        )
        for other in physical:
            other_anchor = (
                str(other.instance_anchor_id)
                if other.instance_anchor_id is not None else None
            )
            if (
                other.layer != item.layer
                or other_anchor in anchors
                or not _positive_area_overlap(
                    item.bbox_dbu, other.bbox_dbu,
                )
            ):
                continue
            known_contact_count += 1
            if not _positive_area_overlap(shifted, other.bbox_dbu):
                return False
    return known_contact_count > 0


def _may_preserve_external_metal_contributors(
    scene: RepairScene,
    seeds: list[FlattenedPhysicalGeometry],
    physical: list[FlattenedPhysicalGeometry],
    *,
    global_dx: int,
    global_dy: int,
) -> bool:
    """Return whether contact preservation is a sufficient carrier closure.

    A conditional-PRL witness is measured on the merged conductor boundary.
    Moving only an occurrence-local landing can keep positive-area contact yet
    merely replace the official marker on the unchanged merged component.  In
    that relation the component, rather than the contact, is the deterministic
    physical unit.  Other spacing/alignment relations may retain external
    contributors when every known contact remains; live DRC and connectivity
    still provide the acceptance truth.
    """

    if (
        isinstance(scene.relation, SpacingRelation)
        and scene.relation.measurement_kind == "CONDITIONAL_PRL"
    ):
        return False
    return _occurrence_translation_keeps_contact_candidates(
        seeds,
        physical,
        global_dx=global_dx,
        global_dy=global_dy,
    )


def _translated_occurrence_replacements(
    representative: FlattenedPhysicalGeometry,
    *,
    objects: list[LayoutObject],
    object_by_id: dict[str, LayoutObject],
    case_id: str,
    global_dx: int,
    global_dy: int,
    selected_layers: set[str] | None = None,
) -> list[InstanceLayerReplacement] | None:
    source = object_by_id.get(str(representative.source_object_id))
    if source is None or not source.geometry_dbu:
        return None
    transform = _occurrence_transform(
        list(source.geometry_dbu), representative,
    )
    if transform is None:
        return None
    rotation, mirror, dx, dy = transform
    layers = sorted({
        item.layer for item in objects
        if item.source_cell == representative.source_cell
        and item.geometry_dbu
        and (
            selected_layers is None
            or item.layer in selected_layers
        )
    })
    if not layers or len(layers) > 4:
        return None
    replacements = []
    for layer in layers:
        layer_objects = sorted(
            (
                item for item in objects
                if item.source_cell == representative.source_cell
                and item.layer == layer
                and item.geometry_dbu
            ),
            key=lambda item: item.object_id,
        )
        original_local = [
            canonical_polygon(list(item.geometry_dbu or []))
            for item in layer_objects
        ]
        replacement_local = [
            canonical_polygon([
                _inverse(
                    Point(
                        x=global_point.x + global_dx,
                        y=global_point.y + global_dy,
                    ),
                    rotation, mirror, dx, dy,
                )
                for global_point in (
                    _transform(point, rotation, mirror, dx, dy)
                    for point in polygon
                )
            ])
            for polygon in original_local
        ]
        replacements.append(InstanceLayerReplacement(
            instance_anchor_id=str(representative.instance_anchor_id),
            parent_cell=f"cell_{case_id}",
            source_cell=str(representative.source_cell),
            layer=layer,
            rotation=rotation,
            mirror=mirror,
            dx_dbu=dx,
            dy_dbu=dy,
            original_polygons_local_dbu=original_local,
            replacement_polygons_local_dbu=replacement_local,
        ))
    return replacements


def _local_fragment_translation_replacement(
    representative: FlattenedPhysicalGeometry,
    *,
    source_id: str,
    source: LayoutObject,
    objects: list[LayoutObject],
    case_id: str,
    global_dx: int,
    global_dy: int,
    marker_edge: Edge,
) -> InstanceLayerReplacement | None:
    """Copy-on-write exactly one source polygon in one exact occurrence.

    This is the bounded authority for a local route fragment inside a shared
    standard-cell definition.  It never edits the shared definition and never
    translates unrelated layers or polygons in that occurrence.
    """

    if (
        representative.instance_anchor_id is None
        or representative.source_cell is None
        or not source.geometry_dbu
        or source.layer != representative.layer
    ):
        return None
    transform = _occurrence_transform(
        list(source.geometry_dbu), representative,
    )
    if transform is None:
        return None
    rotation, mirror, dx, dy = transform
    layer_objects = sorted(
        (
            item for item in objects
            if item.source_cell == representative.source_cell
            and item.layer == representative.layer
            and item.geometry_dbu
        ),
        key=lambda item: item.object_id,
    )
    if not layer_objects or sum(
        item.object_id == source_id for item in layer_objects
    ) != 1:
        return None
    original_local = [
        canonical_polygon(list(item.geometry_dbu or []))
        for item in layer_objects
    ]
    replacement_local = []
    for item, polygon in zip(layer_objects, original_local):
        if item.object_id != source_id:
            replacement_local.append(polygon)
            continue
        global_polygon = [
            _transform(point, rotation, mirror, dx, dy)
            for point in polygon
        ]
        changed = _translate_containing_boundary_edge(
            global_polygon,
            marker_edge=marker_edge,
            dx=global_dx,
            dy=global_dy,
        )
        if changed is None:
            return None
        replacement_local.append(canonical_polygon([
            _inverse(point, rotation, mirror, dx, dy)
            for point in changed
        ]))
    return InstanceLayerReplacement(
        instance_anchor_id=str(representative.instance_anchor_id),
        parent_cell=f"cell_{case_id}",
        source_cell=str(representative.source_cell),
        layer=representative.layer,
        rotation=rotation,
        mirror=mirror,
        dx_dbu=dx,
        dy_dbu=dy,
        original_polygons_local_dbu=original_local,
        replacement_polygons_local_dbu=replacement_local,
        occurrence_local_fragment=True,
    )


def _translate_containing_boundary_edge(
    polygon: list[Point],
    *,
    marker_edge: Edge,
    dx: int,
    dy: int,
) -> list[Point] | None:
    """Move the unique polygon edge containing an exact signoff marker edge."""

    marker_horizontal = marker_edge.start.y == marker_edge.end.y
    marker_vertical = marker_edge.start.x == marker_edge.end.x
    if not (marker_horizontal ^ marker_vertical):
        return None
    if (marker_horizontal and dx != 0) or (marker_vertical and dy != 0):
        return None

    def contains(segment_start: Point, segment_end: Point) -> bool:
        if marker_horizontal:
            return (
                segment_start.y == segment_end.y == marker_edge.start.y
                and min(segment_start.x, segment_end.x)
                <= min(marker_edge.start.x, marker_edge.end.x)
                and max(marker_edge.start.x, marker_edge.end.x)
                <= max(segment_start.x, segment_end.x)
            )
        return (
            segment_start.x == segment_end.x == marker_edge.start.x
            and min(segment_start.y, segment_end.y)
            <= min(marker_edge.start.y, marker_edge.end.y)
            and max(marker_edge.start.y, marker_edge.end.y)
            <= max(segment_start.y, segment_end.y)
        )

    matching = [
        index for index, point in enumerate(polygon)
        if contains(point, polygon[(index + 1) % len(polygon)])
    ]
    if len(matching) != 1:
        return None
    index = matching[0]
    result = list(polygon)
    for vertex_index in (index, (index + 1) % len(result)):
        point = result[vertex_index]
        result[vertex_index] = Point(x=point.x + dx, y=point.y + dy)
    if any(
        left.x != right.x and left.y != right.y
        for left, right in zip(result, result[1:] + result[:1])
    ):
        return None
    return canonical_polygon(result)


def _closed_occurrence_layer_scope(
    repair_family: RepairFamily,
    layer: str,
) -> set[str] | None:
    """Return the layers that must move with a closed occurrence carrier.

    A spacing translation changes the physical location of a connected metal
    participant.  An overlapping via occurrence is therefore a stack-level
    carrier: moving only its selected metal landing deterministically breaks
    the via/landing relations on the other layers.  Alignment, by contrast,
    intentionally adjusts the selected routing-layer contributor while the
    via cut remains fixed and is still checked by live DRC/connectivity.
    """

    return None if repair_family == RepairFamily.SPACING else {layer}


def _contains_box(outer: Box, inner: Box) -> bool:
    return (
        outer.x1 <= inner.x1 <= inner.x2 <= outer.x2
        and outer.y1 <= inner.y1 <= inner.y2 <= outer.y2
    )


def _closed_containing_box(
    *,
    original_landing: Box,
    original_via: Box,
    changed_via: Box,
    current_component: list[Box],
    original_via_component: list[Box] | None = None,
) -> Box | None:
    """Return the minimal landing closure required by an enlarged via cut.

    The source landing is eligible only when it contained the original cut.
    If another contributor in the current same-layer component already contains
    the changed cut, no source expansion is necessary.  Otherwise grow only
    the occurrence-local landing to the bounding union.  This is a deterministic
    containment fact; KLayout still decides every DRC/enclosure consequence.
    """

    if not _contains_box(original_landing, original_via):
        return None
    if any(_contains_box(item, changed_via) for item in current_component):
        return original_landing
    component = [original_via, *(original_via_component or [])]
    original_extent = Box(
        x1=min(item.x1 for item in component),
        y1=min(item.y1 for item in component),
        x2=max(item.x2 for item in component),
        y2=max(item.y2 for item in component),
    )
    left_margin = max(0, original_extent.x1 - original_landing.x1)
    right_margin = max(0, original_landing.x2 - original_extent.x2)
    bottom_margin = max(0, original_extent.y1 - original_landing.y1)
    top_margin = max(0, original_landing.y2 - original_extent.y2)
    return Box(
        x1=min(
            original_landing.x1,
            changed_via.x1 - left_margin
            if changed_via.x1 < original_landing.x1 else changed_via.x1,
        ),
        y1=min(
            original_landing.y1,
            changed_via.y1 - bottom_margin
            if changed_via.y1 < original_landing.y1 else changed_via.y1,
        ),
        x2=max(
            original_landing.x2,
            changed_via.x2 + right_margin
            if changed_via.x2 > original_landing.x2 else changed_via.x2,
        ),
        y2=max(
            original_landing.y2,
            changed_via.y2 + top_margin
            if changed_via.y2 > original_landing.y2 else changed_via.y2,
        ),
    )


def _via_stack_containment_replacements(
    representative: FlattenedPhysicalGeometry,
    *,
    changed_via: Box,
    relation_metal_layer: str,
    objects: list[LayoutObject],
    object_by_id: dict[str, LayoutObject],
    physical: list[FlattenedPhysicalGeometry],
    transform: tuple[int, bool, int, int],
    case_id: str,
) -> list[InstanceLayerReplacement] | None:
    """Close an occurrence-local via resize over its adjacent metal landings."""

    if representative.instance_anchor_id is None or representative.source_cell is None:
        return None
    rotation, mirror, dx, dy = transform
    by_layer: dict[str, list[FlattenedPhysicalGeometry]] = defaultdict(list)
    adjacent_layers: set[str] | None = None
    if (
        representative.layer.startswith("V")
        and representative.layer[1:].isdigit()
    ):
        via_index = int(representative.layer[1:])
        adjacent_layers = {f"M{via_index}", f"M{via_index + 1}"}
    for item in physical:
        if (
            item.instance_anchor_id == representative.instance_anchor_id
            and item.source_cell == representative.source_cell
            and item.layer.startswith("M")
            and (adjacent_layers is None or item.layer in adjacent_layers)
            and item.layer != relation_metal_layer
            and item.source_object_id is not None
            and _contains_box(item.bbox_dbu, representative.bbox_dbu)
        ):
            by_layer[item.layer].append(item)
    projected_sources = {
        (item.layer, str(item.source_object_id))
        for candidates in by_layer.values() for item in candidates
    }
    for source in objects:
        if (
            source.source_cell != representative.source_cell
            or not source.layer.startswith("M")
            or (adjacent_layers is not None and source.layer not in adjacent_layers)
            or source.layer == relation_metal_layer
            or not source.geometry_dbu
            or (source.layer, source.object_id) in projected_sources
        ):
            continue
        polygon = canonical_polygon([
            _transform(point, rotation, mirror, dx, dy)
            for point in source.geometry_dbu
        ])
        bbox = Box(
            x1=min(point.x for point in polygon),
            y1=min(point.y for point in polygon),
            x2=max(point.x for point in polygon),
            y2=max(point.y for point in polygon),
        )
        if not _contains_box(bbox, representative.bbox_dbu):
            continue
        by_layer[source.layer].append(FlattenedPhysicalGeometry(
            geometry_id="source_projection_" + stable_hash([
                representative.instance_anchor_id,
                source.object_id,
                polygon,
            ])[:20],
            occurrence_provenance=representative.occurrence_provenance,
            source_object_id=source.object_id,
            source_anchor_id=getattr(source, "source_anchor_id", None),
            source_cell=representative.source_cell,
            instance_anchor_id=representative.instance_anchor_id,
            hierarchy_path=list(representative.hierarchy_path),
            layer=source.layer,
            polygon_dbu=polygon,
            bbox_dbu=bbox,
            ownership_quality="EXACT_SHARED_CELL",
            editable=False,
            source_instance_count=representative.source_instance_count,
            geometry_hash=stable_hash([
                source.object_id, polygon,
                representative.instance_anchor_id,
            ]),
        ))
    values: list[InstanceLayerReplacement] = []
    for layer, candidates in sorted(by_layer.items()):
        landing = min(
            candidates,
            key=lambda item: (
                item.bbox_dbu.width * item.bbox_dbu.height,
                item.geometry_id,
            ),
        )
        # This operation only GROWS one already-authorized occurrence landing;
        # it does not move its transitive connected metal component. A bounded
        # moving-carrier limit therefore cannot determine expressibility here.
        # Require an exact rectangle, preserving every old point/contact. All
        # halo, source authority, operation-count and live safety gates remain.
        exact_rectangle = lambda item: canonical_polygon(item.polygon_dbu) == canonical_polygon(rectangle_points(item.bbox_dbu))
        if not exact_rectangle(landing) or not exact_rectangle(representative):
            return None
        if _contains_box(landing.bbox_dbu, changed_via):
            continue
        # A directly joined, unchanged exact contributor can already cover the
        # whole changed cut. This is a read proof, never authority to edit it.
        covers = [item.bbox_dbu for item in physical
            if item.layer == layer and exact_rectangle(item)
            and item.bbox_dbu.intersection_area(landing.bbox_dbu) > 0
            and _contains_box(item.bbox_dbu, changed_via)]
        closed = _closed_containing_box(
            original_landing=landing.bbox_dbu,
            original_via=representative.bbox_dbu,
            changed_via=changed_via,
            current_component=covers,
            original_via_component=[
                item.bbox_dbu for item in physical
                if item.layer == representative.layer
                and item.instance_anchor_id == representative.instance_anchor_id
                and item.source_cell == representative.source_cell
            ],
        )
        if closed is not None and not _contains_box(closed, landing.bbox_dbu):
            return None
        if closed is None:
            return None
        if closed == landing.bbox_dbu:
            continue
        layer_objects = sorted(
            (
                item for item in objects
                if item.source_cell == representative.source_cell
                and item.layer == layer and item.geometry_dbu
            ),
            key=lambda item: item.object_id,
        )
        target_index = next(
            (
                index for index, item in enumerate(layer_objects)
                if item.object_id == landing.source_object_id
            ),
            None,
        )
        if target_index is None:
            return None
        original_local = [
            canonical_polygon(list(item.geometry_dbu or []))
            for item in layer_objects
        ]
        replacement_local = list(original_local)
        replacement_local[target_index] = canonical_polygon([
            _inverse(point, rotation, mirror, dx, dy)
            for point in rectangle_points(closed)
        ])
        values.append(InstanceLayerReplacement(
            instance_anchor_id=str(representative.instance_anchor_id),
            parent_cell=f"cell_{case_id}",
            source_cell=str(representative.source_cell),
            layer=layer,
            rotation=rotation,
            mirror=mirror,
            dx_dbu=dx,
            dy_dbu=dy,
            original_polygons_local_dbu=original_local,
            replacement_polygons_local_dbu=replacement_local,
        ))
    return values


def _group_specialization_operations(
    replacements_by_owner: dict[str, list[InstanceLayerReplacement]],
) -> list[SpecializeInstanceLayerOp] | None:
    """Encode a closed occurrence group in bounded operations per owner."""

    unique: dict[
        tuple[str, str], tuple[str, InstanceLayerReplacement]
    ] = {}
    for owner, replacements in sorted(replacements_by_owner.items()):
        for replacement in replacements:
            identity = (
                replacement.instance_anchor_id,
                replacement.layer,
            )
            existing = unique.get(identity)
            if existing is not None:
                if (
                    existing[0] != owner
                    or existing[1].model_dump(mode="json")
                    != replacement.model_dump(mode="json")
                ):
                    return None
                continue
            unique[identity] = (owner, replacement)
    # The bounded unit is an occurrence, not a globally unique layer name.
    # Two adjacent via stacks can legitimately span five distinct routing/cut
    # layers while still compiling to at most four atomic specialization ops.
    # Each occurrence builder is already capped at four layers, so retain the
    # existing four-occurrence bound and its derived 4 x 4 replacement bound.
    if (
        len({key[0] for key in unique}) > 4
        or len(unique) > 16
    ):
        return None
    grouped: dict[str, list[InstanceLayerReplacement]] = defaultdict(list)
    for _, (owner, replacement) in sorted(unique.items()):
        grouped[owner].append(replacement)
    # The wire schema bounds one specialization operation to four layer
    # replacements. Preserve that contract and split a multi-layer closed
    # carrier into adjacent operations; RepairProgram still bounds the complete
    # atomic program to four operations.
    operation_specs = [
        (owner, items[offset:offset + 4])
        for owner, items in sorted(grouped.items())
        for offset in range(0, len(items), 4)
    ]
    if len(operation_specs) > 4:
        return None
    try:
        return [
            SpecializeInstanceLayerOp(
                op="SPECIALIZE_INSTANCE_LAYER",
                source_owner_object_id=owner,
                instance_layer_replacements=items,
            )
            for owner, items in operation_specs
        ]
    except ValueError:
        return None


def build_solution_specialization_programs(
    scene: RepairScene,
    solutions: list[GeometrySolution],
    authority: AuthorityResolution,
    context: dict,
    *,
    rejections: list[dict] | None = None,
) -> list[RepairProgram]:
    # Keep exact DBU calculation in the deterministic solver; this adapter only
    # replaces an unsafe shared-definition edit with a target-occurrence rewrite.
    def reject(solution, code, source_id=None):
        if rejections is not None:
            rejections.append({
                "stage": "SPECIALIZATION_LOWERING",
                "code": code,
                "solution_id": solution.solution_id,
                "selected_dof_ids": list(solution.selected_dof_ids),
                "source_object_id": source_id,
            })

    objects = [
        LayoutObject.model_validate(item)
        for item in context.get("phase3_objects", [])
    ]
    object_by_id = {item.object_id: item for item in objects}
    physical = [
        FlattenedPhysicalGeometry.model_validate(item)
        for item in context.get("phase3_physical_geometries", [])
    ]
    participant_by_source = defaultdict(list)
    for participant in scene.participants:
        if not participant.instance_anchor_ids:
            continue
        for source_id in participant.source_object_ids:
            participant_by_source[source_id].append(participant)
    projection_geometry_ids_by_source: dict[str, set[str]] = defaultdict(set)
    for projection in scene.hierarchy_projections:
        for boundary in projection.boundary_contributors:
            for source_id in boundary.source_object_ids:
                projection_geometry_ids_by_source[source_id].update(
                    boundary.physical_geometry_ids
                )

    editable_ids = {
        str(item["object_id"]) for item in context.get("editable_objects", [])
    }
    programs: list[RepairProgram] = []
    scene_dof_by_id = {
        item.dof_id: item for item in generate_repair_dofs(scene)
    }
    via_participant = None
    relation_metal_layer = None
    if (
        scene.repair_family == RepairFamily.VIA_METAL_CONTEXTUAL
        and isinstance(scene.relation, ViaMetalRelation)
    ):
        participants_by_id = {
            item.participant_id: item for item in scene.participants
        }
        via_participant = participants_by_id.get(
            scene.relation.via_participant_id
        )
        relation_metal = participants_by_id.get(
            scene.relation.metal_participant_id
        )
        relation_metal_layer = (
            relation_metal.layer if relation_metal is not None else None
        )
    solution_source_ids = {
        str(operation.target_object_id)
        for solution in solutions
        for operation in solution.operations
        if getattr(operation, "target_object_id", None)
    }
    specialization_anchor = next((
        item.object_id for item in sorted(objects, key=lambda item: item.object_id)
        if item.object_id in solution_source_ids
        and item.source_anchor_id and item.insertion_source_span is not None
    ), None)
    for solution in solutions:
        operations_by_source = defaultdict(list)
        closed_direct_moves: list[tuple[str, int, int]] = []
        closed_occurrence_moves: list[
            tuple[list[FlattenedPhysicalGeometry], int, int]
        ] = []
        selected_participants_by_source: dict[str, set[str]] = defaultdict(set)
        for dof_id in solution.selected_dof_ids:
            dof = scene_dof_by_id.get(dof_id)
            if dof is None:
                continue
            for source_id in dof.source_object_ids:
                selected_participants_by_source[source_id].add(
                    dof.participant_id
                )
        unsupported = False
        for operation in solution.operations:
            if not isinstance(operation, (MoveObjectOp, ResizeEdgeOp)):
                unsupported = True
                break
            operations_by_source[operation.target_object_id].append(operation)
        if unsupported or not operations_by_source:
            reject(solution, "LOWERING_UNSUPPORTED_OPERATION" if unsupported else "LOWERING_EMPTY_OPERATIONS")
            continue

        replacements: list[InstanceLayerReplacement] = []
        replacements_by_owner: dict[
            str, list[InstanceLayerReplacement]
        ] = defaultdict(list)
        direct_operations = []
        failed = False
        for source_id, source_operations in sorted(operations_by_source.items()):
            participants = participant_by_source.get(source_id, [])
            # Region editability is source-specific.  A merged participant may
            # also contain instance-backed contributors; that aggregate fact
            # must not turn an explicitly editable direct source into an
            # unrelated occurrence specialization.
            if source_id in editable_ids:
                direct_operations.extend(source_operations)
                translations = [
                    item for item in source_operations
                    if isinstance(item, MoveObjectOp)
                ]
                if (
                    scene.repair_family in {
                        RepairFamily.SPACING,
                        RepairFamily.ALIGNMENT,
                    }
                    and len(translations) == len(source_operations)
                ):
                    closed_direct_moves.append((
                        source_id,
                        sum(item.dx_dbu for item in translations),
                        sum(item.dy_dbu for item in translations),
                    ))
                continue
            selected_participant_ids = selected_participants_by_source.get(
                source_id, set()
            )
            if selected_participant_ids:
                participants = [
                    item for item in participants
                    if item.participant_id in selected_participant_ids
                ]
            if len(participants) > 1:
                reject(solution, "LOWERING_MULTIPLE_PARTICIPANTS", source_id)
                failed = True
                break
            participant = participants[0] if participants else None
            target_ids = (
                set(participant.physical_geometry_ids) if participant else
                projection_geometry_ids_by_source.get(source_id, set())
            )
            occurrences = sorted(
                (
                    item for item in physical
                    if item.geometry_id in target_ids
                    and item.source_object_id == source_id
                    and item.instance_anchor_id is not None
                ),
                key=lambda item: (str(item.instance_anchor_id), item.geometry_id),
            )
            if not occurrences:
                reject(solution, "LOWERING_OCCURRENCE_NOT_RESOLVED", source_id)
                failed = True
                break
            participant_layer = (
                participant.layer if participant else occurrences[0].layer
            )
            source = object_by_id.get(source_id)
            source_is_anchor = bool(
                source is not None and source.source_anchor_id
                and source.insertion_source_span is not None
            )
            owner = source_id if source_is_anchor else (
                authority.insertion_owner_by_layer.get(participant_layer)
            )
            if owner is None and authority.insertion_owner_by_layer:
                # SPECIALIZE_INSTANCE_LAYER only uses this object as a stable
                # editable source anchor for the generated declarations; its
                # geometry is not mutated.  Via layers commonly have no
                # top-level polygon, so fall back deterministically to an
                # existing Region insertion owner while preserving the exact
                # target occurrence in the replacement payload.
                owner = sorted(authority.insertion_owner_by_layer.values())[0]
            if owner is None:
                # An occurrence-only Region can legitimately have no editable
                # top-level polygon. Copy-on-write specialization still needs
                # a stable declaration insertion anchor, not permission to
                # mutate the shared definition.
                owner = specialization_anchor
            if owner is None:
                reject(solution, "LOWERING_DECLARATION_OWNER_MISSING", source_id)
                failed = True
                break
            for representative in occurrences:
                transform = (
                    _occurrence_transform(
                        list(source.geometry_dbu or []),
                        representative,
                    )
                    if source is not None else None
                )
                if transform is None:
                    reject(solution, "LOWERING_TRANSFORM_UNRESOLVED", source_id)
                    failed = True
                    break
                rotation, mirror, dx, dy = transform
                translations = [
                    item for item in source_operations
                    if isinstance(item, MoveObjectOp)
                ]
                if (
                    len(translations) == len(source_operations)
                ):
                    global_dx = sum(item.dx_dbu for item in translations)
                    global_dy = sum(item.dy_dbu for item in translations)
                    cell_layers = sorted({
                        item.layer for item in objects
                        if item.source_cell == representative.source_cell
                        and item.geometry_dbu
                    })
                    # A many-layer standard-cell occurrence is not a bounded
                    # mutation carrier for a local spacing edge.  With exact
                    # source/instance lineage, specialize only the offending
                    # source polygon and preserve the rest of the occurrence.
                    if (
                        scene.repair_family == RepairFamily.SPACING
                        and len(cell_layers) > 4
                    ):
                        relation = scene.relation
                        participants_by_id = {
                            item.participant_id: item
                            for item in scene.participants
                        }
                        marker_edge = None
                        if isinstance(relation, SpacingRelation):
                            participant_a = participants_by_id.get(
                                relation.participant_a_id
                            )
                            participant_b = participants_by_id.get(
                                relation.participant_b_id
                            )
                            if (
                                participant_a is not None
                                and source_id
                                in set(participant_a.source_object_ids)
                            ):
                                marker_edge = relation.participant_a_marker_edge
                            elif (
                                participant_b is not None
                                and source_id
                                in set(participant_b.source_object_ids)
                            ):
                                marker_edge = relation.participant_b_marker_edge
                        if marker_edge is None:
                            reject(solution, "LOWERING_FRAGMENT_EDGE_MISSING", source_id)
                            failed = True
                            break
                        local = _local_fragment_translation_replacement(
                            representative,
                            source_id=source_id,
                            source=source,
                            objects=objects,
                            case_id=str(context["sample"]["case_id"]),
                            global_dx=global_dx,
                            global_dy=global_dy,
                            marker_edge=marker_edge,
                        )
                        if local is None:
                            reject(solution, "LOWERING_FRAGMENT_NOT_REPRESENTABLE", source_id)
                            failed = True
                            break
                        replacements.append(local)
                        replacements_by_owner[owner].append(local)
                        continue
                    if (
                        representative is occurrences[0]
                        and scene.repair_family in {
                            RepairFamily.SPACING,
                            RepairFamily.ALIGNMENT,
                        }
                    ):
                        closed_occurrence_moves.append((
                            occurrences, global_dx, global_dy,
                        ))
                    for layer in cell_layers:
                        layer_objects = sorted(
                            (
                                item for item in objects
                                if item.source_cell == representative.source_cell
                                and item.layer == layer and item.geometry_dbu
                            ),
                            key=lambda item: item.object_id,
                        )
                        original_local = [
                            canonical_polygon(list(item.geometry_dbu or []))
                            for item in layer_objects
                        ]
                        replacement_local = []
                        for polygon in original_local:
                            replacement_local.append(canonical_polygon([
                                _inverse(
                                    Point(
                                        x=global_point.x + global_dx,
                                        y=global_point.y + global_dy,
                                    ),
                                    rotation, mirror, dx, dy,
                                )
                                for global_point in (
                                    _transform(point, rotation, mirror, dx, dy)
                                    for point in polygon
                                )
                            ]))
                        replacement = InstanceLayerReplacement(
                            instance_anchor_id=str(representative.instance_anchor_id),
                            parent_cell=f"cell_{context['sample']['case_id']}",
                            source_cell=str(representative.source_cell),
                            layer=layer, rotation=rotation,
                            mirror=mirror, dx_dbu=dx, dy_dbu=dy,
                            original_polygons_local_dbu=original_local,
                            replacement_polygons_local_dbu=replacement_local,
                        )
                        replacements.append(replacement)
                        replacements_by_owner[owner].append(replacement)
                    continue

                cell_objects = sorted(
                    (
                        item for item in objects
                        if item.source_cell == representative.source_cell
                        and item.layer == representative.layer
                        and item.geometry_dbu
                    ),
                    key=lambda item: item.object_id,
                )
                if not cell_objects:
                    reject(solution, "LOWERING_CELL_GEOMETRY_MISSING", source_id)
                    failed = True
                    break
                original_local = [
                    canonical_polygon(list(item.geometry_dbu or []))
                    for item in cell_objects
                ]
                replacement_local = []
                changed_target_box = None
                for item, polygon in zip(cell_objects, original_local):
                    if item.object_id != source_id:
                        replacement_local.append(polygon)
                        continue
                    global_points = [
                        _transform(point, rotation, mirror, dx, dy)
                        for point in polygon
                    ]
                    if len(global_points) != 4:
                        reject(solution, "LOWERING_NON_RECTANGULAR_TARGET", source_id)
                        failed = True
                        break
                    changed = Box(
                        x1=min(point.x for point in global_points),
                        y1=min(point.y for point in global_points),
                        x2=max(point.x for point in global_points),
                        y2=max(point.y for point in global_points),
                    )
                    for operation in source_operations:
                        changed = _apply_rectilinear_operation(changed, operation)
                        if changed is None:
                            reject(solution, "LOWERING_INVALID_RECTANGLE_OPERATION", source_id)
                            failed = True
                            break
                    if failed:
                        break
                    replacement_local.append(canonical_polygon([
                        _inverse(point, rotation, mirror, dx, dy)
                        for point in rectangle_points(changed)
                    ]))
                    changed_target_box = changed
                if failed:
                    break
                replacement = InstanceLayerReplacement(
                    instance_anchor_id=str(representative.instance_anchor_id),
                    parent_cell=f"cell_{context["sample"]["case_id"]}",
                    source_cell=str(representative.source_cell),
                    layer=representative.layer,
                    rotation=rotation,
                    mirror=mirror,
                    dx_dbu=dx,
                    dy_dbu=dy,
                    original_polygons_local_dbu=original_local,
                    replacement_polygons_local_dbu=replacement_local,
                )
                replacements.append(replacement)
                replacements_by_owner[owner].append(replacement)
                if (
                    changed_target_box is not None
                    and via_participant is not None
                    and source_id in set(via_participant.source_object_ids)
                    and relation_metal_layer is not None
                ):
                    closure = _via_stack_containment_replacements(
                        representative,
                        changed_via=changed_target_box,
                        relation_metal_layer=relation_metal_layer,
                        objects=objects,
                        object_by_id=object_by_id,
                        physical=physical,
                        transform=transform,
                        case_id=str(context["sample"]["case_id"]),
                    )
                    if closure is None:
                        reject(solution, "LOWERING_VIA_STACK_CLOSURE_UNAVAILABLE", source_id)
                        failed = True
                        break
                    replacements.extend(closure)
                    replacements_by_owner[owner].extend(closure)
            if failed:
                break
        if (
            not failed
            and not closed_direct_moves
            and closed_occurrence_moves
        ):
            for seeds, global_dx, global_dy in closed_occurrence_moves:
                if _may_preserve_external_metal_contributors(
                    scene,
                    seeds,
                    physical,
                    global_dx=global_dx,
                    global_dy=global_dy,
                ):
                    # The seed occurrence was already emitted as a complete
                    # stack-level specialization.  Preserve external metal
                    # contributors when all known joins remain candidates;
                    # live DRC/connectivity remains the acceptance authority.
                    continue
                component = _closed_same_layer_component(seeds, physical)
                if component is None:
                    reject(solution, "LOWERING_COMPONENT_UNBOUNDED")
                    failed = True
                    break
                seed_anchors = {
                    str(item.instance_anchor_id) for item in seeds
                    if item.instance_anchor_id is not None
                }
                direct_by_source = {
                    str(item.source_object_id): item
                    for item in component
                    if item.instance_anchor_id is None
                    and item.source_object_id is not None
                }
                for source_id in sorted(direct_by_source):
                    if source_id not in editable_ids:
                        reject(solution, "LOWERING_CLOSURE_AUTHORITY_DENIED", source_id)
                        failed = True
                        break
                    existing = [
                        item for item in direct_operations
                        if isinstance(item, MoveObjectOp)
                        and item.target_object_id == source_id
                    ]
                    if existing and any(
                        item.dx_dbu != global_dx
                        or item.dy_dbu != global_dy
                        for item in existing
                    ):
                        reject(solution, "LOWERING_INCONSISTENT_COUPLED_MOVE", source_id)
                        failed = True
                        break
                    if not existing:
                        direct_operations.append(MoveObjectOp(
                            op="MOVE_OBJECT",
                            target_object_id=source_id,
                            dx_dbu=global_dx,
                            dy_dbu=global_dy,
                        ))
                if failed:
                    break
                occurrence_by_anchor: dict[
                    str, FlattenedPhysicalGeometry
                ] = {}
                for item in component:
                    if (
                        item.instance_anchor_id is None
                        or item.source_object_id is None
                        or item.source_cell is None
                    ):
                        continue
                    anchor = str(item.instance_anchor_id)
                    current = occurrence_by_anchor.get(anchor)
                    if current is None or item.geometry_id < current.geometry_id:
                        occurrence_by_anchor[anchor] = item
                if len(occurrence_by_anchor) > 4:
                    reject(solution, "LOWERING_OCCURRENCE_BOUND_EXCEEDED")
                    failed = True
                    break
                for anchor, representative in sorted(
                    occurrence_by_anchor.items()
                ):
                    if anchor in seed_anchors:
                        continue
                    occurrence_replacements = (
                        _translated_occurrence_replacements(
                            representative,
                            objects=objects,
                            object_by_id=object_by_id,
                            case_id=str(context["sample"]["case_id"]),
                            global_dx=global_dx,
                            global_dy=global_dy,
                            selected_layers=_closed_occurrence_layer_scope(
                                scene.repair_family,
                                representative.layer,
                            ),
                        )
                    )
                    source = object_by_id.get(
                        str(representative.source_object_id)
                    )
                    if occurrence_replacements is None or source is None:
                        failed = True
                        break
                    source_is_anchor = bool(
                        source.source_anchor_id
                        and source.insertion_source_span is not None
                    )
                    owner = (
                        source.object_id if source_is_anchor else
                        authority.insertion_owner_by_layer.get(
                            representative.layer
                        )
                    )
                    if owner is None:
                        owner = specialization_anchor
                    if owner is None:
                        failed = True
                        break
                    replacements.extend(occurrence_replacements)
                    replacements_by_owner[owner].extend(
                        occurrence_replacements
                    )
                if failed:
                    break
        if not failed:
            for source_id, global_dx, global_dy in closed_direct_moves:
                direct = object_by_id.get(source_id)
                if direct is None:
                    reject(solution, "LOWERING_DIRECT_SOURCE_MISSING", source_id)
                    failed = True
                    break
                occurrences = _closed_direct_metal_occurrences(
                    direct, physical,
                )
                if occurrences is None:
                    reject(solution, "LOWERING_DIRECT_OCCURRENCE_BOUND_EXCEEDED", source_id)
                    failed = True
                    break
                # A solver-selected shared-cell contributor may have produced
                # a whole-cell translation earlier in this loop.  Once the
                # direct merged-metal carrier is known, replace that broad
                # mutation with the smallest closed relation: every
                # overlapping contributor on the moved metal layer, while
                # keeping lower via/landing layers fixed for connectivity.
                closed_anchors = {
                    str(item.instance_anchor_id) for item in occurrences
                }
                replacements = [
                    item for item in replacements
                    if item.instance_anchor_id not in closed_anchors
                ]
                for owner in list(replacements_by_owner):
                    replacements_by_owner[owner] = [
                        item for item in replacements_by_owner[owner]
                        if item.instance_anchor_id not in closed_anchors
                    ]
                    if not replacements_by_owner[owner]:
                        del replacements_by_owner[owner]
                for representative in occurrences:
                    occurrence_replacements = (
                        _translated_occurrence_replacements(
                            representative,
                            objects=objects,
                            object_by_id=object_by_id,
                            case_id=str(context["sample"]["case_id"]),
                            global_dx=global_dx,
                            global_dy=global_dy,
                            selected_layers=_closed_occurrence_layer_scope(
                                scene.repair_family,
                                direct.layer,
                            ),
                        )
                    )
                    source = object_by_id.get(
                        str(representative.source_object_id)
                    )
                    if occurrence_replacements is None or source is None:
                        failed = True
                        break
                    source_is_anchor = bool(
                        source.source_anchor_id
                        and source.insertion_source_span is not None
                    )
                    owner = (
                        source.object_id if source_is_anchor else
                        authority.insertion_owner_by_layer.get(
                            representative.layer
                        )
                    )
                    if owner is None:
                        owner = specialization_anchor
                    if owner is None:
                        failed = True
                        break
                    replacements.extend(occurrence_replacements)
                    replacements_by_owner[owner].extend(
                        occurrence_replacements
                    )
                if failed:
                    break
        specialized_instances = {
            item.instance_anchor_id for item in replacements
        }
        specializations = _group_specialization_operations(
            replacements_by_owner,
        )
        if (
            failed or not replacements or len(specialized_instances) > 4
            or specializations is None
        ):
            if not failed:
                reject(solution, "LOWERING_EMPTY_REPLACEMENTS" if not replacements else
                       "LOWERING_OCCURRENCE_BOUND_EXCEEDED" if len(specialized_instances) > 4 else
                       "LOWERING_REPLACEMENT_GROUP_REJECTED")
            elif rejections is not None and not any(r["solution_id"] == solution.solution_id for r in rejections):
                reject(solution, "LOWERING_CLOSED_CARRIER_UNRESOLVED")
            continue
        operations = direct_operations + specializations
        if len(operations) > 4:
            reject(solution, "LOWERING_OPERATION_BOUND_EXCEEDED")
            continue
        programs.append(RepairProgram(
            program_id="phase3r_program_" + stable_hash([
                scene.scene_id,
                [item.model_dump(mode="json") for item in operations],
            ])[:20],
            region_id=scene.focus.region_id,
            target_violation_ids=[scene.focus.primary_violation_id],
            witness_ids=[scene.focus.target_witness_id],
            operations=operations,
            strategy_code=(
                f"PHASE3_{scene.repair_family.value}_INSTANCE_SPECIALIZATION"
            ),
        ))
    # ``solutions`` is already ordered by the deterministic geometry objective.
    # Preserve that order while deduplicating: sorting by the content hash here
    # used to turn the bounded root-proposal budget into an accidental random
    # choice between equally valid carrier classes.
    unique = {}
    for item in programs:
        unique.setdefault(stable_hash(item.model_dump(mode="json")), item)
    return list(unique.values())

def build_authority_programs(
    scene: RepairScene, authority: AuthorityResolution, context: dict,
) -> list[RepairProgram]:
    if scene.repair_family == RepairFamily.ENCLOSURE:
        return (
            _enclosure_programs(scene, authority)
            + build_enclosure_specialization_programs(
                scene, authority, context,
            )
        )
    if scene.repair_family == RepairFamily.VIA_METAL_CONTEXTUAL:
        return _via_programs(scene, context)
    return []

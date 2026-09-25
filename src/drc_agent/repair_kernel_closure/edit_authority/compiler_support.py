from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable

from drc_agent.schemas.action import DesignState, InstanceGeometrySpecialization
from drc_agent.schemas.common import Point, canonical_polygon, stable_hash
from drc_agent.schemas.repair_program import InstanceLayerReplacement
from drc_agent.schemas.state import LayoutObject, RegionState


def specialized_cell_name(source_cell: str, identity: object) -> str:
    """Preserve the source cell taxonomy in deterministic specializations."""
    physical_name = (
        source_cell.removeprefix("cell_")
        if source_cell.startswith("cell_") else source_cell
    )
    return f"{physical_name}_drc_agent_spec_{stable_hash(identity)[:16]}"


def _transform_point(
    point: Point, rotation: int, mirror: bool, dx: int, dy: int,
) -> Point:
    x, y = point.x, point.y
    if mirror:
        y = -y
    rotation %= 4
    if rotation == 1:
        x, y = -y, x
    elif rotation == 2:
        x, y = -x, -y
    elif rotation == 3:
        x, y = y, -x
    return Point(x=x + dx, y=y + dy)


def compile_instance_layer_specializations(
    replacements: list[InstanceLayerReplacement],
    *,
    region: RegionState,
    design: DesignState,
    validate_polygon: Callable[..., list[Point]],
    error_factory: Callable[[str, str], Exception],
) -> tuple[
    list[InstanceGeometrySpecialization], list[list[Point]], list[list[Point]],
]:
    """Compile complete target-occurrence replacements, atomically by instance."""

    compiled: list[InstanceGeometrySpecialization] = []
    global_before_polygons: list[list[Point]] = []
    global_after_polygons: list[list[Point]] = []
    groups: dict[
        tuple[str, str, str, int, bool, int, int],
        list[InstanceLayerReplacement],
    ] = defaultdict(list)
    for replacement in replacements:
        groups[(
            replacement.instance_anchor_id,
            replacement.parent_cell,
            replacement.source_cell,
            replacement.rotation,
            replacement.mirror,
            replacement.dx_dbu,
            replacement.dy_dbu,
        )].append(replacement)

    for group_key, group in sorted(groups.items(), key=lambda item: item[0]):
        representative = min(group, key=lambda item: item.layer)
        fragment_modes = {
            item.occurrence_local_fragment for item in group
        }
        if len(fragment_modes) != 1:
            raise error_factory(
                "MIXED_INSTANCE_SPECIALIZATION_SCOPE",
                "one occurrence specialization cannot mix whole-carrier and "
                "local-fragment replacements",
            )
        occurrence_local_fragment = representative.occurrence_local_fragment
        if occurrence_local_fragment and len(group) != 1:
            raise error_factory(
                "UNBOUNDED_LOCAL_FRAGMENT_SPECIALIZATION",
                "an occurrence-local fragment specialization must replace "
                "exactly one complete source layer",
            )
        complete_by_layer: dict[str, list[list[Point]]] = {}
        for raw in design.objects.values():
            if (
                raw.get("source_cell") == representative.source_cell
                and raw.get("geometry_dbu")
                and raw.get("layer") in design.legal_layers
            ):
                complete_by_layer.setdefault(str(raw["layer"]), []).append(
                    canonical_polygon([
                        Point.model_validate(point)
                        for point in raw["geometry_dbu"]
                    ])
                )

        supplied_by_layer: dict[str, list[list[Point]]] = {}
        changed_layers: dict[str, list[list[Point]]] = {}
        for replacement in sorted(group, key=lambda item: item.layer):
            if replacement.layer in changed_layers:
                raise error_factory(
                    "DUPLICATE_INSTANCE_LAYER_REPLACEMENT",
                    "one occurrence specialization may replace each layer once",
                )
            known = [
                LayoutObject.model_validate(raw)
                for raw in design.objects.values()
                if raw.get("source_cell") == replacement.source_cell
                and raw.get("layer") == replacement.layer
                and raw.get("geometry_dbu")
            ]
            known_polygons = sorted(
                (canonical_polygon(list(item.geometry_dbu or [])) for item in known),
                key=stable_hash,
            )
            supplied = sorted(
                (
                    canonical_polygon(list(points))
                    for points in replacement.original_polygons_local_dbu
                ),
                key=stable_hash,
            )
            if [stable_hash(x) for x in known_polygons] != [
                stable_hash(x) for x in supplied
            ]:
                raise error_factory(
                    "INCOMPLETE_INSTANCE_LAYER_REPLACEMENT",
                    "specialization must replace the complete known cell layer",
                )

            local_after: list[list[Point]] = []
            unchanged_hashes = Counter(stable_hash(item) for item in supplied)
            for index, polygon in enumerate(
                replacement.replacement_polygons_local_dbu
            ):
                canonical = canonical_polygon(list(polygon))
                if len(canonical) < 4 or any(
                    point.x % design.manufacturing_grid_dbu != 0
                    or point.y % design.manufacturing_grid_dbu != 0
                    for point in canonical
                ):
                    raise error_factory(
                        "INVALID_SPECIALIZATION_GEOMETRY",
                        "specialized local polygons must be nondegenerate and on grid",
                    )
                global_polygon = [
                    _transform_point(
                        point,
                        replacement.rotation,
                        replacement.mirror,
                        replacement.dx_dbu,
                        replacement.dy_dbu,
                    )
                    for point in canonical
                ]
                polygon_hash = stable_hash(canonical)
                original_at_index = (
                    canonical_polygon(list(
                        replacement.original_polygons_local_dbu[index]
                    ))
                    if index < len(replacement.original_polygons_local_dbu)
                    else None
                )
                changed_at_index = bool(
                    original_at_index is not None
                    and stable_hash(original_at_index) != polygon_hash
                )
                if unchanged_hashes[polygon_hash]:
                    unchanged_hashes[polygon_hash] -= 1
                else:
                    global_after_polygons.append(global_polygon)
                if not occurrence_local_fragment:
                    validate_polygon(
                        global_polygon, region=region, design=design,
                    )
                elif changed_at_index:
                    if len(original_at_index) != len(canonical):
                        raise error_factory(
                            "INVALID_LOCAL_FRAGMENT_SPECIALIZATION",
                            "local-fragment COW must preserve polygon vertex "
                            "cardinality",
                        )
                    global_original = [
                        _transform_point(
                            point,
                            replacement.rotation,
                            replacement.mirror,
                            replacement.dx_dbu,
                            replacement.dy_dbu,
                        )
                        for point in original_at_index
                    ]
                    changed_points = [
                        point
                        for before, after in zip(
                            global_original, global_polygon,
                        )
                        if before != after
                        for point in (before, after)
                    ]
                    halo = region.edit_halo_dbu
                    if not changed_points or any(
                        not (
                            halo.x1 <= point.x <= halo.x2
                            and halo.y1 <= point.y <= halo.y2
                        )
                        for point in changed_points
                    ):
                        raise error_factory(
                            "OUTSIDE_LOCALITY",
                            "local-fragment boundary change exceeds region "
                            "edit halo",
                        )
                    if design.die_boundary is not None and any(
                        not (
                            design.die_boundary.x1 <= point.x
                            <= design.die_boundary.x2
                            and design.die_boundary.y1 <= point.y
                            <= design.die_boundary.y2
                        )
                        for point in global_polygon
                    ):
                        raise error_factory(
                            "OUTSIDE_DIE",
                            "specialized local polygon exceeds die boundary",
                        )
                if changed_at_index:
                    global_before_polygons.append([
                        _transform_point(
                            point,
                            replacement.rotation,
                            replacement.mirror,
                            replacement.dx_dbu,
                            replacement.dy_dbu,
                        )
                        for point in original_at_index
                    ])
                    if global_polygon not in global_after_polygons:
                        global_after_polygons.append(global_polygon)
                local_after.append(canonical)
            supplied_by_layer[replacement.layer] = supplied
            changed_layers[replacement.layer] = local_after
            complete_by_layer[replacement.layer] = local_after

        if occurrence_local_fragment:
            primary_layer = representative.layer
            before_counter = Counter(
                stable_hash(item) for item in supplied_by_layer[primary_layer]
            )
            after_counter = Counter(
                stable_hash(item) for item in changed_layers[primary_layer]
            )
            if (
                sum((before_counter - after_counter).values()) != 1
                or sum((after_counter - before_counter).values()) != 1
                or len(supplied_by_layer[primary_layer])
                != len(changed_layers[primary_layer])
            ):
                raise error_factory(
                    "INVALID_LOCAL_FRAGMENT_SPECIALIZATION",
                    "occurrence-local COW must replace exactly one source "
                    "polygon while preserving layer cardinality",
                )

        complete_by_layer = {
            layer: sorted(polygons, key=stable_hash)
            for layer, polygons in sorted(complete_by_layer.items())
        }
        primary_layer = representative.layer
        specialized_name = specialized_cell_name(
            representative.source_cell,
            (
                [group_key, complete_by_layer, True]
                if occurrence_local_fragment
                else [group_key, complete_by_layer]
            ),
        )
        compiled.append(InstanceGeometrySpecialization(
            instance_anchor_id=representative.instance_anchor_id,
            parent_cell=representative.parent_cell,
            source_cell=representative.source_cell,
            specialized_cell_name=specialized_name,
            layer=primary_layer,
            rotation=representative.rotation,
            mirror=representative.mirror,
            dx_dbu=representative.dx_dbu,
            dy_dbu=representative.dy_dbu,
            original_polygons_local_dbu=supplied_by_layer[primary_layer],
            original_polygons_by_layer_local_dbu=supplied_by_layer,
            replacement_polygons_local_dbu=changed_layers[primary_layer],
            complete_polygons_by_layer_local_dbu=complete_by_layer,
            replaced_layers=sorted(changed_layers),
            occurrence_local_fragment=occurrence_local_fragment,
        ))
    return compiled, global_before_polygons, global_after_polygons

from __future__ import annotations

from collections import defaultdict
from math import isqrt
from typing import Literal

from pydantic import Field

from drc_agent.schemas.common import Box, Point, StrictModel, stable_hash
from drc_agent.schemas.state import LayoutObject, ViolationRecord


class RouteEndpoint(StrictModel):
    endpoint_id: str
    fragment_id: str
    parent_object_id: str
    point_dbu: Point
    end: Literal["START", "END"]
    boundary_kind: Literal["PARENT_ENDPOINT", "WINDOW_BOUNDARY"]
    connectivity_component_id: str | None = None
    evidence_level: Literal["SOURCE_EXACT", "GEOMETRIC"] = "GEOMETRIC"


class RouteFragment(StrictModel):
    fragment_id: str
    parent_object_id: str
    parent_source_anchor_id: str
    violation_ids: list[str]
    layer: str
    bbox_dbu: Box
    geometry_dbu: list[Point]
    orientation: Literal["HORIZONTAL", "VERTICAL", "SQUARE"]
    endpoint_ids: list[str]
    connectivity_component_id: str | None = None
    evidence_level: Literal["SOURCE_EXACT", "GEOMETRIC"]


class ViaLanding(StrictModel):
    landing_id: str
    violation_id: str
    layer: str
    bbox_dbu: Box
    parent_object_id: str | None = None
    parent_source_anchor_id: str | None = None
    connectivity_component_id: str | None = None
    evidence_level: Literal["SOURCE_EXACT", "MARKER_GEOMETRIC"]


class ViaMetalAdjacency(StrictModel):
    adjacency_id: str
    via_landing_id: str
    route_fragment_id: str
    relation: Literal["OVERLAPS", "NEARBY"]
    overlap_area_dbu2: int
    gap_dbu: int
    connectivity_component_id: str | None = None
    evidence_level: Literal["SOURCE_EXACT", "GEOMETRIC"]


class SameLayerAdjacency(StrictModel):
    adjacency_id: str
    a_fragment_id: str
    b_fragment_id: str
    layer: str
    gap_dbu: int
    evidence_level: Literal["SOURCE_EXACT", "GEOMETRIC"]


class LocalRoutingTopology(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    fragments: dict[str, RouteFragment] = Field(default_factory=dict)
    endpoints: dict[str, RouteEndpoint] = Field(default_factory=dict)
    via_landings: dict[str, ViaLanding] = Field(default_factory=dict)
    via_metal_adjacencies: list[ViaMetalAdjacency] = Field(default_factory=list)
    same_layer_adjacencies: list[SameLayerAdjacency] = Field(default_factory=list)
    parent_to_fragment_ids: dict[str, list[str]] = Field(default_factory=dict)
    source_anchor_to_fragment_ids: dict[str, list[str]] = Field(default_factory=dict)
    coverage: dict[str, int | float] = Field(default_factory=dict)
    evidence_level: Literal["SOURCE_PLUS_CONNECTIVITY", "SOURCE_GEOMETRIC"]


def nm_to_dbu(
    nm: int, dbu_per_um: int, *,
    rounding: Literal["nearest", "outward", "inward"] = "nearest",
) -> int:
    """Convert physical nm using runtime metadata; LLM values are never used."""
    if nm < 0 or dbu_per_um <= 0:
        raise ValueError("nm and dbu_per_um must be non-negative/positive")
    numerator = nm * dbu_per_um
    if rounding == "outward":
        return (numerator + 999) // 1000
    if rounding == "inward":
        return numerator // 1000
    return (numerator + 500) // 1000


def _polygon(box: Box) -> list[Point]:
    return [
        Point(x=box.x1, y=box.y1),
        Point(x=box.x1, y=box.y2),
        Point(x=box.x2, y=box.y2),
        Point(x=box.x2, y=box.y1),
    ]


def outward_box_gap_dbu(a: Box, b: Box) -> int:
    """Return conservative integer Euclidean separation in runtime DBU."""
    dx = max(a.x1 - b.x2, b.x1 - a.x2, 0)
    dy = max(a.y1 - b.y2, b.y1 - a.y2, 0)
    squared = dx * dx + dy * dy
    root = isqrt(squared)
    return root if root * root == squared else root + 1


def _intersection(a: Box, b: Box) -> Box | None:
    x1, y1 = max(a.x1, b.x1), max(a.y1, b.y1)
    x2, y2 = min(a.x2, b.x2), min(a.y2, b.y2)
    if x1 >= x2 or y1 >= y2:
        return None
    return Box(x1=x1, y1=y1, x2=x2, y2=y2)


def _orientation(box: Box) -> Literal["HORIZONTAL", "VERTICAL", "SQUARE"]:
    if box.width > box.height:
        return "HORIZONTAL"
    if box.height > box.width:
        return "VERTICAL"
    return "SQUARE"


def _via_adjacent_metals(layer: str) -> set[str]:
    if not layer.startswith("V") or not layer[1:].isdigit():
        return set()
    lower = int(layer[1:])
    return {f"M{lower}", f"M{lower + 1}"}


class LocalTopologyBuilder:
    """Build bounded, traceable topology without inventing logical net names."""

    def __init__(self, *, dbu_per_um: int, minimum_window_dbu: int = 256):
        self.dbu_per_um = dbu_per_um
        self.minimum_window_dbu = minimum_window_dbu

    def build(
        self, violations: list[ViolationRecord], objects: list[LayoutObject],
    ) -> LocalRoutingTopology:
        fragments: dict[str, RouteFragment] = {}
        endpoints: dict[str, RouteEndpoint] = {}
        landings: dict[str, ViaLanding] = {}
        by_parent: defaultdict[str, list[str]] = defaultdict(list)
        by_anchor: defaultdict[str, list[str]] = defaultdict(list)
        source_objects = [
            obj for obj in objects
            if obj.source_anchor_id and obj.geometry_dbu
            and obj.routing_type == "routing"
        ]
        long_eligible = {
            obj.object_id for obj in source_objects
            if max(obj.bbox_dbu.width, obj.bbox_dbu.height)
            >= 2 * self.dbu_per_um
        }

        for violation in sorted(
            violations, key=lambda item: item.violation_id
        ):
            marker = violation.marker_bbox_dbu
            influence = max(
                self.minimum_window_dbu, marker.width, marker.height,
            )
            window = marker.expand(influence)
            for obj in source_objects:
                if obj.kind not in {"polygon", "path"}:
                    continue
                if violation.layers and obj.layer not in violation.layers:
                    continue
                clipped = _intersection(obj.bbox_dbu, window)
                if clipped is None:
                    continue
                fragment_id = "fragment_" + stable_hash([
                    obj.object_id, violation.violation_id, clipped,
                ])[:20]
                orientation = _orientation(obj.bbox_dbu)
                endpoint_ids = []
                if orientation != "SQUARE":
                    if orientation == "HORIZONTAL":
                        points = [
                            Point(x=clipped.x1, y=(clipped.y1 + clipped.y2) // 2),
                            Point(x=clipped.x2, y=(clipped.y1 + clipped.y2) // 2),
                        ]
                        parent_bounds = [obj.bbox_dbu.x1, obj.bbox_dbu.x2]
                        local_bounds = [clipped.x1, clipped.x2]
                    else:
                        points = [
                            Point(x=(clipped.x1 + clipped.x2) // 2, y=clipped.y1),
                            Point(x=(clipped.x1 + clipped.x2) // 2, y=clipped.y2),
                        ]
                        parent_bounds = [obj.bbox_dbu.y1, obj.bbox_dbu.y2]
                        local_bounds = [clipped.y1, clipped.y2]
                    for index, (point, bound, parent_bound) in enumerate(zip(
                        points, local_bounds, parent_bounds, strict=True,
                    )):
                        endpoint_id = "endpoint_" + stable_hash([
                            fragment_id, index, point,
                        ])[:20]
                        endpoints[endpoint_id] = RouteEndpoint(
                            endpoint_id=endpoint_id,
                            fragment_id=fragment_id,
                            parent_object_id=obj.object_id,
                            point_dbu=point,
                            end=("START" if index == 0 else "END"),
                            boundary_kind=(
                                "PARENT_ENDPOINT"
                                if bound == parent_bound else "WINDOW_BOUNDARY"
                            ),
                            connectivity_component_id=(
                                obj.connectivity_component_id
                            ),
                            evidence_level=(
                                "SOURCE_EXACT"
                                if obj.source_span else "GEOMETRIC"
                            ),
                        )
                        endpoint_ids.append(endpoint_id)
                fragment = RouteFragment(
                    fragment_id=fragment_id,
                    parent_object_id=obj.object_id,
                    parent_source_anchor_id=obj.source_anchor_id,
                    violation_ids=[violation.violation_id],
                    layer=obj.layer,
                    bbox_dbu=clipped,
                    geometry_dbu=_polygon(clipped),
                    orientation=orientation,
                    endpoint_ids=endpoint_ids,
                    connectivity_component_id=obj.connectivity_component_id,
                    evidence_level=(
                        "SOURCE_EXACT" if obj.source_span else "GEOMETRIC"
                    ),
                )
                fragments[fragment_id] = fragment
                by_parent[obj.object_id].append(fragment_id)
                by_anchor[obj.source_anchor_id].append(fragment_id)

            for layer in violation.layers:
                if not layer.startswith("V"):
                    continue
                matching = [
                    obj for obj in source_objects
                    if obj.layer == layer and obj.bbox_dbu.intersects(marker)
                ]
                if matching:
                    for obj in matching:
                        landing_id = "landing_" + stable_hash([
                            violation.violation_id, obj.object_id,
                        ])[:20]
                        landings[landing_id] = ViaLanding(
                            landing_id=landing_id,
                            violation_id=violation.violation_id,
                            layer=layer,
                            bbox_dbu=obj.bbox_dbu,
                            parent_object_id=obj.object_id,
                            parent_source_anchor_id=obj.source_anchor_id,
                            connectivity_component_id=(
                                obj.connectivity_component_id
                            ),
                            evidence_level="SOURCE_EXACT",
                        )
                else:
                    landing_id = "landing_" + stable_hash([
                        violation.violation_id, layer, marker,
                    ])[:20]
                    landings[landing_id] = ViaLanding(
                        landing_id=landing_id,
                        violation_id=violation.violation_id,
                        layer=layer,
                        bbox_dbu=marker,
                        evidence_level="MARKER_GEOMETRIC",
                    )

        same_layer = []
        ordered = sorted(fragments.values(), key=lambda item: item.fragment_id)
        for index, a in enumerate(ordered):
            for b in ordered[index + 1:]:
                if a.layer != b.layer or a.parent_object_id == b.parent_object_id:
                    continue
                gap = outward_box_gap_dbu(a.bbox_dbu, b.bbox_dbu)
                if gap > self.minimum_window_dbu:
                    continue
                same_layer.append(SameLayerAdjacency(
                    adjacency_id="same_layer_" + stable_hash([
                        a.fragment_id, b.fragment_id,
                    ])[:20],
                    a_fragment_id=a.fragment_id,
                    b_fragment_id=b.fragment_id,
                    layer=a.layer,
                    gap_dbu=gap,
                    evidence_level=(
                        "SOURCE_EXACT"
                        if a.evidence_level == b.evidence_level == "SOURCE_EXACT"
                        else "GEOMETRIC"
                    ),
                ))

        via_metal = []
        for landing in landings.values():
            legal_metals = _via_adjacent_metals(landing.layer)
            for fragment in fragments.values():
                if fragment.layer not in legal_metals:
                    continue
                if landing.violation_id not in fragment.violation_ids:
                    continue
                overlap = landing.bbox_dbu.intersection_area(
                    fragment.bbox_dbu
                )
                gap = outward_box_gap_dbu(
                    landing.bbox_dbu, fragment.bbox_dbu,
                )
                if not overlap and gap > self.minimum_window_dbu:
                    continue
                component = (
                    landing.connectivity_component_id
                    if landing.connectivity_component_id
                    == fragment.connectivity_component_id else None
                )
                via_metal.append(ViaMetalAdjacency(
                    adjacency_id="via_metal_" + stable_hash([
                        landing.landing_id, fragment.fragment_id,
                    ])[:20],
                    via_landing_id=landing.landing_id,
                    route_fragment_id=fragment.fragment_id,
                    relation=("OVERLAPS" if overlap else "NEARBY"),
                    overlap_area_dbu2=overlap,
                    gap_dbu=gap,
                    connectivity_component_id=component,
                    evidence_level=(
                        "SOURCE_EXACT"
                        if (
                            landing.evidence_level == "SOURCE_EXACT"
                            and fragment.evidence_level == "SOURCE_EXACT"
                        ) else "GEOMETRIC"
                    ),
                ))

        component_objects = {
            obj.object_id for obj in source_objects
            if obj.connectivity_component_id
        }
        fragmented_long = long_eligible & set(by_parent)
        coverage = {
            "source_routing_object_count": len(source_objects),
            "fragmented_parent_count": len(by_parent),
            "long_route_parent_count": len(long_eligible),
            "fragmented_long_route_parent_count": len(fragmented_long),
            "connectivity_component_object_count": len(component_objects),
            "via_landing_count": len(landings),
            "via_metal_adjacency_count": len(via_metal),
            "same_layer_adjacency_count": len(same_layer),
            "parent_mapping_coverage_milli": (
                1000 * len(by_parent) // max(len(source_objects), 1)
            ),
            "long_route_mapping_coverage_milli": (
                1000 * len(fragmented_long) // max(len(long_eligible), 1)
            ),
        }
        return LocalRoutingTopology(
            fragments=dict(sorted(fragments.items())),
            endpoints=dict(sorted(endpoints.items())),
            via_landings=dict(sorted(landings.items())),
            via_metal_adjacencies=sorted(
                via_metal, key=lambda item: item.adjacency_id,
            ),
            same_layer_adjacencies=sorted(
                same_layer, key=lambda item: item.adjacency_id,
            ),
            parent_to_fragment_ids={
                key: sorted(set(value))
                for key, value in sorted(by_parent.items())
            },
            source_anchor_to_fragment_ids={
                key: sorted(set(value))
                for key, value in sorted(by_anchor.items())
            },
            coverage=coverage,
            evidence_level=(
                "SOURCE_PLUS_CONNECTIVITY"
                if component_objects else "SOURCE_GEOMETRIC"
            ),
        )

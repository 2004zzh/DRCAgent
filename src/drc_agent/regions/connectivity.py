from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from pydantic import Field

from drc_agent.schemas.common import StrictModel, stable_hash
from drc_agent.schemas.state import LayoutObject


class ConnectivityMappingReport(StrictModel):
    evidence_level: str = "CONNECTIVITY_COMPONENT_EXACT_GEOMETRY"
    named_net_available: bool = False
    source_format: str = "DAC26_GOLDEN_PATH_ENDPOINTS"
    object_count: int
    endpoint_count: int
    uniquely_matched_endpoint_count: int
    ambiguous_endpoint_count: int
    mapped_object_count: int
    unavailable_object_count: int
    component_count: int
    mapped_object_ids: list[str] = Field(default_factory=list)


class _UnionFind:
    def __init__(self, values: list[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if b < a:
            a, b = b, a
        self.parent[b] = a


def _canonical_polygon(points) -> tuple[tuple[int, int], ...]:
    values = tuple((int(point[0]), int(point[1])) for point in points)
    if not values:
        return ()
    rotations = []
    for sequence in (values, tuple(reversed(values))):
        rotations.extend(
            sequence[index:] + sequence[:index]
            for index in range(len(sequence))
        )
    return min(rotations)


def map_connectivity_components(
    objects: list[LayoutObject], connectivity_json: Path,
) -> tuple[list[LayoutObject], ConnectivityMappingReport]:
    """Map only exact, unique top-level DAC26 endpoint geometry.

    Golden connectivity contains path endpoints rather than named nets.  The
    transitive endpoint graph therefore supports stable component ownership,
    but never authoritative net names or router-resource capacity.
    """
    raw = json.loads(connectivity_json.read_text(encoding="utf-8"))
    index: dict[tuple[int, tuple], list[str]] = defaultdict(list)
    by_id = {item.object_id: item for item in objects}
    for item in objects:
        if (
            item.routing_type != "routing" or item.layer_number is None
            or not item.geometry_dbu
        ):
            continue
        key = (
            int(item.layer_number),
            _canonical_polygon([(point.x, point.y) for point in item.geometry_dbu]),
        )
        index[key].append(item.object_id)

    matched_paths: list[tuple[str | None, str | None]] = []
    matched_ids: set[str] = set()
    ambiguous = 0
    endpoint_count = 0
    unique_endpoint_count = 0
    for path in raw.get("paths", []):
        endpoints = []
        for endpoint in (path.get("start") or {}, path.get("end") or {}):
            endpoint_count += 1
            try:
                key = (
                    int(endpoint["layer"]),
                    _canonical_polygon(endpoint["points"]),
                )
            except (KeyError, TypeError, ValueError):
                endpoints.append(None)
                continue
            matches = index.get(key, [])
            if len(matches) == 1:
                endpoints.append(matches[0])
                matched_ids.add(matches[0])
                unique_endpoint_count += 1
            else:
                ambiguous += int(len(matches) > 1)
                endpoints.append(None)
        matched_paths.append((endpoints[0], endpoints[1]))

    union = _UnionFind(sorted(matched_ids))
    for left, right in matched_paths:
        if left is not None and right is not None:
            union.union(left, right)
    members: dict[str, list[str]] = defaultdict(list)
    for object_id in sorted(matched_ids):
        members[union.find(object_id)].append(object_id)
    component_by_object = {}
    for values in members.values():
        component_id = "cc_" + stable_hash(sorted(values))[:20]
        for object_id in values:
            component_by_object[object_id] = component_id

    mapped = []
    for item in objects:
        component_id = component_by_object.get(item.object_id)
        if component_id is None:
            mapped.append(item)
        else:
            mapped.append(item.model_copy(update={
                "net_id": component_id,
                "net_mapping_quality": "connectivity_component",
                "connectivity_component_id": component_id,
            }))
    report = ConnectivityMappingReport(
        object_count=len(objects), endpoint_count=endpoint_count,
        uniquely_matched_endpoint_count=unique_endpoint_count,
        ambiguous_endpoint_count=ambiguous,
        mapped_object_count=len(component_by_object),
        unavailable_object_count=len(objects) - len(component_by_object),
        component_count=len(set(component_by_object.values())),
        mapped_object_ids=sorted(component_by_object),
    )
    return mapped, report

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field

from drc_agent.schemas.common import Box, Point, StrictModel


class LocalFragmentStatus(StrEnum):
    EXTRACTED = "EXTRACTED"
    PARENT_UNAVAILABLE = "PARENT_UNAVAILABLE"
    FRAGMENT_UNAVAILABLE = "FRAGMENT_UNAVAILABLE"
    AMBIGUOUS_FRAGMENT = "AMBIGUOUS_FRAGMENT"
    NON_RECTANGULAR_PARENT = "NON_RECTANGULAR_PARENT"
    UNBOUNDED_FRAGMENT = "UNBOUNDED_FRAGMENT"
    TARGET_CONTRIBUTOR_MISMATCH = "TARGET_CONTRIBUTOR_MISMATCH"
    STALE_TARGET = "STALE_TARGET"


class LocalRouteFragment(StrictModel):
    """Source-owned bounded window over one longer route polygon."""

    schema_version: Literal["1.0"] = "1.0"
    fragment_id: str
    parent_source_object_id: str
    source_anchor_id: str
    layer: str
    orientation: Literal["HORIZONTAL", "VERTICAL"]
    local_bbox: Box
    entry_boundary: Box
    exit_boundary: Box
    local_geometry_before: list[Point]
    preserved_prefix_geometry: list[Point]
    preserved_suffix_geometry: list[Point]
    connection_points: list[Point] = Field(min_length=4, max_length=4)
    edit_halo: Box
    connectivity_component_id: str | None = None
    connectivity_quality: str
    authority_ref: str
    parent_geometry_hash: str
    parent_source_hash: str
    fragment_fingerprint: str


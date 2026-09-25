from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field

from drc_agent.schemas.common import StrictModel


class EditAuthorityClass(StrEnum):
    DIRECT_LOCAL_SOURCE_EDIT = "DIRECT_LOCAL_SOURCE_EDIT"
    INSTANCE_LOCAL_EDIT = "INSTANCE_LOCAL_EDIT"
    LOCAL_ADDITIVE_OVERRIDE = "LOCAL_ADDITIVE_OVERRIDE"
    SHARED_SOURCE_REQUIRES_SPECIALIZATION = (
        "SHARED_SOURCE_REQUIRES_SPECIALIZATION"
    )
    SHARED_SOURCE_GLOBAL_UNSAFE = "SHARED_SOURCE_GLOBAL_UNSAFE"
    FROZEN_TECH_OR_PIN = "FROZEN_TECH_OR_PIN"
    UNMAPPED = "UNMAPPED"


class AuthorityImpactReport(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    report_id: str
    participant_id: str
    source_anchor_id: str
    definition_id: str | None = None
    instance_ids: list[str] = Field(default_factory=list)
    occurrence_count: int = 0
    target_occurrence_ids: list[str] = Field(default_factory=list)
    off_focus_occurrence_ids: list[str] = Field(default_factory=list)
    global_edit_would_touch_count: int = 0
    authority_class: EditAuthorityClass
    reason_codes: list[str] = Field(default_factory=list)
    insertion_owner_object_id: str | None = None
    local_override_capability: bool = False
    specialization_expressible: bool = False


class AuthorityResolution(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    resolution_id: str
    scene_id: str
    reports: list[AuthorityImpactReport]
    insertion_owner_by_layer: dict[str, str] = Field(default_factory=dict)
    available_modes: list[EditAuthorityClass] = Field(default_factory=list)
    failure_codes: list[str] = Field(default_factory=list)


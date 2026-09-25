from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field

from drc_agent.schemas.common import StrictModel


class FunnelStage(StrEnum):
    ORACLE = "ORACLE"
    PLAN = "PLAN"
    SOLVER = "SOLVER"
    PREDICATE_PROBE = "PREDICATE_PROBE"
    SANDBOX = "SANDBOX"


class FunnelStatus(StrEnum):
    STARTED = "STARTED"
    PASS = "PASS"
    FAIL = "FAIL"
    SKIPPED = "SKIPPED"


class FunnelRecord(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    root_sample_id: str
    trajectory_id: str
    expansion_id: str
    parent_node_id: str
    depth: int = Field(ge=0)
    stage: FunnelStage
    status: FunnelStatus
    failure_code: str | None = None
    input_fingerprint: str
    output_fingerprint: str | None = None
    active_obligation_id: str | None = None
    candidate_ids: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    wall_time_seconds: float = Field(ge=0)
    artifact_refs: list[str] = Field(default_factory=list)


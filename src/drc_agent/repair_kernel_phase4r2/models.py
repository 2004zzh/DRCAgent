from __future__ import annotations

from typing import Literal

from pydantic import Field

from drc_agent.schemas.common import StrictModel


class ArtifactEvidence(StrictModel):
    path: str
    sha256: str | None = None
    exists: bool
    size_bytes: int | None = None


class MarkerDelta(StrictModel):
    rule_id: str
    marker: dict
    multiplicity: int = Field(ge=1)


class M4FailureAttribution(StrictModel):
    classification: Literal[
        "UNSUPPORTED_CURRENT_RULE",
        "CURRENT_RULE_PREDICATE_UNAVAILABLE",
        "CURRENT_RULE_WITNESS_UNAVAILABLE",
        "AMBIGUOUS_PHYSICAL_RELATION",
        "NO_PHYSICAL_CONTRIBUTOR",
        "STALE_SOURCE_LINEAGE",
        "DUPLICATE_GROUP_IDENTITY",
        "OTHER_EXACT",
    ]
    root_step_added: list[MarkerDelta]
    root_step_removed: list[MarkerDelta]
    depth2_added_by_snapshot: dict[str, list[MarkerDelta]]
    current_debt_rule_ids: list[str]
    unsupported_current_rule_ids: list[str]
    m4_s4_observed: bool
    reason_codes: list[str]
    artifacts: list[ArtifactEvidence]


class ToolStageEvidence(StrictModel):
    action: str
    status: str
    return_code: int | None = None
    error_code: str | None = None
    error_message: str | None = None
    artifacts: list[ArtifactEvidence] = Field(default_factory=list)


class EnclosureFailureAttribution(StrictModel):
    classification: Literal[
        "PATCH_COMPILE_FAILURE",
        "PATCH_APPLICATION_FAILURE",
        "SCRIPT_VALIDATION_FAILURE",
        "LAYOUT_GENERATION_FAILURE",
        "DRC_EXECUTION_FAILURE",
        "LYRPT_INVALID",
        "REPORT_CONVERSION_FAILURE",
        "GDS_SANITY_FAILURE",
        "CONNECTIVITY_EXECUTION_FAILURE",
        "CONNECTIVITY_FAIL",
        "ARTIFACT_REFERENCE_INCOMPLETE",
        "FROZEN_EVALUATOR_ATTESTATION_MISMATCH",
        "OTHER_EXACT",
    ]
    first_failed_action: str
    first_failed_return_code: int | None = None
    stderr_excerpt: str | None = None
    tool_stages: list[ToolStageEvidence]
    attempted_script: ArtifactEvidence
    attempted_gds: ArtifactEvidence
    attempted_drc: ArtifactEvidence
    attempted_lyrpt: ArtifactEvidence
    connectivity_result: ArtifactEvidence
    root_cause: str
    reason_codes: list[str]
    artifacts: list[ArtifactEvidence]


class FailureAttributionReport(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    phase: Literal["4-R2-A"] = "4-R2-A"
    m4: M4FailureAttribution
    enclosure: EnclosureFailureAttribution
    status: Literal["PASS", "FAIL"]

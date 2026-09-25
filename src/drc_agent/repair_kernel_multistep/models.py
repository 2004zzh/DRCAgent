from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from drc_agent.development.repair_kernel_bench.models import KernelAttemptResult
from drc_agent.schemas.common import ArtifactRef, Box, StrictModel


class TransitionClassification(StrEnum):
    FINAL_CLEAN = "FINAL_CLEAN"
    ADMITTED_TEMPORARY_DEBT = "ADMITTED_TEMPORARY_DEBT"
    ADMITTED_PROGRESS = "ADMITTED_PROGRESS"
    PRUNED_NO_PROGRESS = "PRUNED_NO_PROGRESS"
    PRUNED_CONNECTIVITY = "PRUNED_CONNECTIVITY"
    PRUNED_UNBOUNDED_DEBT = "PRUNED_UNBOUNDED_DEBT"
    PRUNED_UNSUPPORTED_DEBT = "PRUNED_UNSUPPORTED_DEBT"
    PRUNED_PROTECTED_RELATION_REGRESSION = (
        "PRUNED_PROTECTED_RELATION_REGRESSION"
    )
    EXECUTION_FAIL = "EXECUTION_FAIL"


class RootTask(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    root_sample_id: str
    case_id: str
    target_violation_id: str
    target_violation_fingerprint: str
    target_rule_id: str
    target_layers: list[str]
    target_bbox: Box
    trajectory_halo: Box
    rule_deck_path: str
    connectivity_reference_path: str
    root_script_path: str
    root_gds_path: str
    root_drc_path: str


class SearchSnapshot(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    snapshot_id: str
    root_sample_id: str
    case_id: str
    depth: int = Field(ge=0)
    script_ref: str
    gds_ref: str
    drc_ref: str
    connectivity_reference_ref: str
    script_sha256: str
    lineage_receipt_ref: ArtifactRef | None = None
    parent_script_ref: str | None = None
    parent_script_sha256: str | None = None
    provenance_run_id: str | None = None
    provenance_relation_source: str | None = None
    gds_sha256: str
    drc_sha256: str
    connectivity_reference_sha256: str
    root_relative_removed_original: int = Field(ge=0)
    root_relative_new: int = Field(ge=0)
    root_target_present: bool
    parent_relative_removed: int = Field(ge=0)
    parent_relative_new: int = Field(ge=0)
    temporary_debt_fingerprints: list[str] = Field(default_factory=list)
    protected_relation_ids: list[str] = Field(default_factory=list)
    touched_source_anchor_ids: list[str] = Field(default_factory=list)
    modified_instance_anchor_ids: list[str] = Field(default_factory=list)
    local_geometry_fingerprint: str
    state_fingerprint: str
    parent_node_id: str | None = None


class TransitionResult(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    transition_id: str
    parent_snapshot_id: str
    child_snapshot: SearchSnapshot | None = None
    phase2_classification: str
    phase4_classification: TransitionClassification
    phase2_attempt: KernelAttemptResult
    starting_script_sha256: str
    starting_gds_sha256: str
    starting_drc_sha256: str
    attempted_script_sha256: str | None = None
    attempted_gds_sha256: str | None = None
    attempted_drc_sha256: str | None = None
    reason_codes: list[str] = Field(default_factory=list)
    admitted: bool = False

    @model_validator(mode="after")
    def admitted_has_child(self) -> "TransitionResult":
        if self.admitted != (self.child_snapshot is not None):
            raise ValueError("admitted transition and child snapshot disagree")
        return self

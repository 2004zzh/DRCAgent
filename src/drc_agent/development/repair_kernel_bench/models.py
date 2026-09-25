from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, model_validator

from drc_agent.schemas.common import ArtifactRef, Box, StrictModel
from drc_agent.schemas.tools import AttemptTerminalStatus, EvidenceValidity


MANDATORY_RULES = (
    "M2.S.7",
    "V2.M3.AUX.2",
    "V1.M1.EN.1",
    "M4.AUX.1",
    "M4.AUX.2",
    "M1.S.2",
    "M4.S.5",
)


class EvidenceMode(StrEnum):
    HISTORICAL_REPLAY = "HISTORICAL_REPLAY"
    LIVE_REEXECUTED_CONTROL = "LIVE_REEXECUTED_CONTROL"
    LIVE_CURRENT_KERNEL = "LIVE_CURRENT_KERNEL"


class AttemptClassification(StrEnum):
    CLEAN_PROGRESS = "CLEAN_PROGRESS"
    NO_PROGRESS = "NO_PROGRESS"
    REGRESSION = "REGRESSION"
    CONNECTIVITY_FAIL = "CONNECTIVITY_FAIL"
    EXECUTION_FAIL = "EXECUTION_FAIL"


class FailureStage(StrEnum):
    NOT_RUN = "NOT_RUN"
    SAMPLE_DISCOVERY_FAIL = "SAMPLE_DISCOVERY_FAIL"
    RULE_NOT_PRESENT = "RULE_NOT_PRESENT"
    RULE_UNSUPPORTED = "RULE_UNSUPPORTED"
    PREDICATE_UNAVAILABLE = "PREDICATE_UNAVAILABLE"
    WITNESS_UNAVAILABLE = "WITNESS_UNAVAILABLE"
    WITNESS_LOW_FIDELITY = "WITNESS_LOW_FIDELITY"
    HIERARCHY_GROUNDING_FAIL = "HIERARCHY_GROUNDING_FAIL"
    NO_PHYSICAL_CONTRIBUTOR = "NO_PHYSICAL_CONTRIBUTOR"
    NO_EDITABLE_CONTRIBUTOR = "NO_EDITABLE_CONTRIBUTOR"
    CONNECTIVITY_UNAVAILABLE = "CONNECTIVITY_UNAVAILABLE"
    REGION_BUILD_FAIL = "REGION_BUILD_FAIL"
    CONTEXT_BUILD_FAIL = "CONTEXT_BUILD_FAIL"
    NO_SEMANTIC_INTENT = "NO_SEMANTIC_INTENT"
    LOWERING_EMPTY = "LOWERING_EMPTY"
    CANDIDATE_CHECK_FAIL = "CANDIDATE_CHECK_FAIL"
    REPAIR_PROGRAM_NOT_INVOKED = "REPAIR_PROGRAM_NOT_INVOKED"
    REPAIR_PROGRAM_NO_OUTPUT = "REPAIR_PROGRAM_NO_OUTPUT"
    PROGRAM_COMPILE_FAIL = "PROGRAM_COMPILE_FAIL"
    PREDICATE_PROBE_REJECT = "PREDICATE_PROBE_REJECT"
    SANDBOX_NO_PROGRESS = "SANDBOX_NO_PROGRESS"
    SANDBOX_REGRESSION = "SANDBOX_REGRESSION"
    CONNECTIVITY_FAIL = "CONNECTIVITY_FAIL"
    CLEAN_PROGRESS = "CLEAN_PROGRESS"


class InputLock(StrictModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class KernelSample(StrictModel):
    sample_id: str
    development_only: Literal[True] = True
    case_id: str
    rule_id: str
    rule_family: str
    baseline_script: InputLock
    baseline_gds: InputLock
    baseline_drc: InputLock
    connectivity: InputLock
    rule_deck: InputLock
    evaluator_hash: str
    original_violation_id: str
    target_violation_fingerprint: str
    target_bbox: Box
    target_layers: list[str]
    predicate_id: str
    predicate_fidelity: str
    predicate_signature: str
    witness_id: str
    witness_evidence_quality: str
    participating_physical_geometry_ids: list[str]
    editable_contributor_ids: list[str]
    co_located_violation_ids: list[str]
    source_anchor_ids: list[str]
    hierarchy_multiplicity: int
    connectivity_quality: str
    region_id: str
    selection_reason: str
    selection_version: str


class RuleInventory(StrictModel):
    rule_id: str
    rule_family: str
    occurrences_by_case: dict[str, int]
    sample_ids: list[str] = Field(default_factory=list)
    absence_reason: str | None = None


class SampleManifest(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    development_only: Literal[True] = True
    selection_version: str
    samples_per_rule: int
    case_preference: list[str]
    rules: list[RuleInventory]
    samples: list[KernelSample]


class ControlProvenance(StrictModel):
    control_id: str
    development_only: Literal[True] = True
    control_kind: Literal["POSITIVE", "NEGATIVE_NO_PROGRESS"]
    source_run_id: str
    source_iteration: int
    candidate_id: str
    candidate_artifact: InputLock
    historical_patch_artifact: InputLock
    historical_verification_artifact: InputLock
    starting_script: InputLock
    starting_gds: InputLock
    starting_drc: InputLock
    starting_connectivity: InputLock
    rule_deck: InputLock
    case_id: str
    rule_id: str
    original_violation_id: str
    target_violation_fingerprint: str
    expected_classification: AttemptClassification
    selection_reason: str


class ControlManifest(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    development_only: Literal[True] = True
    controls: list[ControlProvenance]


class LiveExecutionAttestation(StrictModel):
    backend: Literal["KLayoutBackend"]
    backend_image: str
    backend_image_digest: str
    fresh_workspace: str
    cache_used: Literal[False] = False
    layout_executed: bool
    drc_executed: bool
    sanity_executed: bool = True
    connectivity_executed: bool
    master_transaction_invoked: Literal[False] = False
    experience_graph_written: Literal[False] = False
    tool_command_fingerprints: list[str] = Field(default_factory=list)
    execution_protocol_version: str = "legacy-v1"
    attempt_id: str | None = None
    logical_evidence_key: str | None = None

    @model_validator(mode="after")
    def live_is_complete(self) -> "LiveExecutionAttestation":
        if not (
            self.layout_executed
            and self.drc_executed
            and self.sanity_executed
            and self.connectivity_executed
        ):
            raise ValueError("live evidence requires layout, DRC, and connectivity")
        if self.execution_protocol_version.startswith("p3.2") and (
            not self.attempt_id or not self.logical_evidence_key
        ):
            raise ValueError("P3.2 live evidence requires attempt ownership")
        return self


class BaselineValidationResult(StrictModel):
    case_id: str
    evidence_mode: Literal["LIVE_BASELINE_ROUND_TRIP"]
    official_total_drv: int
    fresh_total_drv: int
    official_per_rule_drv: dict[str, int]
    fresh_per_rule_drv: dict[str, int]
    normalized_marker_multiset_equal: bool
    connectivity_preserved: bool
    official_script_sha256: str
    generated_gds_sha256: str
    generated_drc_sha256: str
    rule_deck_sha256: str
    evaluator_hash: str
    wall_time_seconds: float
    attestation: LiveExecutionAttestation
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)


class KernelAttemptResult(StrictModel):
    attempt_id: str
    sample_id: str
    candidate_id: str
    starting_snapshot_fingerprint: str
    physical_effect_fingerprint: str
    final_normalized_drc_delta_fingerprint: str | None
    action_family: str
    operation_summary: list[str]
    target_violation_ids: list[str]
    compile_status: str
    checker_status: str
    checker_reason: str | None = None
    probe_status: str
    sandbox_status: str
    removed_target_count: int | None = Field(ge=0)
    removed_original_count: int | None = Field(ge=0)
    new_violation_count: int | None = Field(ge=0)
    connectivity_preserved: bool | None
    before_total_drv: int = Field(ge=0)
    after_total_drv: int | None = Field(ge=0)
    before_target_present: bool
    after_target_present: bool | None
    verification_artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    wall_time_seconds: float = Field(ge=0)
    evidence_mode: EvidenceMode
    classification: AttemptClassification
    strict_clean: bool = False
    attestation: LiveExecutionAttestation | None = None
    execution_protocol_version: str = "legacy-v1"
    logical_evidence_key: str | None = None
    terminal_status: AttemptTerminalStatus | None = None
    drc_validity: EvidenceValidity = EvidenceValidity.NOT_EVALUATED
    sanity_validity: EvidenceValidity = EvidenceValidity.NOT_EVALUATED
    connectivity_validity: EvidenceValidity = EvidenceValidity.NOT_EVALUATED
    fresh_evidence_valid: bool = False

    @model_validator(mode="before")
    @classmethod
    def derive_candidate_local_truth(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        raw = dict(value)
        removed_target = raw.get("removed_target_count")
        removed_original = raw.get("removed_original_count")
        new_violations = raw.get("new_violation_count")
        protocol = str(raw.get("execution_protocol_version", "legacy-v1"))
        validity_complete = (
            not protocol.startswith("p3.2")
            or (
                raw.get("drc_validity") in {
                    EvidenceValidity.VALID, EvidenceValidity.VALID.value,
                }
                and raw.get("sanity_validity") in {
                    EvidenceValidity.VALID, EvidenceValidity.VALID.value,
                }
                and raw.get("connectivity_validity") in {
                    EvidenceValidity.VALID, EvidenceValidity.VALID.value,
                }
                and raw.get("fresh_evidence_valid") is True
            )
        )
        raw["strict_clean"] = bool(
            removed_target is not None and int(removed_target) > 0
            and removed_original is not None and int(removed_original) > 0
            and new_violations is not None and int(new_violations) == 0
            and raw.get("connectivity_preserved") is True
            and validity_complete
        )
        if raw["strict_clean"]:
            raw["classification"] = AttemptClassification.CLEAN_PROGRESS
        elif raw.get("classification") in {
            AttemptClassification.CLEAN_PROGRESS,
            AttemptClassification.CLEAN_PROGRESS.value,
        }:
            raise ValueError(
                "CLEAN_PROGRESS requires candidate-local strict-clean truth"
            )
        return raw

    @model_validator(mode="after")
    def evidence_mode_is_attested(self) -> "KernelAttemptResult":
        if self.evidence_mode != EvidenceMode.HISTORICAL_REPLAY:
            if (
                self.attestation is None
                and self.classification != AttemptClassification.EXECUTION_FAIL
            ):
                raise ValueError("live evidence requires execution attestation")
        elif self.attestation is not None:
            raise ValueError("historical replay cannot carry live attestation")
        return self


class KernelSampleResult(StrictModel):
    sample_id: str
    case_id: str
    rule_id: str
    target_violation_fingerprint: str
    predicate_fidelity: str
    witness_quality: str
    witness_built: bool
    physical_geometry_count: int = 0
    editable_contributor_count: int = 0
    hierarchy_multiplicity: int = 0
    connectivity_quality: str = "unavailable"
    region_id: str | None = None
    region_marker_count: int = 0
    co_located_violation_count: int = 0
    semantic_intent_count: int = 0
    lowered_variant_count: int = 0
    checked_candidate_count: int = 0
    valid_candidate_count: int = 0
    repair_program_invoked: bool = False
    repair_program_count: int = 0
    compile_success_count: int = 0
    compile_failure_count: int = 0
    predicate_probe_pass_count: int = 0
    predicate_probe_reject_count: int = 0
    sandbox_attempt_count: int = 0
    clean_candidate_count: int = 0
    failure_stage: FailureStage
    attempt_ids: list[str] = Field(default_factory=list)


class RuleSummary(StrictModel):
    rule_id: str
    sample_count: int
    attempt_count: int
    strict_clean_attempt_count: int
    classifications: dict[str, int]
    deepest_failure_stages: dict[str, int]


class GateCheck(StrictModel):
    name: str
    status: Literal["PASS", "FAIL", "NOT_RUN"]
    evidence: str


class HarnessGate(StrictModel):
    passed: bool
    checks: list[GateCheck]

    @model_validator(mode="after")
    def derive_passed(self) -> "HarnessGate":
        expected = bool(self.checks) and all(
            item.status == "PASS" for item in self.checks
        )
        if self.passed != expected:
            raise ValueError("harness gate must be derived from all checks")
        return self


class RepairReadinessGate(StrictModel):
    passed: bool
    status: Literal["PASS", "FAIL", "NOT_RUN"]
    evidence: dict[str, Any] = Field(default_factory=dict)


class ControlExecutionResult(StrictModel):
    control_id: str
    expected_classification: AttemptClassification
    attempts: list[KernelAttemptResult]
    repeatable: bool

    @model_validator(mode="after")
    def derive_repeatability(self) -> "ControlExecutionResult":
        signatures = {
            (
                item.physical_effect_fingerprint,
                item.starting_snapshot_fingerprint,
                item.final_normalized_drc_delta_fingerprint,
                item.removed_target_count,
                item.removed_original_count,
                item.new_violation_count,
                item.connectivity_preserved,
                item.classification.value,
            )
            for item in self.attempts
        }
        expected = (
            len(self.attempts) >= 2
            and len(signatures) == 1
            and all(
                item.classification == self.expected_classification
                for item in self.attempts
            )
        )
        if self.repeatable != expected:
            raise ValueError("control repeatability must be derived")
        return self


class Phase2BenchmarkReport(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    development_only: Literal[True] = True
    sample_results: list[KernelSampleResult]
    attempts: list[KernelAttemptResult]
    rule_summaries: list[RuleSummary]
    baseline_results: list[BaselineValidationResult]
    control_results: list[ControlExecutionResult]
    harness_gate: HarnessGate
    future_repair_readiness: RepairReadinessGate

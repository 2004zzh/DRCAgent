from __future__ import annotations

from pydantic import Field

from drc_agent.schemas.common import StrictModel

from .models import AttemptClassification, FailureStage, KernelAttemptResult


class FunnelObservation(StrictModel):
    discovery_ok: bool = True
    rule_present: bool = True
    rule_supported: bool = True
    predicate_available: bool = True
    witness_available: bool = True
    witness_low_fidelity: bool = False
    hierarchy_grounded: bool = True
    physical_contributor_count: int = Field(default=1, ge=0)
    editable_contributor_count: int = Field(default=1, ge=0)
    connectivity_available: bool = True
    region_built: bool = True
    context_built: bool = True
    semantic_intent_count: int = Field(default=1, ge=0)
    lowered_variant_count: int = Field(default=1, ge=0)
    checked_candidate_count: int = Field(default=1, ge=0)
    valid_candidate_count: int = Field(default=1, ge=0)
    repair_program_required: bool = False
    repair_program_invoked: bool = False
    repair_program_count: int = Field(default=0, ge=0)
    compile_failure_count: int = Field(default=0, ge=0)
    predicate_probe_reject_count: int = Field(default=0, ge=0)
    attempts: list[KernelAttemptResult] = Field(default_factory=list)


def deepest_failure_stage(value: FunnelObservation) -> FailureStage:
    if not value.discovery_ok:
        return FailureStage.SAMPLE_DISCOVERY_FAIL
    if not value.rule_present:
        return FailureStage.RULE_NOT_PRESENT
    if not value.rule_supported:
        return FailureStage.RULE_UNSUPPORTED
    if not value.predicate_available:
        return FailureStage.PREDICATE_UNAVAILABLE
    if not value.witness_available:
        return FailureStage.WITNESS_UNAVAILABLE
    if value.witness_low_fidelity:
        return FailureStage.WITNESS_LOW_FIDELITY
    if not value.hierarchy_grounded:
        return FailureStage.HIERARCHY_GROUNDING_FAIL
    if value.physical_contributor_count == 0:
        return FailureStage.NO_PHYSICAL_CONTRIBUTOR
    if value.editable_contributor_count == 0:
        return FailureStage.NO_EDITABLE_CONTRIBUTOR
    if not value.region_built:
        return FailureStage.REGION_BUILD_FAIL
    if not value.context_built:
        return FailureStage.CONTEXT_BUILD_FAIL
    if value.semantic_intent_count == 0:
        return FailureStage.NO_SEMANTIC_INTENT
    if value.lowered_variant_count == 0:
        return FailureStage.LOWERING_EMPTY
    if (
        value.checked_candidate_count > 0
        and value.valid_candidate_count == 0
    ):
        return FailureStage.CANDIDATE_CHECK_FAIL
    if value.repair_program_required and not value.repair_program_invoked:
        return FailureStage.REPAIR_PROGRAM_NOT_INVOKED
    if value.repair_program_invoked and value.repair_program_count == 0:
        return FailureStage.REPAIR_PROGRAM_NO_OUTPUT
    if value.compile_failure_count > 0 and not value.attempts:
        return FailureStage.PROGRAM_COMPILE_FAIL
    if value.predicate_probe_reject_count > 0 and not value.attempts:
        return FailureStage.PREDICATE_PROBE_REJECT
    if any(item.strict_clean for item in value.attempts):
        return FailureStage.CLEAN_PROGRESS
    if any(
        item.classification == AttemptClassification.CONNECTIVITY_FAIL
        for item in value.attempts
    ):
        return FailureStage.CONNECTIVITY_FAIL
    if any(
        item.classification == AttemptClassification.REGRESSION
        for item in value.attempts
    ):
        return FailureStage.SANDBOX_REGRESSION
    if any(
        item.classification == AttemptClassification.NO_PROGRESS
        for item in value.attempts
    ):
        return FailureStage.SANDBOX_NO_PROGRESS
    if not value.connectivity_available:
        return FailureStage.CONNECTIVITY_UNAVAILABLE
    return FailureStage.REPAIR_PROGRAM_NO_OUTPUT

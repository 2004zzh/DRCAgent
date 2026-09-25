from __future__ import annotations

from pydantic import BaseModel, Field

from drc_agent.config.loader import AcceptanceConfig
from drc_agent.schemas.action import VerificationOutcome, VerificationResult


class AcceptanceDecision(BaseModel):
    accepted: bool
    reasons: list[str] = Field(default_factory=list)


def evaluate_acceptance(result: VerificationResult, cfg: AcceptanceConfig) -> AcceptanceDecision:
    reasons = []
    if (
        result.execution_protocol_version.startswith("p3.2")
        and not result.fresh_evidence_valid
    ):
        reasons.append("FRESH_EVIDENCE_INCOMPLETE")
    if result.script_valid is not True:
        reasons.append("SCRIPT_INVALID")
    if result.gds_sanity_pass is not True:
        reasons.append("GDS_SANITY_FAILED")
    if cfg.require_connectivity and result.connectivity_preserved is not True:
        reasons.append("CONNECTIVITY_FAILED")
    if result.timing_gate_pass is False:
        reasons.append("TIMING_FAILED")
    if cfg.require_drc_progress:
        if result.residual_violation_count is None:
            reasons.append("DRC_NOT_EVALUATED")
        elif result.residual_violation_count >= result.baseline_residual_count:
            reasons.append("NO_DRC_PROGRESS")
    if cfg.reject_new_violations and result.new_violation_count:
        reasons.append("NEW_DRC_INTRODUCED")
    accepted = not reasons
    if accepted and result.outcome != VerificationOutcome.SUCCESS:
        reasons.append("OUTCOME_NOT_SUCCESS")
        accepted = False
    return AcceptanceDecision(accepted=accepted, reasons=reasons)


def is_better_snapshot(candidate: VerificationResult, best: VerificationResult | None,
                       *, candidate_disturbance: int = 0, best_disturbance: int = 0,
                       candidate_cost: int = 0, best_cost: int = 0) -> bool:
    if (
        candidate.connectivity_preserved is not True
        or candidate.new_violation_count is None
        or candidate.residual_violation_count is None
    ):
        return False
    if best is None:
        return True
    if (
        best.connectivity_preserved is not True
        or best.new_violation_count is None
        or best.residual_violation_count is None
    ):
        return True
    candidate_key = (
        int(candidate.connectivity_preserved), -candidate.new_violation_count,
        -candidate.residual_violation_count, -candidate_disturbance, -candidate_cost,
    )
    best_key = (
        int(best.connectivity_preserved), -best.new_violation_count,
        -best.residual_violation_count, -best_disturbance, -best_cost,
    )
    return candidate_key > best_key

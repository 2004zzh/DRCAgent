from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field, model_validator

from .common import StrictModel, stable_hash


class ExecutionEnvelope(StrictModel):
    """Mutable authorization limits, kept outside scientific identity."""

    schema_version: Literal["1.0"] = "1.0"
    max_iterations: int = Field(ge=1)
    max_http_attempts_per_run: int | None = Field(default=None, ge=1)
    max_eda_evaluations_per_run: int | None = Field(default=None, ge=1)
    whole_run_budget_seconds: float | None = Field(default=None, gt=0.0)

    @property
    def content_hash(self) -> str:
        return stable_hash(self.model_dump(mode="json"))


class ContinuationRecord(StrictModel):
    """One immutable, hash-chained authorization for a continuation segment."""

    schema_version: Literal["1.0"] = "1.0"
    protocol: Literal["p5-iteration-extension-v1"] = (
        "p5-iteration-extension-v1"
    )
    run_id: str
    segment_index: int = Field(ge=1)
    nonce: str = Field(min_length=1, max_length=128)
    requester_authorization: str = Field(min_length=1, max_length=256)
    created_at: datetime
    prior_status: str
    completed_iterations: int = Field(ge=0)
    next_iteration: int = Field(ge=1)
    scientific_config_hash: str
    initial_resolved_config_hash: str
    executable_digest: str
    identity: dict[str, Any]
    old_envelope: ExecutionEnvelope
    old_envelope_hash: str
    new_envelope: ExecutionEnvelope
    new_envelope_hash: str
    amendment_fields: list[Literal[
        "max_iterations", "max_http_attempts_per_run",
        "max_eda_evaluations_per_run", "whole_run_budget_seconds",
    ]]
    safe_checkpoint: dict[str, Any]
    cumulative_usage: dict[str, Any]
    state_hashes: dict[str, str]
    new_execution_epoch: str
    previous_record_hash: str | None = None
    record_hash: str

    @model_validator(mode="after")
    def validate_hashes_and_extension(self) -> "ContinuationRecord":
        if self.old_envelope.content_hash != self.old_envelope_hash:
            raise ValueError("CONTINUATION_OLD_ENVELOPE_HASH_MISMATCH")
        if self.new_envelope.content_hash != self.new_envelope_hash:
            raise ValueError("CONTINUATION_NEW_ENVELOPE_HASH_MISMATCH")
        if self.new_envelope.max_iterations < self.old_envelope.max_iterations:
            raise ValueError("CONTINUATION_ITERATION_LIMIT_DECREASED")
        changed = []
        for field in (
            "max_iterations",
            "max_http_attempts_per_run",
            "max_eda_evaluations_per_run",
            "whole_run_budget_seconds",
        ):
            old = getattr(self.old_envelope, field)
            new = getattr(self.new_envelope, field)
            if old == new:
                continue
            if field != "max_iterations" and (
                old is None or new is None or new < old
            ):
                raise ValueError("CONTINUATION_BUDGET_MUST_INCREASE_BOUNDEDLY")
            changed.append(field)
        if not changed or self.amendment_fields != changed:
            raise ValueError("CONTINUATION_AMENDMENT_FIELDS_MISMATCH")
        if self.next_iteration != self.completed_iterations + 1:
            raise ValueError("CONTINUATION_NEXT_ITERATION_MISMATCH")
        iteration_extended = "max_iterations" in changed
        if iteration_extended and (
            self.completed_iterations != self.old_envelope.max_iterations
            or self.prior_status != "COMPLETED_MAX_ITERATIONS"
        ):
            raise ValueError("CONTINUATION_NOT_AT_COMPLETED_MAX_BOUNDARY")
        if not iteration_extended:
            if self.prior_status != "BUDGET_EXHAUSTED":
                raise ValueError("BUDGET_AMENDMENT_REQUIRES_EXHAUSTED_BOUNDARY")
            if self.completed_iterations >= self.old_envelope.max_iterations:
                raise ValueError("BUDGET_AMENDMENT_HAS_NO_REMAINING_ITERATION")
        expected = stable_hash(
            self.model_dump(mode="json", exclude={"record_hash"})
        )
        if self.record_hash != expected:
            raise ValueError("CONTINUATION_RECORD_HASH_MISMATCH")
        return self


class ContinuationVerification(StrictModel):
    """Zero-paid result for verification and idempotent no-op requests."""

    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    status: str
    verify_only: bool
    prior_status: str
    completed_iterations: int = Field(ge=0)
    next_iteration: int | None = Field(default=None, ge=1)
    safe_checkpoint: dict[str, Any]
    scientific_config_hash: str
    executable_digest: str
    current_envelope: ExecutionEnvelope
    proposed_envelope: ExecutionEnvelope | None = None
    continuation_nonce: str | None = None
    continuation_segment_index: int = Field(ge=0)
    continuation_record_hash: str | None = None
    current_snapshot_id: str
    best_snapshot_id: str
    experience_knowledge_cutoff: int | None = Field(default=None, ge=0)
    repair_attempt_memory_hash: str
    frontier_state_hash: str
    cumulative_usage: dict[str, Any]
    remaining_budget: dict[str, Any]
    paid_operations: dict[Literal["http", "eda"], int] = Field(
        default_factory=lambda: {"http": 0, "eda": 0}
    )

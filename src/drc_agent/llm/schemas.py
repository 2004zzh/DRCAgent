from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from drc_agent.schemas.common import StrictModel


class LLMCallContext(StrictModel):
    run_id: str
    iteration: int
    subgraph_id: str | None = None
    region_id: str | None = None
    purpose: Literal[
        "region_diagnosis", "neighbor_intent", "candidate_generation",
        "repair_program_generation", "repair_program_revision",
        "repair_agent_step", "repair_agent_revision_step",
        "repair_agent_execution_feedback_step",
        "blueprint_synthesis", "schema_repair", "provider_preflight",
    ]
    prompt_version: str = "region-agent-v1"


class LLMUsage(StrictModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMCallRecord(StrictModel):
    call_id: str
    provider: str
    model: str
    returned_model: str | None = None
    endpoint: str | None = None
    raw_usage: dict[str, Any] = Field(default_factory=dict)
    request_parameters: dict[str, Any] = Field(default_factory=dict)
    response_schema_hash: str | None = None
    request_id: str | None = None
    context: LLMCallContext
    started_at: datetime
    ended_at: datetime
    latency_seconds: float
    queue_wait_seconds: float = 0.0
    retry_count: int
    schema_validation_status: Literal["VALID", "INVALID", "NOT_RUN"]
    finish_reason: str | None = None
    usage: LLMUsage = Field(default_factory=LLMUsage)
    error_code: str | None = None
    error_message: str | None = None
    failure_domain: str | None = None
    failure_code: str | None = None
    retryable: bool | None = None
    http_status_code: int | None = None
    response_sha256: str | None = None


class StructuredLLMResult(StrictModel):
    value: Any
    record: LLMCallRecord


class PreflightResponse(StrictModel):
    ok: bool
    provider: str
    model: str

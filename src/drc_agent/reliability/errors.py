from __future__ import annotations

from enum import StrEnum
from typing import Any

class FailureDomain(StrEnum):
    INFRASTRUCTURE = "INFRASTRUCTURE"
    INTEGRITY = "INTEGRITY"
    INFRASTRUCTURE_TRANSIENT = "INFRASTRUCTURE_TRANSIENT"
    SEMANTIC_BINDING = "SEMANTIC_BINDING"
    PHYSICAL_VERDICT = "PHYSICAL_VERDICT"
    UNSUPPORTED_CAPABILITY = "UNSUPPORTED_CAPABILITY"
    DEFERRED_RESOURCE = "DEFERRED_RESOURCE"


class FailureCode(StrEnum):
    CONFIGURATION = "CONFIGURATION"
    AUTHENTICATION = "AUTHENTICATION"
    RATE_LIMIT = "RATE_LIMIT"
    PROVIDER_SERVER_ERROR = "PROVIDER_SERVER_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"
    REQUEST_TIMEOUT = "REQUEST_TIMEOUT"
    QUEUE_TIMEOUT = "QUEUE_TIMEOUT"
    PROVIDER_PROTOCOL_ERROR = "PROVIDER_PROTOCOL_ERROR"
    CIRCUIT_OPEN = "CIRCUIT_OPEN"
    WORKSPACE_OWNERSHIP = "WORKSPACE_OWNERSHIP"
    EVIDENCE_OWNERSHIP = "EVIDENCE_OWNERSHIP"
    ARTIFACT_PUBLICATION = "ARTIFACT_PUBLICATION"
    EDA_UNAVAILABLE = "EDA_UNAVAILABLE"
    EDA_CANCEL_TIMEOUT = "EDA_CANCEL_TIMEOUT"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    RESOURCE_QUEUE_TIMEOUT = "RESOURCE_QUEUE_TIMEOUT"
    STORAGE_PRESSURE = "STORAGE_PRESSURE"
    STORAGE_EXHAUSTED = "STORAGE_EXHAUSTED"
    OUTPUT_MISSING = "OUTPUT_MISSING"


class InfrastructureFailure(RuntimeError):
    """A provider/configuration failure that must abort formal planning.

    Deliberately not an ``LLMResponseError``: model/planning degradation
    handlers must never catch infrastructure failures.
    """

    failure_domain = FailureDomain.INFRASTRUCTURE

    def __init__(
        self,
        message: str,
        *,
        failure_code: FailureCode | str,
        retryable: bool,
        provider: str,
        model: str,
        http_status_code: int | None = None,
        retry_after_seconds: float | None = None,
        original_exception_type: str | None = None,
        failure_stage: str = "LLM_REQUEST",
    ):
        compatibility_code = (
            f"HTTP_{http_status_code}"
            if http_status_code is not None
            else original_exception_type
            if original_exception_type in {
                "ConnectError", "ReadTimeout", "ConnectTimeout",
                "TimeoutError", "TimeoutException", "LLMQueueTimeout",
            }
            else FailureCode(failure_code).value
        )
        super().__init__(message)
        # Retain the historical attributes used by provider callers without
        # inheriting from the LLM package. Infrastructure failures are shared
        # by LLM and EDA paths and must never be swallowed as model output.
        self.error_code = compatibility_code
        self.failure_code = FailureCode(failure_code)
        self.retryable = retryable
        self.provider = provider
        self.model = model
        self.http_status_code = http_status_code
        self.retry_after_seconds = retry_after_seconds
        self.original_exception_type = original_exception_type
        self.failure_stage = failure_stage

    def with_stage(self, stage: str) -> "InfrastructureFailure":
        self.failure_stage = stage
        return self

    def metadata(self) -> dict[str, Any]:
        return {
            "failure_domain": self.failure_domain.value,
            "failure_code": self.failure_code.value,
            "failure_stage": self.failure_stage,
            "retryable": self.retryable,
            "http_status_code": self.http_status_code,
            "retry_after_seconds": self.retry_after_seconds,
            "provider": self.provider,
            "model": self.model,
            "original_exception_type": self.original_exception_type,
        }


class IntegrityFailure(RuntimeError):
    """Deterministic provenance/evidence failure that blocks a run.

    Integrity failures are deliberately distinct from ordinary planner,
    binding, or candidate failures. Retrying the same snapshot cannot repair
    them, so Region-level degradation handlers must let them reach the runtime
    stop boundary.
    """

    failure_domain = FailureDomain.INTEGRITY
    retryable = False

    def __init__(
        self,
        message: str,
        *,
        failure_code: str,
        failure_stage: str = "PROVENANCE_VALIDATION",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.failure_code = str(failure_code)
        self.failure_stage = failure_stage
        self.details = dict(details or {})

    def with_stage(self, stage: str) -> "IntegrityFailure":
        self.failure_stage = stage
        return self

    def metadata(self) -> dict[str, Any]:
        return {
            "failure_domain": self.failure_domain.value,
            "failure_code": self.failure_code,
            "failure_stage": self.failure_stage,
            "retryable": False,
            "details": self.details,
        }


class InfrastructurePause(InfrastructureFailure):
    """Typed, checkpoint-preserving stop requested by shared infrastructure."""

    def __init__(
        self, message: str, *, pause_scope: str,
        health_snapshot: dict[str, Any] | None = None, **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.pause_scope = pause_scope
        self.health_snapshot = health_snapshot or {}

    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "typed_stop": "INFRASTRUCTURE_PAUSE",
            "pause_scope": self.pause_scope,
            "health_snapshot": self.health_snapshot,
        }

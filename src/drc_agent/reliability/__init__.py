"""Phase 1 infrastructure-reliability contracts."""

from .circuit_breaker import CircuitBreaker, CircuitState
from .errors import (
    FailureCode, FailureDomain, InfrastructureFailure, InfrastructurePause,
    IntegrityFailure,
)
from .policy import FailureClassification, RetryPolicy, classify_failure
from .run_validity import SAFE_CHECKPOINT_BOUNDARIES, failure_metadata

__all__ = [
    "CircuitBreaker",
    "CircuitState",
    "FailureClassification",
    "FailureCode",
    "FailureDomain",
    "InfrastructureFailure",
    "InfrastructurePause",
    "IntegrityFailure",
    "RetryPolicy",
    "SAFE_CHECKPOINT_BOUNDARIES",
    "classify_failure",
    "failure_metadata",
]

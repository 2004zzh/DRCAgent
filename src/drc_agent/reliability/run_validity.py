from __future__ import annotations

from typing import Any

from .errors import InfrastructureFailure, IntegrityFailure


SAFE_CHECKPOINT_BOUNDARIES = frozenset({
    "initialize_case",
    "window_commit",
    "window_complete",
    "iteration_complete",
})


def failure_metadata(exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, (InfrastructureFailure, IntegrityFailure)):
        return exc.metadata()
    return {
        "failure_domain": "WORKFLOW",
        "failure_code": type(exc).__name__,
        "failure_stage": "WORKFLOW",
        "retryable": False,
    }

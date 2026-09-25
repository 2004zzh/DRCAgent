"""Development-only Phase 4-R2 evidence-first closure adapters."""

from .attribution import attribute_enclosure_failure, attribute_m4_failure
from .models import (
    EnclosureFailureAttribution,
    FailureAttributionReport,
    M4FailureAttribution,
)

__all__ = [
    "EnclosureFailureAttribution",
    "FailureAttributionReport",
    "M4FailureAttribution",
    "attribute_enclosure_failure",
    "attribute_m4_failure",
]

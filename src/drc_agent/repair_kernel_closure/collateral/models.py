from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field

from drc_agent.schemas.common import Box, StrictModel


class CollateralAttributionClass(StrEnum):
    DIRECT_EDIT_COLLATERAL = "DIRECT_EDIT_COLLATERAL"
    LIKELY_LOCAL_COLLATERAL = "LIKELY_LOCAL_COLLATERAL"
    UNRELATED_BASELINE_CHANGE = "UNRELATED_BASELINE_CHANGE"
    UNKNOWN = "UNKNOWN"


class ConstraintFidelity(StrEnum):
    EXACT = "EXACT"
    REVIEWED = "REVIEWED"
    PROXY = "PROXY"
    UNKNOWN = "UNKNOWN"


class ConstraintHardness(StrEnum):
    HARD = "HARD"
    SOFT = "SOFT"


class CollateralRelationType(StrEnum):
    SAME_LAYER_SPACING = "SAME_LAYER_SPACING"
    OPPOSITE_SIDE_CLEARANCE = "OPPOSITE_SIDE_CLEARANCE"
    TIP_SIDE_SPACING = "TIP_SIDE_SPACING"
    MINIMUM_WIDTH = "MINIMUM_WIDTH"
    VIA_ENCLOSURE = "VIA_ENCLOSURE"
    VIA_METAL_INSIDE = "VIA_METAL_INSIDE"
    LOCAL_CONNECTEDNESS = "LOCAL_CONNECTEDNESS"


class GuardStatus(StrEnum):
    ACCEPT = "ACCEPT"
    ONE_STEP_UNSAT_WITH_COLLATERAL = "ONE_STEP_UNSAT_WITH_COLLATERAL"
    UNKNOWN_RISK = "UNKNOWN_RISK"


class CollateralAttribution(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    attribution_id: str
    sample_id: str
    attempt_id: str
    new_rule_id: str
    marker_bbox: Box
    layers: list[str]
    modified_contributor_ids: list[str]
    other_participant_ids: list[str] = Field(default_factory=list)
    before_relation: str
    after_relation: str
    required_relation: str | None = None
    distance_from_edit_footprint_dbu: int = Field(ge=0)
    predicate_fidelity: ConstraintFidelity
    attribution: CollateralAttributionClass
    reason_codes: list[str]


class CollateralConstraint(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    constraint_id: str
    relation_type: CollateralRelationType
    rule_id: str
    participant_ids: list[str] = Field(min_length=1)
    edge_ids: list[str] = Field(default_factory=list)
    before_measurement_dbu: int | None = None
    proposed_after_measurement_dbu: int | None = None
    required_measurement_dbu: int | None = None
    satisfied_when: Literal["AT_LEAST", "BOOLEAN_TRUE"] = "AT_LEAST"
    before_boolean: bool | None = None
    proposed_after_boolean: bool | None = None
    affected_carrier_ids: list[str] = Field(default_factory=list)
    affected_dof_ids: list[str] = Field(default_factory=list)
    fidelity: ConstraintFidelity
    hardness: ConstraintHardness
    reason_codes: list[str]


class GuardDecision(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    proposal_id: str
    status: GuardStatus
    violated_constraint_ids: list[str] = Field(default_factory=list)
    unknown_constraint_ids: list[str] = Field(default_factory=list)
    satisfied_constraint_ids: list[str] = Field(default_factory=list)
    objective_tuple: tuple[int, int, int] = (0, 0, 0)


class CollateralAttributionReport(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    report_id: str
    sample_id: str
    rule_id: str
    attempt_id: str
    edit_footprint: Box
    attributions: list[CollateralAttribution]
    constraints: list[CollateralConstraint]
    evidence_refs: list[str]


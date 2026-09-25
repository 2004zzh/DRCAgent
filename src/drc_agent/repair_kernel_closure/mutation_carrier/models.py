from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field

from drc_agent.schemas.common import StrictModel
from drc_agent.schemas.repair_program import InstanceLayerReplacement, RepairProgram


class MutationCarrierType(StrEnum):
    DIRECT_SOURCE_EDGE_EDIT = "DIRECT_SOURCE_EDGE_EDIT"
    INSTANCE_SPECIALIZED_EDGE_EDIT = "INSTANCE_SPECIALIZED_EDGE_EDIT"
    WHOLE_VIA_STACK_INSTANCE_MOVE = "WHOLE_VIA_STACK_INSTANCE_MOVE"
    INSTANCE_SPECIALIZED_LAYER_EDIT = "INSTANCE_SPECIALIZED_LAYER_EDIT"
    LOCAL_ADDITIVE_OVERRIDE = "LOCAL_ADDITIVE_OVERRIDE"
    LOCAL_ROUTE_FRAGMENT_EDIT = "LOCAL_ROUTE_FRAGMENT_EDIT"


class MutationCarrierStatus(StrEnum):
    AVAILABLE = "AVAILABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    AUTHORITY_UNSAT = "ONE_STEP_AUTHORITY_UNSAT"
    STALE_TARGET = "STALE_TARGET"


class PredicateEffectStatus(StrEnum):
    PREDICATE_TARGET_PROGRESS = "PREDICATE_TARGET_PROGRESS"
    PREDICATE_TARGET_SATISFIED = "PREDICATE_TARGET_SATISFIED"
    PREDICATE_NO_EFFECT = "PREDICATE_NO_EFFECT"


class TargetEdgeGoal(StrictModel):
    goal_id: str
    relation_id: str
    participant_id: str
    edge_id: str
    separation_axis: Literal["X", "Y"]
    direction: Literal[-1, 1]
    exact_delta_dbu: int = Field(gt=0)
    current_measurement_dbu: int = Field(ge=0)
    predicted_measurement_dbu: int = Field(ge=0)
    required_measurement_dbu: int = Field(gt=0)


class MutationCarrier(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    carrier_id: str
    carrier_type: MutationCarrierType
    status: MutationCarrierStatus
    target_participant_id: str
    target_physical_geometry_ids: list[str] = Field(min_length=1)
    target_source_object_ids: list[str] = Field(min_length=1)
    target_source_anchor_ids: list[str] = Field(min_length=1)
    target_instance_anchor_ids: list[str] = Field(min_length=1)
    compiler_insertion_owner_object_id: str
    authority_report_ids: list[str] = Field(min_length=1)
    source_object_hashes: dict[str, str]
    target_edge_goal: TargetEdgeGoal
    instance_layer_replacements: list[InstanceLayerReplacement] = Field(
        min_length=1, max_length=4,
    )
    requires_connectivity_preview: Literal[True] = True
    reason_codes: list[str]
    physical_effect_fingerprint: str


class MutationCarrierSet(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    relation_id: str
    carriers: list[MutationCarrier]
    rejected_context_object_ids: list[str] = Field(default_factory=list)
    failure_codes: list[str] = Field(default_factory=list)


class MutationProposal(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    proposal_id: str
    carrier_id: str
    predicate_effect_status: PredicateEffectStatus
    target_measurement_before_dbu: int
    target_measurement_after_dbu: int
    target_required_measurement_dbu: int
    repair_program: RepairProgram
    physical_effect_fingerprint: str
    source_validation_status: str


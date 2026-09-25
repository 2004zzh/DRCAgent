from __future__ import annotations

from typing import Literal, Optional

from pydantic import (
    ConfigDict, Field, create_model, field_validator, model_validator,
)

from drc_agent.schemas.common import StrictModel


Direction = Literal["positive_x", "negative_x", "positive_y", "negative_y"]
RoutingStrategy = Literal[
    "AUTO", "DETOUR", "TRIM_OPEN_END", "BRIDGE_ADJACENT_TRACKS",
]
SemanticObjective = Literal[
    "AUTO", "MATCH_REFERENCE_WIDTH", "SHRINK_INWARD",
    "EXPAND_OUTWARD", "ENSURE_ENCLOSURE",
    "SNAP_EDGE_TO_LEGAL_ALIGNMENT", "INCREASE_SPACING",
    "LOCAL_DETOUR_AROUND_OBSTACLE",
]
IntentAction = Literal[
    "MOVE_SHAPE", "RESIZE_SHAPE", "ADJUST_ENDPOINT",
    "ADD_POLYGON", "DELETE_POLYGON", "MOVE_VIA_STACK", "ADD_VIA_STACK",
    "DELETE_VIA_STACK", "REPLACE_VIA_STACK", "CHANGE_LAYER",
    "LOCAL_DETOUR", "LOCAL_REROUTE",
]


class RoutingRepairConstraints(StrictModel):
    routing_strategy: RoutingStrategy = "AUTO"
    preserve_connectivity: bool = True
    preserve_endpoints: bool = True
    avoid_object_ids: list[str] = Field(default_factory=list)
    preferred_direction: Direction | None = None
    target_layer: str | None = None
    tip_guard_dbu: int | None = Field(default=None, ge=0)
    max_added_length_dbu: int | None = Field(default=None, gt=0)
    require_via_at_start: bool = False
    require_via_at_end: bool = False


class LLMSemanticRoutingHints(StrictModel):
    # Old replay responses may contain removed default-heavy fields. Ignore
    # them while keeping the provider-facing JSON schema minimal.
    model_config = ConfigDict(extra="ignore")

    routing_strategy: RoutingStrategy = "AUTO"
    preserve_endpoints: bool = True
    preferred_direction: Direction | None = None
    target_layer: str | None = None


class RegionDiagnosis(StrictModel):
    root_cause_hypothesis: str
    evidence_ids: list[str]
    editable_object_ids: list[str] = Field(default_factory=list)
    frozen_object_ids: list[str] = Field(default_factory=list)
    secondary_drc_risks: list[str] = Field(default_factory=list)
    coordination_needs: list[str] = Field(default_factory=list)
    confidence_milli: int = Field(default=0, ge=0, le=1000)


class CandidateIntent(StrictModel):
    intent_id: str
    action_family: IntentAction
    target_object_ids: list[str] = Field(min_length=1)
    target_violation_ids: list[str] = Field(min_length=1)
    objective: SemanticObjective = "AUTO"
    reference_object_ids: list[str] = Field(default_factory=list)
    direction: Direction | None = None
    target_net_ids: list[str] = Field(default_factory=list)
    target_segment_ids: list[str] = Field(default_factory=list)
    target_via_ids: list[str] = Field(default_factory=list)
    repair_constraints: RoutingRepairConstraints = Field(default_factory=RoutingRepairConstraints)
    distance_candidates_dbu: list[int] = Field(default_factory=list, max_length=8)
    rationale_evidence_ids: list[str] = Field(min_length=1)
    expected_effect: str
    changed_precondition_ids: list[str] = Field(default_factory=list)
    coordination_request: str | None = None
    confidence_milli: int = Field(default=0, ge=0, le=1000)

    @field_validator("distance_candidates_dbu")
    @classmethod
    def distances_are_positive_and_unique(cls, values: list[int]) -> list[int]:
        if any(value <= 0 for value in values):
            raise ValueError("distance candidates must be positive integer DBU")
        return sorted(set(values))

    @model_validator(mode="after")
    def geometry_parameters_are_present(self) -> "CandidateIntent":
        directional = {
            "MOVE_SHAPE", "RESIZE_SHAPE", "ADJUST_ENDPOINT",
            "LOCAL_DETOUR", "LOCAL_REROUTE", "MOVE_VIA_STACK",
        }
        if self.action_family in directional and self.objective == "AUTO":
            if self.direction is None or not self.distance_candidates_dbu:
                raise ValueError(
                    "legacy AUTO directional intents require direction and "
                    "distance hints; goal-oriented intents delegate DBU "
                    "arithmetic to the deterministic lowerer"
                )
        if self.action_family in {
            "LOCAL_DETOUR", "LOCAL_REROUTE", "CHANGE_LAYER",
        } and not self.target_segment_ids:
            raise ValueError("routing macro requires target_segment_ids")
        if (
            self.repair_constraints.routing_strategy
            == "BRIDGE_ADJACENT_TRACKS"
            and (
                self.action_family != "LOCAL_REROUTE"
                or self.repair_constraints.preserve_endpoints
            )
        ):
            raise ValueError(
                "BRIDGE_ADJACENT_TRACKS requires LOCAL_REROUTE and "
                "preserve_endpoints=false; guard geometry is deterministic"
            )
        if (
            self.action_family == "CHANGE_LAYER"
            and not self.repair_constraints.target_layer
        ):
            raise ValueError("CHANGE_LAYER requires repair_constraints.target_layer")
        if self.action_family == "MOVE_VIA_STACK" and not self.target_via_ids:
            raise ValueError("MOVE_VIA_STACK requires target_via_ids")
        if set(getattr(
            self.repair_constraints, "avoid_object_ids", [],
        )) & set(self.target_object_ids):
            raise ValueError("routing target cannot also be an avoided object")
        return self


class CandidateIntentBatch(StrictModel):
    intents: list[CandidateIntent] = Field(default_factory=list, max_length=3)


class RepairHypothesis(StrictModel):
    """Non-exhaustive deterministic affordance or safe fallback.

    It constrains geometry facts but never defines the LLM candidate space.
    """

    hypothesis_id: str
    rule_id: str
    target_violation_id: str
    guidance_code: str
    root_cause: str
    topology_evidence_ids: list[str] = Field(default_factory=list)
    intent: CandidateIntent


class RuleGuidanceDecision(StrictModel):
    violation_id: str
    rule_id: str
    status: Literal["ELIGIBLE", "NO_SAFE_ACTION"]
    guidance_code: str
    reason: str
    hypothesis_ids: list[str] = Field(default_factory=list)


class RepairHypothesisChoice(StrictModel):
    hypothesis_id: str
    confidence_milli: int = Field(default=0, ge=0, le=1000)
    rationale: str = ""


class RepairHypothesisChoiceBatch(StrictModel):
    choices: list[RepairHypothesisChoice] = Field(
        default_factory=list, max_length=3,
    )


def constrained_hypothesis_choice_model(
    hypothesis_ids: set[str],
) -> type[RepairHypothesisChoiceBatch]:
    if not hypothesis_ids:
        raise ValueError("hypothesis choice constraints cannot be empty")
    hypothesis_type = Literal.__getitem__(tuple(sorted(hypothesis_ids)))
    choice_model = create_model(
        "ConstrainedRepairHypothesisChoice",
        __base__=RepairHypothesisChoice,
        hypothesis_id=(hypothesis_type, ...),
    )
    return create_model(
        "ConstrainedRepairHypothesisChoiceBatch",
        __base__=RepairHypothesisChoiceBatch,
        choices=(list[choice_model], Field(default_factory=list, max_length=3)),
    )


def constrained_region_diagnosis_model(
    *, evidence_ids: set[str], editable_object_ids: set[str],
    frozen_object_ids: set[str],
) -> type[RegionDiagnosis]:
    """Constrain all diagnosis references in the provider JSON schema."""
    evidence_type = Literal.__getitem__(tuple(sorted(evidence_ids)))
    editable_type = (
        Literal.__getitem__(tuple(sorted(editable_object_ids)))
        if editable_object_ids else str
    )
    frozen_type = (
        Literal.__getitem__(tuple(sorted(frozen_object_ids)))
        if frozen_object_ids else str
    )
    return create_model(
        "ConstrainedRegionDiagnosis",
        __base__=RegionDiagnosis,
        evidence_ids=(list[evidence_type], Field(min_length=1)),
        editable_object_ids=(
            list[editable_type],
            Field(default_factory=list, max_length=(64 if editable_object_ids else 0)),
        ),
        frozen_object_ids=(
            list[frozen_type],
            Field(default_factory=list, max_length=(64 if frozen_object_ids else 0)),
        ),
    )


def constrained_candidate_batch_model(
    *, action_families: set[str], object_ids: set[str],
    violation_ids: set[str], evidence_ids: set[str],
    distance_candidates_dbu: set[int],
    segment_ids: set[str] | None = None,
    via_ids: set[str] | None = None,
    net_ids: set[str] | None = None,
    avoid_object_ids: set[str] | None = None,
    legal_layers: set[str] | None = None,
) -> type[CandidateIntentBatch]:
    """Build the exact per-region response schema sent to the LLM."""
    if not all((action_families, object_ids, violation_ids, evidence_ids)):
        raise ValueError("candidate constraints require non-empty enums")
    if not distance_candidates_dbu:
        raise ValueError("candidate distance enum cannot be empty")

    action_type = Literal.__getitem__(tuple(sorted(action_families)))
    object_type = Literal.__getitem__(tuple(sorted(object_ids)))
    violation_type = Literal.__getitem__(tuple(sorted(violation_ids)))
    evidence_type = Literal.__getitem__(tuple(sorted(evidence_ids)))
    distance_type = Literal.__getitem__(tuple(sorted(distance_candidates_dbu)))
    segment_values = set(segment_ids or set())
    via_values = set(via_ids or set())
    net_values = set(net_ids or set())
    layer_values = set(legal_layers or set())
    layer_type = (
        Literal.__getitem__(tuple(sorted(layer_values))) if layer_values else str
    )
    constraints_model = create_model(
        "ConstrainedSemanticRoutingHints",
        __base__=LLMSemanticRoutingHints,
        target_layer=(Optional[layer_type], None),
    )
    segment_type = Literal.__getitem__(tuple(sorted(segment_values))) if segment_values else str
    via_type = Literal.__getitem__(tuple(sorted(via_values))) if via_values else str
    net_type = Literal.__getitem__(tuple(sorted(net_values))) if net_values else str
    intent_model = create_model(
        "ConstrainedCandidateIntent",
        __base__=CandidateIntent,
        action_family=(action_type, ...),
        objective=(SemanticObjective, ...),
        target_object_ids=(list[object_type], Field(min_length=1)),
        reference_object_ids=(
            list[object_type], Field(default_factory=list, max_length=4),
        ),
        target_violation_ids=(list[violation_type], Field(min_length=1)),
        repair_constraints=(
            constraints_model, Field(default_factory=constraints_model),
        ),
        target_segment_ids=(
            list[segment_type], Field(default_factory=list, max_length=(8 if segment_values else 0)),
        ),
        target_via_ids=(
            list[via_type], Field(default_factory=list, max_length=(8 if via_values else 0)),
        ),
        target_net_ids=(
            list[net_type], Field(default_factory=list, max_length=(8 if net_values else 0)),
        ),
        distance_candidates_dbu=(
            list[distance_type], Field(default_factory=list, max_length=8),
        ),
        rationale_evidence_ids=(list[evidence_type], Field(min_length=1)),
        expected_effect=(str, Field(max_length=200)),
    )
    return create_model(
        "ConstrainedCandidateIntentBatch",
        __base__=CandidateIntentBatch,
        intents=(list[intent_model], Field(default_factory=list, max_length=3)),
    )


class IntentFailure(StrictModel):
    intent_id: str
    code: str
    message: str
    semantic_fingerprint: str | None = None


class RegionAgentResult(StrictModel):
    region_id: str
    subgraph_id: str
    diagnosis: RegionDiagnosis | None
    planning_path: Literal[
        "LLM_SEMANTIC", "DETERMINISTIC_SOUND", "DEGRADED_NOOP",
        "REPAIR_KERNEL", "LEGACY_REPAIR_PROGRAMMER",
        "UNSUPPORTED_NOOP", "HELPER_UNSUPPORTED_NOOP", "CONTEXT_ONLY_NOOP",
    ] = "LLM_SEMANTIC"
    repair_hypotheses: list[RepairHypothesis] = Field(default_factory=list)
    guidance_audit: list[RuleGuidanceDecision] = Field(default_factory=list)
    hypothesis_choices: list[RepairHypothesisChoice] = Field(default_factory=list)
    intents: list[CandidateIntent] = Field(default_factory=list)
    llm_invocation_count: int = 0
    candidates: list[dict]
    failures: list[IntentFailure] = Field(default_factory=list)

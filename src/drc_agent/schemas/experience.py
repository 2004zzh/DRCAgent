from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from .common import ArtifactRef, Predicate, Provenance, StrictModel


class PredicateTemplate(StrictModel):
    kind: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class PriorKnowledge(StrictModel):
    knowledge_id: str
    title: str
    rule_families: list[str]
    layers: list[str]
    applicability: list[PredicateTemplate]
    recommended_action_families: list[str]
    avoid_conditions: list[PredicateTemplate]
    expected_secondary_risks: list[str]
    verification_order: list[str]
    provenance: Provenance
    verification_status: Literal["UNVERIFIED_PRIOR", "PARTIALLY_VALIDATED"]
    confidence_milli: int
    process_scope: list[str]
    pdk_scope: list[str]
    version: str


class ContextSignature(StrictModel):
    rule_histogram: dict[str, int]
    rule_family_histogram: dict[str, int]
    layer_histogram: dict[str, int]
    marker_type_histogram: dict[str, int]
    normalized_bbox_aspect_bins: list[int]
    geometry_moment_vector: list[int]
    editable_class_histogram: dict[str, int]
    object_kind_histogram: dict[str, int]
    net_count: int | None
    net_mapping_quality_histogram: dict[str, int] = Field(default_factory=dict)
    resource_source: str
    resource_bin_histogram: dict[str, int]
    agent_edge_histogram: dict[str, int]
    degree_histogram: list[int]
    wl_graph_hash: str
    failure_fingerprint_histogram: dict[str, int]
    timing_signature: dict | None


class ActionSignature(StrictModel):
    action_families: list[str]
    semantic_fingerprints: list[str]
    target_object_kinds: list[str] = Field(default_factory=list)
    joint: bool = False


class EpisodeMetrics(StrictModel):
    residual_violations: int | None
    new_violations: int | None
    connectivity_preserved: bool | None
    setup_wns: float | None = None
    hold_wns: float | None = None


class EpisodeOutcome(StrEnum):
    WINDOW_OBSERVED = "WINDOW_OBSERVED"
    VERIFIED_SUCCESS = "VERIFIED_SUCCESS"
    VERIFIED_PARTIAL_PROGRESS = "VERIFIED_PARTIAL_PROGRESS"
    VERIFIED_FAILURE_NEW_DRC = "VERIFIED_FAILURE_NEW_DRC"
    VERIFIED_FAILURE_CONNECTIVITY = "VERIFIED_FAILURE_CONNECTIVITY"
    VERIFIED_FAILURE_TIMING = "VERIFIED_FAILURE_TIMING"
    EXECUTION_FAILURE = "EXECUTION_FAILURE"
    INVALID_CANDIDATE = "INVALID_CANDIDATE"
    NO_OP_NO_PROGRESS = "NO_OP_NO_PROGRESS"


class ApplicabilityStatus(StrEnum):
    APPLICABLE = "TRUE"
    INAPPLICABLE = "FALSE"
    CONDITIONAL = "UNKNOWN"


class ExperienceAttributionQuality(StrEnum):
    DIRECT_CANDIDATE = "DIRECT_CANDIDATE"
    DIRECT_SMALL_BUNDLE = "DIRECT_SMALL_BUNDLE"
    DELTA_DEBUGGED = "DELTA_DEBUGGED"
    BUNDLE_ONLY = "BUNDLE_ONLY"
    UNKNOWN = "UNKNOWN"


class SymbolicLesson(StrictModel):
    """Transferable conditions; never an executable historical patch."""
    lesson_id: str
    prototype_id: str
    scope: dict[str, str]
    relation_kind: str
    rule_ids: list[str]
    role_conditions: list[dict[str, Any]]
    strategy: list[dict[str, str]]
    carrier_conditions: list[dict[str, Any]] = Field(default_factory=list)
    topology_conditions: list[dict[str, Any]] = Field(default_factory=list)
    condition_facts: list[dict[str, Any]] = Field(default_factory=list)
    physical_actions: list[dict[str, str]] = Field(default_factory=list)
    transfer_constraints: list[str] = Field(default_factory=list)
    action_families: list[str] = Field(default_factory=list)
    process_pattern: list[str] = Field(default_factory=list)
    process_steps: list[dict[str, Any]] = Field(default_factory=list)
    outcome: str
    new_rule_ids: list[str] = Field(default_factory=list)
    observation: str
    attribution_quality: str
    evidence_ids: list[str]
    trial_family_id: str
    required_current_checks: list[str]


class VerifiedCandidateTrial(StrictModel):
    trial_id: str
    window_index: int | None = None
    publication_version: int = 0
    scope: dict[str, str] = Field(default_factory=dict)
    trial_family_id: str | None = None
    symbolic_lessons: list[SymbolicLesson] = Field(default_factory=list)
    run_id: str
    case_id: str
    iteration: int
    base_snapshot_id: str
    planning_scope_id: str
    view_ids: list[str] = Field(default_factory=list)
    candidate_ids: list[str]
    candidate_fingerprint: str
    patch_plan_ref: ArtifactRef
    verification_ref: ArtifactRef
    process_evidence_refs: list[ArtifactRef] = Field(default_factory=list)
    outcome: Literal[
        "CLEAN_PROGRESS", "NO_PROGRESS", "REGRESSION",
        "OFF_TARGET_EFFECT", "CONNECTIVITY_FAIL", "SANITY_FAIL", "EXECUTION_FAIL", "TIMEOUT",
    ]
    removed_original_violation_ids: list[str] = Field(default_factory=list)
    new_violation_ids: list[str] = Field(default_factory=list)
    attribution_quality: Literal[
        "DIRECT_CANDIDATE", "DIRECT_SMALL_BUNDLE", "BUNDLE_ONLY"
    ]
    created_at: datetime


class VerifiedRepairEpisode(StrictModel):
    episode_id: str
    window_index: int | None = None
    publication_version: int = 0
    scope: dict[str, str] = Field(default_factory=dict)
    symbolic_lessons: list[SymbolicLesson] = Field(default_factory=list)
    graph_version: str
    run_id: str
    case_id: str
    pdk: str
    backend: str
    iteration: int
    planning_scope_id: str = "LEGACY_SCOPE"
    planning_scope_type: Literal[
        "FLAT_SUBGRAPH", "HIERARCHICAL_HARD_COMPONENT"
    ] = "FLAT_SUBGRAPH"
    view_ids: list[str] = Field(default_factory=list)
    subgraph_snapshot_ref: ArtifactRef
    candidate_set_ref: ArtifactRef
    selected_bundle_ref: ArtifactRef
    transaction_ref: ArtifactRef
    verification_ref: ArtifactRef
    outcome: EpisodeOutcome
    attribution_quality: ExperienceAttributionQuality = (
        ExperienceAttributionQuality.UNKNOWN
    )
    context_signature: ContextSignature
    action_signature: ActionSignature
    before_metrics: EpisodeMetrics
    after_metrics: EpisodeMetrics
    failure_stage: str | None = None
    failure_causes: list[str] = Field(default_factory=list)
    rollback_result: str | None = None
    created_at: datetime
    tool_fidelity: float
    reproducibility_status: str
    provenance: Provenance


class EpisodeNodeType(StrEnum):
    CONTEXT = "CONTEXT"
    DIAGNOSIS = "DIAGNOSIS"
    CANDIDATE = "CANDIDATE"
    SELECTED_ACTION = "SELECTED_ACTION"
    TOOL_EXECUTION = "TOOL_EXECUTION"
    DRC_FEEDBACK = "DRC_FEEDBACK"
    CONNECTIVITY_FEEDBACK = "CONNECTIVITY_FEEDBACK"
    TIMING_FEEDBACK = "TIMING_FEEDBACK"
    OUTCOME = "OUTCOME"
    FOLLOW_UP = "FOLLOW_UP"
    WINDOW_CONTEXT = "WINDOW_CONTEXT"
    REGION_CONTEXT = "REGION_CONTEXT"
    LLM_PLAN = "LLM_PLAN"
    ROOT_ATTEMPT = "ROOT_ATTEMPT"
    DEBT_ATTEMPT = "DEBT_ATTEMPT"
    CONDENSATION_ATTEMPT = "CONDENSATION_ATTEMPT"
    VERIFIED_RESULT = "VERIFIED_RESULT"
    EXECUTION_FEEDBACK = "EXECUTION_FEEDBACK"
    REVISED_PLAN = "REVISED_PLAN"
    FINAL_CLEAN = "FINAL_CLEAN"
    WINDOW_SELECTION = "WINDOW_SELECTION"
    JOINT_VERIFICATION = "JOINT_VERIFICATION"
    MASTER_OUTCOME = "MASTER_OUTCOME"


class EpisodeNode(StrictModel):
    node_id: str
    episode_id: str
    node_type: EpisodeNodeType
    payload_ref: ArtifactRef | None = None
    structured_payload: dict[str, Any] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    created_order: int
    confidence_milli: int


class EpisodeEdge(StrictModel):
    edge_id: str
    episode_id: str
    source_node_id: str
    target_node_id: str
    relation: str
    evidence_ids: list[str] = Field(default_factory=list)


class BlueprintClaim(StrictModel):
    claim: str
    evidence_ids: list[str]


class BlueprintAction(StrictModel):
    action_family: str
    applicability: list[Predicate] = Field(default_factory=list)
    avoid_conditions: list[Predicate] = Field(default_factory=list)
    parameter_bounds: dict[str, int | float | str] = Field(default_factory=dict)
    expected_secondary_risks: list[str] = Field(default_factory=list)
    evidence_ids: list[str]
    evidence_class: Literal[
        "UNVERIFIED_PRIOR", "CANDIDATE_TRIAL", "VERIFIED_EPISODE"
    ] = "UNVERIFIED_PRIOR"
    attribution_quality: str | None = None
    confidence_milli: int = 0

    @model_validator(mode="after")
    def nonempty_applicability(self) -> "BlueprintAction":
        if not self.applicability:
            raise ValueError("geometry action requires explicit applicability")
        return self


class BlueprintWarning(StrictModel):
    warning: str
    conditions: list[Predicate] = Field(default_factory=list)
    evidence_ids: list[str]


class CoordinationRequirement(StrictModel):
    region_ids: list[str]
    requirement: str
    evidence_ids: list[str]


class VerificationStep(StrictModel):
    name: str
    required: bool = True


class RollbackTrigger(StrictModel):
    code: str
    evidence_ids: list[str] = Field(default_factory=list)


class RepairBlueprint(StrictModel):
    blueprint_id: str
    symbolic_lessons: list[SymbolicLesson] = Field(default_factory=list)
    knowledge_cutoff: int | None = None
    subgraph_id: str
    planning_scope_id: str = "LEGACY_SCOPE"
    planning_scope_type: Literal[
        "FLAT_SUBGRAPH", "HIERARCHICAL_VIEW", "FACTOR_SUMMARY"
    ] = "FLAT_SUBGRAPH"
    view_ids: list[str] = Field(default_factory=list)
    graph_version: str
    query_signature_hash: str
    root_causes: list[BlueprintClaim]
    preferred_actions: list[BlueprintAction]
    conditional_actions: list[BlueprintAction] = Field(default_factory=list)
    avoid_conditions: list[BlueprintWarning]
    coordination_requirements: list[CoordinationRequirement]
    verification_order: list[VerificationStep]
    rollback_triggers: list[RollbackTrigger]
    supporting_experience_ids: list[str]
    supporting_trial_ids: list[str] = Field(default_factory=list)
    missing_evidence: list[str]
    confidence_milli: int
    generated_by: str
    validation_status: str

    @classmethod
    def empty(cls, subgraph_id: str) -> "RepairBlueprint":
        return cls(
            blueprint_id=f"{subgraph_id}:empty", subgraph_id=subgraph_id,
            planning_scope_id=subgraph_id,
            graph_version="NONE", query_signature_hash="0" * 64,
            root_causes=[], preferred_actions=[], conditional_actions=[],
            avoid_conditions=[],
            coordination_requirements=[],
            verification_order=[VerificationStep(name=name) for name in
                                ["syntax", "layout", "drc", "connectivity"]],
            rollback_triggers=[RollbackTrigger(code=code) for code in
                               ["tool_failure", "new_drc", "connectivity_failure"]],
            supporting_experience_ids=[], missing_evidence=[], confidence_milli=0,
            generated_by="EMPTY", validation_status="VALID",
        )

    def validate_citations(self, allowed_ids: set[str], action_whitelist: set[str]) -> None:
        cited: list[str] = []
        for item in [*self.root_causes, *self.preferred_actions,
                     *self.conditional_actions, *self.avoid_conditions,
                     *self.coordination_requirements]:
            cited.extend(item.evidence_ids)
            if not item.evidence_ids:
                raise ValueError("every blueprint claim must cite evidence")
        unknown = set(cited) - allowed_ids
        if unknown:
            raise ValueError(f"unknown blueprint evidence ids: {sorted(unknown)}")
        outside = {
            item.action_family
            for item in [*self.preferred_actions, *self.conditional_actions]
        } - action_whitelist
        if outside:
            raise ValueError(f"actions outside whitelist: {sorted(outside)}")


class ExperienceQuery(StrictModel):
    subgraph_id: str
    scope: dict[str, str] = Field(default_factory=dict)
    knowledge_cutoff: int | None = None
    available_dof_types: set[str] = Field(default_factory=set)
    planning_scope_id: str | None = None
    planning_scope_type: Literal[
        "FLAT_SUBGRAPH", "HIERARCHICAL_VIEW", "FACTOR_SUMMARY"
    ] = "FLAT_SUBGRAPH"
    view_ids: list[str] = Field(default_factory=list)
    current_snapshot_id: str | None = None
    pdk: str
    backend: str
    signature: ContextSignature
    allowed_action_families: set[str]
    strict_policy: bool = True
    current_coordination_requirements: list[CoordinationRequirement] = Field(
        default_factory=list
    )


class RetrievedExperience(StrictModel):
    experience_id: str
    joint: bool = False
    symbolic_lessons: list[SymbolicLesson] = Field(default_factory=list)
    publication_version: int = 0
    kind: Literal["prior", "trial", "episode"]
    outcome: str | None = None
    action_families: list[str] = Field(default_factory=list)
    channel_ranks: dict[str, int] = Field(default_factory=dict)
    rrf_score: float = 0.0
    transfer_risks: list[str] = Field(default_factory=list)
    applicability_status: ApplicabilityStatus = ApplicabilityStatus.APPLICABLE
    applicability: list[Predicate] = Field(default_factory=list)
    avoid_conditions: list[Predicate] = Field(default_factory=list)
    expected_secondary_risks: list[str] = Field(default_factory=list)
    attribution_quality: str | None = None
    tool_evidence_ids: list[str] = Field(default_factory=list)


class SeedRetrievalResult(StrictModel):
    query_hash: str
    items: list[RetrievedExperience]
    quota_gaps: list[str] = Field(default_factory=list)


class EvidencePack(StrictModel):
    query_hash: str
    items: list[RetrievedExperience]
    evidence_ids: set[str]
    sections: dict[str, list[str]]


class AdmissionResult(StrictModel):
    admitted: bool
    episode_id: str
    duplicate: bool = False
    occurrence_count: int = 1
    reason: str | None = None


class ExperienceAdoptionStage(StrEnum):
    RETRIEVED = "RETRIEVED"
    CURRENT_ROLE_BOUND = "CURRENT_ROLE_BOUND"
    EXPOSED_TO_LLM = "EXPOSED_TO_LLM"
    CITED_BY_LLM = "CITED_BY_LLM"
    SEMANTICALLY_ADOPTED = "SEMANTICALLY_ADOPTED"
    EXECUTED = "EXECUTED"
    FRESH_VERIFIED = "FRESH_VERIFIED"


class ExperienceAdoptionStageEvidence(StrictModel):
    stage: ExperienceAdoptionStage
    reached: bool
    evidence_ids: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class ExperienceAdoptionRecord(StrictModel):
    trace_record_id: str
    source_experience_id: str
    publication_version: int = 0
    lesson_id: str
    prototype_id: str
    blueprint_id: str
    region_id: str
    current_scene_ids: list[str] = Field(default_factory=list)
    current_participant_ids: list[str] = Field(default_factory=list)
    current_dof_ids: list[str] = Field(default_factory=list)
    plan_ids: list[str] = Field(default_factory=list)
    candidate_ids: list[str] = Field(default_factory=list)
    stages: list[ExperienceAdoptionStageEvidence]
    fresh_physical_evaluated: bool = False
    physical_verdict: str | None = None
    physical_evidence_ids: list[str] = Field(default_factory=list)
    avoided_by_condition: bool = False
    avoidance_condition_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def monotone_stages(self) -> "ExperienceAdoptionRecord":
        expected = list(ExperienceAdoptionStage)
        if [item.stage for item in self.stages] != expected:
            raise ValueError("experience adoption stages must be complete and ordered")
        reached = [item.reached for item in self.stages]
        for index, value in enumerate(reached):
            if value and index and not reached[index - 1]:
                raise ValueError("experience adoption stage cannot skip its predecessor")
        return self


class ExperienceAdoptionTrace(StrictModel):
    trace_id: str
    subgraph_id: str
    current_snapshot_id: str | None = None
    query_hash: str
    records: list[ExperienceAdoptionRecord] = Field(default_factory=list)
    stage_counts: dict[str, int] = Field(default_factory=dict)

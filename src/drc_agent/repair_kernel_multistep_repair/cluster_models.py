from __future__ import annotations

from pydantic import Field

from drc_agent.schemas.common import Box, StrictModel


class DebtCluster(StrictModel):
    cluster_id: str
    binding_ids: list[str]
    obligation_ids: list[str]
    rule_ids: list[str]
    relation_types: list[str]
    physical_geometry_ids: list[str]
    source_anchor_ids: list[str]
    instance_anchor_ids: list[str]
    merged_component_ids: list[str] = Field(default_factory=list)
    protected_relation_ids: list[str] = Field(default_factory=list)
    current_measurements: dict[str, object] = Field(default_factory=dict)
    required_measurements: dict[str, object] = Field(default_factory=dict)
    cluster_bbox: Box
    cluster_fidelity: str


class CoupledDebtProposal(StrictModel):
    proposal_id: str
    cluster_id: str
    carrier_ids: list[str]
    operation_intents: list[str]
    protected_relations_preserved: bool
    cluster_hard_predicates_satisfied: bool
    unresolved_obligation_count_after: int = Field(ge=0)
    modified_occurrence_count: int = Field(ge=0)
    operation_count: int = Field(ge=1, le=4)
    risk_score: int = Field(ge=0)
    physical_effect_fingerprint: str
    reason_codes: list[str] = Field(default_factory=list)


class CoupledSolveResult(StrictModel):
    cluster_id: str
    status: str
    proposals: list[CoupledDebtProposal]
    reason_codes: list[str] = Field(default_factory=list)

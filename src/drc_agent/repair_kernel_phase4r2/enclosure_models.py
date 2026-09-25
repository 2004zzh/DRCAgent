from __future__ import annotations

from pydantic import Field

from drc_agent.schemas.common import Box, StrictModel


class ProtectedEnclosureRelation(StrictModel):
    relation_id: str
    via_bbox: Box
    landing_bbox_after: Box
    left_margin_dbu: int
    right_margin_dbu: int
    bottom_margin_dbu: int
    top_margin_dbu: int
    minimum_side_dbu: int
    strengthened_side_dbu: int
    effect: str


class M1TipSideHazard(StrictModel):
    hazard_id: str
    axis: str
    tip_edge_length_dbu: int
    side_edge_length_dbu: int
    spacing_dbu: int
    required_dbu: int = 100
    participant_geometry_id: str


class EnclosureClosureScene(StrictModel):
    scene_id: str
    target_rule_id: str
    target_violation_id: str
    via_bbox: Box
    landing_bbox_before: Box
    target_instance_anchor_ids: list[str]
    minimum_side_dbu: int
    strengthened_side_dbu: int
    m1_tip_threshold_dbu: int = 144
    m1_tip_side_required_dbu: int = 100
    connectivity_quality: str
    reason_codes: list[str] = Field(default_factory=list)


class EnclosureCandidateAssessment(StrictModel):
    candidate_id: str
    carrier_type: str
    occurrence_local: bool
    target_occurrence: bool
    protected_enclosure: ProtectedEnclosureRelation | None = None
    predicted_m1_s2_hazards: list[M1TipSideHazard] = Field(default_factory=list)
    predicted_enclosure_satisfied: bool
    predicted_m1_s2_preserved: bool
    connectivity_risk: str
    executable: bool
    rank_tuple: tuple[int, int, int, int, int, str]
    reason_codes: list[str] = Field(default_factory=list)

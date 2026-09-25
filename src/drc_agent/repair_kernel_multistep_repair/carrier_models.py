from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from drc_agent.schemas.common import StrictModel


class DebtCarrierType(StrEnum):
    DIRECT_LOCAL_SOURCE_EDGE_EDIT = "DIRECT_LOCAL_SOURCE_EDGE_EDIT"
    INSTANCE_SPECIALIZED_EDGE_EDIT = "INSTANCE_SPECIALIZED_EDGE_EDIT"
    INSTANCE_SPECIALIZED_LAYER_EDIT = "INSTANCE_SPECIALIZED_LAYER_EDIT"
    WHOLE_VIA_STACK_INSTANCE_MOVE = "WHOLE_VIA_STACK_INSTANCE_MOVE"
    LOCAL_ADDITIVE_OVERRIDE = "LOCAL_ADDITIVE_OVERRIDE"
    LOCAL_ROUTE_FRAGMENT_EDIT_IF_TARGET_OWNED = "LOCAL_ROUTE_FRAGMENT_EDIT_IF_TARGET_OWNED"


class ProtectedRelationEffect(StrEnum):
    PRESERVES = "PRESERVES"
    MAY_AFFECT = "MAY_AFFECT"
    BREAKS = "BREAKS"


class DebtCarrier(StrictModel):
    carrier_id: str
    carrier_type: DebtCarrierType
    binding_id: str
    physical_geometry_ids: list[str]
    source_object_ids: list[str]
    source_anchor_ids: list[str]
    instance_anchor_ids: list[str]
    source_cells: list[str]
    occurrence_local: bool
    canonical_expressible: bool
    requires_connectivity_preview: bool
    protected_relation_effect: ProtectedRelationEffect
    authority_basis: list[str]
    predicted_relation_effect: str
    risk_flags: list[str] = Field(default_factory=list)


class ParticipantAuthorityProof(StrictModel):
    physical_geometry_id: str
    source_object_id: str | None = None
    source_anchor_id: str | None = None
    instance_anchor_id: str | None = None
    source_cell: str | None = None
    direct_editable: bool
    occurrence_specializable: bool
    whole_via_capable: bool
    additive_override_capable: bool
    legal_carrier_ids: list[str]
    reason_codes: list[str]


class DebtCarrierSet(StrictModel):
    carrier_set_id: str
    binding_id: str
    carriers: list[DebtCarrier]
    participant_authority: list[ParticipantAuthorityProof]
    status: str
    reason_codes: list[str]

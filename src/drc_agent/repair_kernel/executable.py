from __future__ import annotations

from enum import StrEnum
from math import lcm
from typing import Literal

from pydantic import Field, model_validator

from drc_agent.schemas.common import StrictModel

from .models import RepairDOF


def intersect_dof_domains(dofs: list[RepairDOF]) -> tuple[list[tuple[int, int]], int]:
    """Intersect contributor unions, clipped to domains and the shared grid."""
    if not dofs:
        return [], 1
    grid = lcm(*(item.manufacturing_grid_dbu for item in dofs))
    intersection = None
    for item in dofs:
        intervals = []
        for low, high in item.allowed_intervals:
            low, high = max(low,item.domain_min_dbu), min(high,item.domain_max_dbu)
            low, high = -(-low//grid)*grid, (high//grid)*grid
            if low <= high:
                intervals.append((low,high))
        if intersection is not None:
            intervals = [(max(a,c),min(b,d)) for a,b in intersection for c,d in intervals
                         if max(a,c)<=min(b,d)]
        merged = []
        for low, high in sorted(intervals):
            if merged and low <= merged[-1][1]+grid:
                merged[-1] = (merged[-1][0],max(high,merged[-1][1]))
            else:
                merged.append((low,high))
        intersection = merged
        if not intersection:
            break
    return intersection or [], grid


class CompilerCapabilityStatus(StrEnum):
    EXPRESSIBLE = "EXPRESSIBLE"
    EXPRESSIBLE_WITH_INSTANCE_SPECIALIZATION = (
        "EXPRESSIBLE_WITH_INSTANCE_SPECIALIZATION"
    )
    UNEXPRESSIBLE = "UNEXPRESSIBLE"
    AMBIGUOUS = "AMBIGUOUS"
    STALE = "STALE"


class ExecutableDOFFailureCode(StrEnum):
    NO_CARRIER = "EXEC_DOF_NO_CARRIER"
    AUTHORITY_FAIL = "EXEC_DOF_AUTHORITY_FAIL"
    COMPILER_UNEXPRESSIBLE = "EXEC_DOF_COMPILER_UNEXPRESSIBLE"
    AMBIGUOUS_LINEAGE = "EXEC_DOF_AMBIGUOUS_LINEAGE"
    UNBOUNDED_CARRIER = "EXEC_DOF_UNBOUNDED_CARRIER"
    STALE = "EXEC_DOF_STALE"
    PROTECTED_CONSTRAINT = "EXEC_DOF_PROTECTED_CONSTRAINT"


class SourceSnapshotIdentity(StrictModel):
    """The source and physical truth used to construct executable variables."""

    schema_version: Literal["1.0"] = "1.0"
    snapshot_id: str
    script_sha256: str
    gds_sha256: str | None = None
    drc_sha256: str | None = None
    connectivity_sha256: str | None = None
    source_mapping_sha256: str
    hierarchy_projection_sha256: str
    hierarchy_projection_version: str = "repair-scene-v1"


class CompilerCapabilityResult(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    capability_id: str
    status: CompilerCapabilityStatus
    primitive: str | None = None
    atomic_source_target_ids: list[str] = Field(default_factory=list)
    source_anchor_ids: list[str] = Field(default_factory=list)
    instance_anchor_ids: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)


class ExecutableMutationVariable(StrictModel):
    """A physical DOF already bound to one atomic source mutation carrier."""

    schema_version: Literal["1.0"] = "1.0"
    variable_id: str
    raw_dofs: list[RepairDOF] = Field(min_length=1)
    physical_participant_ids: list[str] = Field(min_length=1)
    source_object_ids: list[str] = Field(default_factory=list)
    source_anchor_ids: list[str] = Field(default_factory=list)
    instance_anchor_ids: list[str] = Field(default_factory=list)
    carrier_kind: str
    atomic_source_target_ids: list[str] = Field(min_length=1)
    operation_family: str
    parameter_kind: str
    axis: Literal["X", "Y"] | None = None
    edge: Literal["LEFT", "RIGHT", "BOTTOM", "TOP"] | None = None
    legal_intervals_dbu: list[tuple[int, int]] = Field(min_length=1)
    manufacturing_grid_dbu: int = Field(gt=0)
    authority_status: str
    compiler_status: CompilerCapabilityStatus
    requires_instance_specialization: bool = False
    protected_relation_ids: list[str] = Field(default_factory=list)
    connectivity_risk: str = "REQUIRES_PREVIEW"
    capability: CompilerCapabilityResult

    @property
    def raw_dof_ids(self) -> list[str]:
        return [item.dof_id for item in self.raw_dofs]

    @model_validator(mode="after")
    def executable_by_construction(self) -> "ExecutableMutationVariable":
        if self.compiler_status not in {
            CompilerCapabilityStatus.EXPRESSIBLE,
            CompilerCapabilityStatus.EXPRESSIBLE_WITH_INSTANCE_SPECIALIZATION,
        }:
            raise ValueError("an executable variable requires compiler capability")
        if self.compiler_status != self.capability.status:
            raise ValueError("variable/compiler capability status mismatch")
        if self.requires_instance_specialization != (
            self.compiler_status
            == CompilerCapabilityStatus.EXPRESSIBLE_WITH_INSTANCE_SPECIALIZATION
        ):
            raise ValueError("specialization flag does not match compiler capability")
        if any(low > high for low, high in self.legal_intervals_dbu):
            raise ValueError("executable variable has an empty legal interval")
        return self


class ExecutableConstraint(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    constraint_id: str
    constraint_class: Literal[
        "PRIMARY", "PROTECTED", "COUPLING", "LOCALITY", "COMPILER",
    ]
    relation_kind: str
    variable_ids: list[str] = Field(default_factory=list)
    predicate: str
    parameters: dict = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)


class ExecutableDOFFailure(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    raw_dof_ids: list[str] = Field(default_factory=list)
    participant_ids: list[str] = Field(default_factory=list)
    source_object_ids: list[str] = Field(default_factory=list)
    code: ExecutableDOFFailureCode
    reason: str
    compiler_status: CompilerCapabilityStatus | None = None


class ExecutableRepairModel(StrictModel):
    """The only variable space accepted by executable root realization."""

    schema_version: Literal["1.0"] = "1.0"
    model_id: str
    scene_id: str
    target_witness_id: str
    variables: list[ExecutableMutationVariable] = Field(default_factory=list)
    primary_constraints: list[ExecutableConstraint] = Field(default_factory=list)
    protected_constraints: list[ExecutableConstraint] = Field(default_factory=list)
    coupling_constraints: list[ExecutableConstraint] = Field(default_factory=list)
    locality_constraints: list[ExecutableConstraint] = Field(default_factory=list)
    compiler_constraints: list[ExecutableConstraint] = Field(default_factory=list)
    rejected_dofs: list[ExecutableDOFFailure] = Field(default_factory=list)
    source_snapshot: SourceSnapshotIdentity
    compiler_preflight_status: Literal["PASS", "FAIL"]
    raw_dof_count: int = Field(ge=0)

    @property
    def executable_dof_ids(self) -> set[str]:
        return {
            item.dof_id for variable in self.variables for item in variable.raw_dofs
        }

    @model_validator(mode="after")
    def coherent_model(self) -> "ExecutableRepairModel":
        if self.compiler_preflight_status == "FAIL" and self.variables:
            raise ValueError("failed compiler preflight cannot expose variables")
        variable_ids = [item.variable_id for item in self.variables]
        if len(variable_ids) != len(set(variable_ids)):
            raise ValueError("executable variable IDs must be unique")
        if len(self.executable_dof_ids) > self.raw_dof_count:
            raise ValueError("executable DOF count exceeds raw DOF count")
        return self

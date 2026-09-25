from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from drc_agent.repair_kernel import RepairFamily, classify_repair_family
from drc_agent.repair_kernel.scene import build_repair_scene
from drc_agent.repair_kernel.dof import generate_repair_dofs
from drc_agent.schemas.action import DesignState
from drc_agent.schemas.common import StrictModel, stable_hash
from drc_agent.schemas.state import (
    AgentSubgraph,
    LayoutObject,
    NeighborMessage,
    RegionState,
    RuleCatalog,
    ViolationRecord,
)
from drc_agent.schemas.experience import RepairBlueprint


class FormalKernelContext(StrictModel):
    """Read-only formal Region context presented to the kernel.

    The context deliberately keeps the authoritative source map and the
    reviewed predicate/witness dictionaries together.  A planner may choose
    a participant or DOF, but it cannot supply geometry authority.
    """

    run_id: str
    iteration: int
    snapshot_id: str | None
    subgraph_id: str
    region: RegionState
    violations: list[ViolationRecord]
    objects: list[LayoutObject]
    design: DesignState
    catalog: RuleCatalog
    script: Path
    neighbor_messages: list[NeighborMessage] = Field(default_factory=list)
    blueprint: RepairBlueprint | None = None
    failure_memory: list[dict[str, Any]] = Field(default_factory=list)
    # ``None`` preserves the historical all-Region interface. Production P4
    # planning supplies an explicit bounded subset; an empty list therefore
    # means that no target was admitted, rather than "all targets".
    planning_target_violation_ids: list[str] | None = None
    planning_role: Literal["TARGET_REPAIR", "DEPENDENCY_ASSIST"] = "TARGET_REPAIR"
    assisted_region_ids: list[str] = Field(default_factory=list)
    assisted_target_violation_ids: list[str] = Field(default_factory=list)
    helper_dependency_evidence_ids: list[str] = Field(default_factory=list)
    helper_potential_dof_capability_ids: list[str] = Field(default_factory=list)
    helper_potential_carrier_ids: list[str] = Field(default_factory=list)
    helper_protected_relation_ids: list[str] = Field(default_factory=list)
    context_hash: str
    planner_semantic_view: dict[str, Any] = Field(default_factory=dict)
    dof_capability_summary: dict[str, dict[str, Any]] = Field(default_factory=dict)
    executable_constraint_dictionary: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
    )
    executable_rejection_reason_dictionary: dict[str, list[str]] = Field(
        default_factory=dict,
    )
    # Installed once from source-backed current scenes/DOFs and shared by
    # prompt, follow-up, binding, lowering, and root registration.
    plan_semantic_registry: dict[str, Any] = Field(default_factory=dict)

    @property
    def region_violations(self) -> list[ViolationRecord]:
        wanted = (
            set(self.assisted_target_violation_ids)
            if self.planning_role == "DEPENDENCY_ASSIST"
            else set(self.region.violation_ids)
        )
        if self.planning_target_violation_ids is not None:
            wanted &= set(self.planning_target_violation_ids)
        return sorted(
            [item for item in self.violations if item.violation_id in wanted],
            key=lambda item: item.violation_id,
        )

    @property
    def omitted_region_violation_ids(self) -> list[str]:
        selected = {item.violation_id for item in self.region_violations}
        return sorted(set(self.region.violation_ids) - selected)


def compatibility_metadata(
    context: object, field_name: str,
) -> dict[str, Any]:
    """Return optional P4 planner metadata for typed and legacy contexts.

    Production ``FormalKernelContext`` declares these fields explicitly. Older
    frozen replays and integration callers legitimately use writable namespace
    objects which predate them. Install an explicit empty mapping there so all
    planner paths share one compatibility contract. Invalid values stay errors.
    """
    value = getattr(context, field_name, None)
    if value is None:
        value = {}
        try:
            setattr(context, field_name, value)
        except AttributeError:
            # A read-only legacy context can consume the conservative default;
            # production typed contexts never take this compatibility branch.
            pass
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a mapping")
    return value


def dof_capability_summary(context: object) -> dict[str, dict[str, Any]]:
    return compatibility_metadata(context, "dof_capability_summary")


def planner_semantic_view(context: object) -> dict[str, Any]:
    return compatibility_metadata(context, "planner_semantic_view")


def build_formal_kernel_context(
    *,
    run_id: str,
    iteration: int,
    snapshot_id: str | None,
    subgraph: AgentSubgraph,
    region: RegionState,
    violations: list[ViolationRecord],
    objects: list[LayoutObject],
    design: DesignState,
    catalog: RuleCatalog,
    script: Path,
    neighbor_messages: list[NeighborMessage] | None = None,
    blueprint: RepairBlueprint | None = None,
    failure_memory: list[dict[str, Any]] | None = None,
    planning_target_violation_ids: list[str] | None = None,
    planning_role: Literal["TARGET_REPAIR", "DEPENDENCY_ASSIST"] = "TARGET_REPAIR",
    assisted_region_ids: list[str] | None = None,
    assisted_target_violation_ids: list[str] | None = None,
    helper_dependency_evidence_ids: list[str] | None = None,
    helper_potential_dof_capability_ids: list[str] | None = None,
    helper_potential_carrier_ids: list[str] | None = None,
    helper_protected_relation_ids: list[str] | None = None,
) -> FormalKernelContext:
    assisted_region_ids = list(dict.fromkeys(assisted_region_ids or []))
    assisted_target_violation_ids = list(dict.fromkeys(
        assisted_target_violation_ids or []
    ))
    helper_dependency_evidence_ids = list(dict.fromkeys(
        helper_dependency_evidence_ids or []
    ))
    if planning_role == "DEPENDENCY_ASSIST" and (
        not assisted_region_ids
        or not assisted_target_violation_ids
        or not helper_dependency_evidence_ids
    ):
        raise ValueError("DEPENDENCY_ASSIST_CONTEXT_INCOMPLETE")
    if planning_target_violation_ids is not None:
        selected = list(dict.fromkeys(planning_target_violation_ids))
        allowed_targets = (
            set(assisted_target_violation_ids)
            if planning_role == "DEPENDENCY_ASSIST"
            else set(region.violation_ids)
        )
        outside = set(selected) - allowed_targets
        if outside:
            raise ValueError(
                "PLANNING_TARGET_OUTSIDE_AUTHORIZED_ROLE:"
                + ",".join(sorted(outside))
            )
        known = {item.violation_id for item in violations}
        missing = set(selected) - known
        if missing:
            raise ValueError(
                "PLANNING_TARGET_RECORD_UNAVAILABLE:"
                + ",".join(sorted(missing))
            )
        planning_target_violation_ids = selected
    design_payload = design.model_dump(mode="json")
    # Pydantic serializes a set to a list before ``stable_hash`` sees it.
    # Hash-randomized set iteration would otherwise change the formal context,
    # semantic registry, and frozen prompt across otherwise identical Python
    # processes.  Preserve list order everywhere else; only the declared set
    # field receives canonical ordering.
    design_payload["legal_layers"] = sorted(design.legal_layers)
    payload = {
        "run_id": run_id,
        "iteration": iteration,
        "snapshot_id": snapshot_id,
        "subgraph_id": subgraph.subgraph_id,
        "region": region.model_dump(mode="json"),
        "violations": [item.model_dump(mode="json") for item in violations],
        "objects": [item.model_dump(mode="json") for item in objects],
        "design": design_payload,
        "catalog": catalog.model_dump(mode="json"),
        "script": str(script.resolve()),
        "neighbor_messages": [
            item.model_dump(mode="json") for item in (neighbor_messages or [])
        ],
        "blueprint": blueprint.model_dump(mode="json") if blueprint else None,
        "failure_memory": failure_memory or [],
        "planning_target_violation_ids": planning_target_violation_ids,
        "planning_role": planning_role,
        "assisted_region_ids": assisted_region_ids,
        "assisted_target_violation_ids": assisted_target_violation_ids,
        "helper_dependency_evidence_ids": helper_dependency_evidence_ids,
        "helper_potential_dof_capability_ids": (
            helper_potential_dof_capability_ids or []
        ),
        "helper_potential_carrier_ids": helper_potential_carrier_ids or [],
        "helper_protected_relation_ids": helper_protected_relation_ids or [],
    }
    return FormalKernelContext(
        run_id=run_id,
        iteration=iteration,
        snapshot_id=snapshot_id,
        subgraph_id=subgraph.subgraph_id,
        region=region,
        violations=violations,
        objects=objects,
        design=design,
        catalog=catalog,
        script=script,
        neighbor_messages=list(neighbor_messages or []),
        blueprint=blueprint,
        failure_memory=list(failure_memory or []),
        planning_target_violation_ids=planning_target_violation_ids,
        planning_role=planning_role,
        assisted_region_ids=assisted_region_ids,
        assisted_target_violation_ids=assisted_target_violation_ids,
        helper_dependency_evidence_ids=helper_dependency_evidence_ids,
        helper_potential_dof_capability_ids=(
            helper_potential_dof_capability_ids or []
        ),
        helper_potential_carrier_ids=helper_potential_carrier_ids or [],
        helper_protected_relation_ids=helper_protected_relation_ids or [],
        context_hash=stable_hash(payload),
    )


def _editable_objects(
    context: FormalKernelContext,
    violation: ViolationRecord | None = None,
) -> list[LayoutObject]:
    """Return editable objects, witness-filtered for the primary relation.

    The frozen Phase-3 scene builder accepts an editable-object list.  Passing
    every nearby editable object would let proximity win over the authoritative
    witness, so the integration adapter projects only witness contributors into
    the primary scene; other objects remain available to the formal runtime's
    surrounding context.
    """
    allowed = set(context.region.editable_object_ids)
    if violation is not None:
        raw = context.design.rule_witnesses.get(violation.violation_id) or {}
        try:
            witness = raw if isinstance(raw, dict) else raw.model_dump(mode="json")
        except AttributeError:
            witness = {}
        grounded = set(witness.get("editable_source_object_ids", []) or [])
        grounded.update(witness.get("editable_contributor_ids", []) or [])
        grounded.update(witness.get("source_contributor_ids", []) or [])
        for geometry in witness.get("physical_geometries", []) or []:
            if geometry.get("editable"):
                for key in ("source_object_id", "geometry_id"):
                    value = geometry.get(key)
                    if value:
                        grounded.add(str(value))
        projected = allowed & grounded
        if projected:
            allowed = projected
        else:
            # No witness-owned editable object is a valid primary carrier.
            allowed = set()
    return [
        item for item in sorted(context.objects, key=lambda value: value.object_id)
        if item.object_id in allowed
    ]


def build_scene_and_dofs(
    context: FormalKernelContext,
    violation: ViolationRecord,
) -> tuple[Any, list[Any], dict[str, Any]]:
    """Build the frozen Phase-3 scene/DOF through its public deterministic APIs."""
    raw_predicate = context.design.rule_predicates.get(violation.rule_id)
    raw_witness = context.design.rule_witnesses.get(violation.violation_id)
    if raw_predicate is None:
        raise ValueError("RULE_PREDICATE_UNAVAILABLE")
    if raw_witness is None:
        raise ValueError("RULE_WITNESS_UNAVAILABLE")
    scene_context = {
        "sample": {"co_located_violation_ids": context.region.violation_ids},
        "target_violation": violation.model_dump(mode="json"),
        "rule_predicate": raw_predicate,
        "rule_witness": raw_witness,
        "region": context.region.model_dump(mode="json"),
        "editable_objects": [
            item.model_dump(mode="json") for item in _editable_objects(context, violation)
        ],
        "phase3_physical_geometries": list(
            context.design.physical_geometries.values()
        ),
    }
    scene = build_repair_scene(scene_context)
    dofs = generate_repair_dofs(scene)
    details = {
        "family": scene.repair_family.value,
        "scene": scene.model_dump(mode="json"),
        "dofs": [item.model_dump(mode="json") for item in dofs],
    }
    return scene, dofs, details


def support_for_violation(
    context: FormalKernelContext,
    violation: ViolationRecord,
) -> tuple[RepairFamily, str, list[str]]:
    raw = context.design.rule_predicates.get(violation.rule_id)
    if raw is None:
        return RepairFamily.UNSUPPORTED, "UNAVAILABLE", ["RULE_PREDICATE_UNAVAILABLE"]
    try:
        family = classify_repair_family(raw)
    except (TypeError, ValueError) as exc:
        return RepairFamily.UNSUPPORTED, "UNAVAILABLE", [
            "RULE_PREDICATE_INVALID", type(exc).__name__,
        ]
    if family == RepairFamily.UNSUPPORTED:
        return family, "UNAVAILABLE", ["UNSUPPORTED_REPAIR_FAMILY"]
    raw_witness = context.design.rule_witnesses.get(violation.violation_id) or {}
    fidelity = str(raw_witness.get("predicate_fidelity", "UNAVAILABLE"))
    if "UNAVAILABLE" in fidelity.upper():
        return family, fidelity, ["WITNESS_FIDELITY_UNAVAILABLE"]
    if "PROXY" in fidelity.upper():
        return family, fidelity, ["WITNESS_FIDELITY_PROXY"]
    return family, fidelity, []

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from drc_agent.schemas.common import StrictModel

from .llm_planner import KernelRepairPlan


PlanOrigin = Literal[
    "REAL_LLM",
    "FROZEN_REAL_LLM_REPLAY",
    "FAKE_LLM",
    "DETERMINISTIC_TEST",
]


class BoundKernelPlan(StrictModel):
    plan_id: str
    origin: PlanOrigin
    selected_violation_ids: list[str]
    selected_witness_ids: list[str] = Field(default_factory=list)
    preferred_participant_ids: list[str] = Field(default_factory=list)
    preferred_dof_ids: list[str] = Field(default_factory=list)
    allowed_dof_ids: list[str] = Field(default_factory=list)
    forbidden_dof_ids: list[str] = Field(default_factory=list)
    max_operation_count: int
    max_trajectory_depth: int
    strategy: str
    fallback_strategy: str | None = None
    coordination_request: str | None = None
    binding_effects: list[str] = Field(default_factory=list)


class PlanBindingReport(StrictModel):
    plan_id: str
    status: Literal["PASS", "FAIL"]
    origin: PlanOrigin
    errors: list[str] = Field(default_factory=list)
    binding_effects: list[str] = Field(default_factory=list)
    registry_id: str | None = None
    registry_schema_version: str | None = None
    registry_snapshot_id: str | None = None
    identifier_diagnostics: list[dict[str, Any]] = Field(default_factory=list)


class PlanBindingError(ValueError):
    def __init__(self, report: PlanBindingReport):
        super().__init__("; ".join(report.errors) or "kernel plan binding failed")
        self.report = report


class ExecutionFeedbackRevisionError(ValueError):
    pass


def validate_execution_feedback_revision(
    original: KernelRepairPlan,
    revised: KernelRepairPlan,
    *,
    kind: Literal[
        "PREFERRED_DOF_INFEASIBLE", "DEPTH_EXTENSION_AVAILABLE",
        "PHYSICAL_CONSTRAINT_FAILURE",
    ],
    allowed_ids: dict[str, list[str]],
    configured_max_depth: int,
    feasible_alternative_dof_sets: list[list[str]] | None = None,
    feasible_alternative_operation_bounds: list[dict[str, Any]] | None = None,
) -> Literal["ACCEPTED", "DECLINED"]:
    """Validate the one-field authority granted by execution feedback."""
    mutable_by_kind = {
        "PREFERRED_DOF_INFEASIBLE": {
            "preferred_dof_ids", "forbidden_dof_ids",
            "max_operation_count",
        },
        "DEPTH_EXTENSION_AVAILABLE": {"max_trajectory_depth"},
        "PHYSICAL_CONSTRAINT_FAILURE": {
            "preferred_participant_ids", "preferred_dof_ids", "strategy",
            "fallback_strategy", "max_operation_count", "coordination_request",
        },
    }
    mutable = {
        "plan_id", "rationale_evidence_ids", "confidence_milli",
        *mutable_by_kind[kind],
    }
    before = original.model_dump(mode="json")
    after = revised.model_dump(mode="json")
    changed = sorted(
        key for key in before if before[key] != after[key] and key not in mutable
    )
    if changed:
        raise ExecutionFeedbackRevisionError(
            "EXECUTION_REVISION_SCOPE_VIOLATION:" + ",".join(changed)
        )
    if (
        kind != "PREFERRED_DOF_INFEASIBLE"
        and revised.forbidden_dof_ids != original.forbidden_dof_ids
    ):
        raise ExecutionFeedbackRevisionError(
            "EXECUTION_REVISION_FORBIDDEN_DOF_CHANGED"
        )
    if kind == "PREFERRED_DOF_INFEASIBLE":
        alternatives = {
            frozenset(str(item) for item in option)
            for option in (feasible_alternative_dof_sets or [])
            if option
        }
        if not alternatives:
            raise ExecutionFeedbackRevisionError(
                "EXECUTION_REVISION_ALTERNATIVE_PROOF_MISSING"
            )
        operation_bounds: dict[frozenset[str], int] = {}
        for item in feasible_alternative_operation_bounds or []:
            option = frozenset(str(value) for value in item.get("dof_ids", []))
            required = item.get("minimum_max_operation_count")
            if (
                option not in alternatives
                or not isinstance(required, int)
                or required < 1
                or required > 4
            ):
                raise ExecutionFeedbackRevisionError(
                    "EXECUTION_REVISION_OPERATION_BOUND_PROOF_INVALID"
                )
            operation_bounds[option] = min(
                required, operation_bounds.get(option, required)
            )
        unknown = sorted(
            (
                set(revised.preferred_dof_ids)
                | set(revised.forbidden_dof_ids)
            ) - set(allowed_ids["dof_ids"])
        )
        if unknown:
            raise ExecutionFeedbackRevisionError(
                "EXECUTION_REVISION_UNKNOWN_DOF:" + ",".join(unknown)
            )
        added_forbidden = (
            set(revised.forbidden_dof_ids)
            - set(original.forbidden_dof_ids)
        )
        if added_forbidden:
            raise ExecutionFeedbackRevisionError(
                "EXECUTION_REVISION_ADDED_FORBIDDEN_DOF:"
                + ",".join(sorted(added_forbidden))
            )
        proved_ids = set().union(*alternatives)
        newly_unforbidden = (
            set(original.forbidden_dof_ids)
            - set(revised.forbidden_dof_ids)
        )
        if not newly_unforbidden <= proved_ids:
            raise ExecutionFeedbackRevisionError(
                "EXECUTION_REVISION_UNPROVED_DOF_UNFORBIDDEN:"
                + ",".join(sorted(newly_unforbidden - proved_ids))
            )
        preferred = frozenset(revised.preferred_dof_ids)
        if preferred and preferred not in alternatives:
            raise ExecutionFeedbackRevisionError(
                "EXECUTION_REVISION_PREFERRED_SET_NOT_PROVED"
            )
        if not preferred and not any(
            option.isdisjoint(revised.forbidden_dof_ids)
            for option in alternatives
        ):
            raise ExecutionFeedbackRevisionError(
                "EXECUTION_REVISION_ALL_PROVED_ALTERNATIVES_FORBIDDEN"
            )
        viable = (
            [preferred]
            if preferred
            else [
                option for option in alternatives
                if option.isdisjoint(revised.forbidden_dof_ids)
            ]
        )
        proved_required = [
            operation_bounds[option] for option in viable
            if option in operation_bounds
        ]
        if revised.max_operation_count != original.max_operation_count:
            if not proved_required:
                raise ExecutionFeedbackRevisionError(
                    "EXECUTION_REVISION_OPERATION_BOUND_INCREASE_UNPROVED"
                )
            minimum_required = min(proved_required)
            if minimum_required <= original.max_operation_count:
                raise ExecutionFeedbackRevisionError(
                    "EXECUTION_REVISION_OPERATION_BOUND_INCREASE_UNNECESSARY"
                )
            if revised.max_operation_count != minimum_required:
                raise ExecutionFeedbackRevisionError(
                    "EXECUTION_REVISION_OPERATION_BOUND_NOT_PROVED_MINIMUM"
                )
        elif proved_required and revised.max_operation_count < min(proved_required):
            raise ExecutionFeedbackRevisionError(
                "EXECUTION_REVISION_OPERATION_BOUND_STILL_INFEASIBLE"
            )
        if (
            revised.preferred_dof_ids == original.preferred_dof_ids
            and revised.forbidden_dof_ids == original.forbidden_dof_ids
            and revised.max_operation_count == original.max_operation_count
        ):
            return "DECLINED"
        if revised.plan_id == original.plan_id:
            raise ExecutionFeedbackRevisionError(
                "EXECUTION_REVISION_PLAN_ID_NOT_CHANGED"
            )
        return "ACCEPTED"
    if kind == "PHYSICAL_CONSTRAINT_FAILURE":
        unknown_participants = sorted(
            set(revised.preferred_participant_ids)
            - set(allowed_ids["participant_ids"])
        )
        unknown_dofs = sorted(
            set(revised.preferred_dof_ids) - set(allowed_ids["dof_ids"])
        )
        if unknown_participants or unknown_dofs:
            raise ExecutionFeedbackRevisionError(
                "EXECUTION_REVISION_UNKNOWN_PHYSICAL_BINDING:"
                + ",".join(unknown_participants + unknown_dofs)
            )
        semantic_fields = mutable_by_kind[kind]
        changed_semantics = any(
            before[field] != after[field] for field in semantic_fields
        )
        if not changed_semantics:
            return "DECLINED"
        if revised.plan_id == original.plan_id:
            raise ExecutionFeedbackRevisionError(
                "EXECUTION_REVISION_PLAN_ID_NOT_CHANGED"
            )
        return "ACCEPTED"
    if revised.max_trajectory_depth < original.max_trajectory_depth:
        raise ExecutionFeedbackRevisionError(
            "EXECUTION_REVISION_DEPTH_DECREASED"
        )
    if revised.max_trajectory_depth > configured_max_depth:
        raise ExecutionFeedbackRevisionError(
            "EXECUTION_REVISION_DEPTH_EXCEEDS_CONFIGURED_MAX"
        )
    if (
        revised.max_trajectory_depth > original.max_trajectory_depth
        and revised.plan_id == original.plan_id
    ):
        raise ExecutionFeedbackRevisionError(
            "EXECUTION_REVISION_PLAN_ID_NOT_CHANGED"
        )
    return (
        "ACCEPTED"
        if revised.max_trajectory_depth > original.max_trajectory_depth
        else "DECLINED"
    )


def allowed_binding_ids(scenes: list[Any], dofs: list[Any]) -> dict[str, list[str]]:
    return {
        "target_violation_ids": sorted({
            scene.focus.primary_violation_id for scene in scenes
        }),
        "witness_ids": sorted({
            scene.focus.target_witness_id for scene in scenes
            if scene.focus.target_witness_id
        }),
        "participant_ids": sorted({
            participant.participant_id
            for scene in scenes for participant in scene.participants
        }),
        "dof_ids": sorted({dof.dof_id for dof in dofs}),
    }


def _registry_diagnostics(
    plan: KernelRepairPlan,
    registry: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not registry:
        return []
    scenes = registry.get("scenes", {})
    participants = registry.get("participants", {})
    dofs = registry.get("dofs", {})
    target_owners: dict[str, list[str]] = {}
    witness_owners: dict[str, list[str]] = {}
    relation_owners: dict[str, list[str]] = {}
    for scene_id, scene in scenes.items():
        target_owners.setdefault(str(scene.get("target_violation_id")), []).append(scene_id)
        witness = scene.get("target_witness_id")
        if witness:
            witness_owners.setdefault(str(witness), []).append(scene_id)
        for relation in scene.get("target_relation_ids", []) or []:
            relation_owners.setdefault(str(relation), []).append(scene_id)

    rows: list[dict[str, Any]] = []

    def add(identifier: str, kind: str, present: bool, owners: list[str]) -> None:
        rows.append({
            "identifier": identifier,
            "kind": kind,
            "present_in_sent_payload": present,
            "present_in_binding_registry": present,
            "target_ownership": sorted(set(owners)),
            "registry_id": registry.get("registry_id"),
        })

    for identifier in plan.target_violation_ids:
        add(identifier, "TARGET_VIOLATION", identifier in target_owners,
            target_owners.get(identifier, []))
    for identifier in plan.target_witness_ids:
        add(identifier, "TARGET_WITNESS", identifier in witness_owners,
            witness_owners.get(identifier, []))
    for identifier in plan.preferred_participant_ids:
        owner_scenes = (participants.get(identifier) or {}).get("scene_ids", [])
        add(identifier, "PARTICIPANT", identifier in participants, owner_scenes)
    for identifier in plan.preferred_dof_ids + plan.forbidden_dof_ids:
        owner = dofs.get(identifier) or {}
        add(identifier, "DOF", identifier in dofs,
            [owner["scene_id"]] if owner.get("scene_id") else [])
    if plan.target_relation != "AUTO":
        add(plan.target_relation, "RELATION", plan.target_relation in relation_owners,
            relation_owners.get(plan.target_relation, []))
    return rows


def _validate_registry_surface(
    registry: dict[str, Any] | None,
    scenes: list[Any],
    dofs: list[Any],
) -> list[str]:
    if registry is None:
        return []
    errors: list[str] = []
    if not registry.get("registry_id"):
        return ["PLAN_SEMANTIC_REGISTRY_ID_REQUIRED"]
    expected_scenes = {str(scene.scene_id) for scene in scenes}
    expected_dofs = {str(dof.dof_id) for dof in dofs}
    expected_participants = {
        str(participant.participant_id)
        for scene in scenes for participant in scene.participants
    }
    comparisons = (
        ("SCENE", expected_scenes, set(registry.get("scenes", {}))),
        ("DOF", expected_dofs, set(registry.get("dofs", {}))),
        ("PARTICIPANT", expected_participants, set(registry.get("participants", {}))),
    )
    for kind, expected, actual in comparisons:
        if expected != actual:
            errors.append(
                f"PLAN_SEMANTIC_REGISTRY_{kind}_SURFACE_MISMATCH:"
                f"missing={','.join(sorted(expected - actual))};"
                f"extra={','.join(sorted(actual - expected))}"
            )
    return errors


def bind_kernel_plan(
    plan: KernelRepairPlan,
    *,
    origin: PlanOrigin,
    supported_violation_ids: set[str],
    scenes: list[Any],
    dofs: list[Any],
    configured_max_depth: int,
    witness_relation_by_violation: dict[str, str] | None = None,
    registry: dict[str, Any] | None = None,
) -> tuple[BoundKernelPlan, PlanBindingReport]:
    errors: list[str] = _validate_registry_surface(registry, scenes, dofs)
    effects: list[str] = []
    scene_by_violation = {
        scene.focus.primary_violation_id: scene for scene in scenes
    }
    requested_targets = list(dict.fromkeys(plan.target_violation_ids))
    if not requested_targets:
        errors.append("TARGET_VIOLATION_REQUIRED")
    unknown_targets = sorted(
        set(requested_targets) - supported_violation_ids - set(scene_by_violation)
    )
    unavailable_targets = sorted(
        set(requested_targets) - set(scene_by_violation)
    )
    if unknown_targets:
        errors.append("TARGET_VIOLATION_NOT_CURRENT_OR_SUPPORTED:" + ",".join(unknown_targets))
    elif unavailable_targets:
        errors.append("TARGET_VIOLATION_SCENE_UNAVAILABLE:" + ",".join(unavailable_targets))
    selected_targets = [
        item for item in requested_targets
        if item in supported_violation_ids and item in scene_by_violation
    ]
    selected_scenes = [scene_by_violation[item] for item in selected_targets]
    if selected_targets:
        effects.append("TARGET_VIOLATION_HARD_FILTER")

    current_witnesses = {
        scene.focus.target_witness_id for scene in selected_scenes
        if scene.focus.target_witness_id
    }
    requested_witnesses = list(dict.fromkeys(plan.target_witness_ids))
    invalid_witnesses = sorted(set(requested_witnesses) - current_witnesses)
    if invalid_witnesses:
        errors.append("TARGET_WITNESS_MISMATCH:" + ",".join(invalid_witnesses))
    elif requested_witnesses:
        effects.append("WITNESS_VALIDATED")

    current_relations = {
        relation
        for scene in selected_scenes if scene.relation is not None
        for relation in (
            scene.relation.relation_kind,
            getattr(scene.relation, "constraint_kind", None),
        )
        if relation is not None
    }
    witness_relation_by_violation = witness_relation_by_violation or {}
    current_relations.update(
        witness_relation_by_violation[target]
        for target in selected_targets
        if target in witness_relation_by_violation
    )
    if plan.target_relation != "AUTO":
        if plan.target_relation not in current_relations:
            errors.append("TARGET_RELATION_MISMATCH:" + plan.target_relation)
        else:
            effects.append("RELATION_VALIDATED")

    participant_ids = {
        participant.participant_id
        for scene in selected_scenes for participant in scene.participants
    }
    invalid_participants = sorted(
        set(plan.preferred_participant_ids) - participant_ids
    )
    if invalid_participants:
        errors.append("PARTICIPANT_NOT_IN_SELECTED_SCENE:" + ",".join(invalid_participants))
    elif plan.preferred_participant_ids:
        effects.append("PARTICIPANT_ORDERING_ONLY")

    selected_scene_ids = {scene.scene_id for scene in selected_scenes}
    selected_dof_ids = {
        dof.dof_id for dof in dofs if dof.scene_id in selected_scene_ids
    }
    preferred = list(dict.fromkeys(plan.preferred_dof_ids))
    forbidden = list(dict.fromkeys(plan.forbidden_dof_ids))
    invalid_preferred = sorted(set(preferred) - selected_dof_ids)
    invalid_forbidden = sorted(set(forbidden) - selected_dof_ids)
    if invalid_preferred:
        errors.append("PREFERRED_DOF_NOT_IN_SELECTED_SCENE:" + ",".join(invalid_preferred))
    if invalid_forbidden:
        errors.append("FORBIDDEN_DOF_NOT_IN_SELECTED_SCENE:" + ",".join(invalid_forbidden))
    overlap = sorted(set(preferred) & set(forbidden))
    if overlap:
        errors.append("DOF_ALLOWED_FORBIDDEN_OVERLAP:" + ",".join(overlap))

    family_names = {
        str(getattr(scene.repair_family, "value", scene.repair_family))
        for scene in selected_scenes
    }
    hard_dof_family = bool(family_names) and family_names <= {"SPACING"}
    if hard_dof_family:
        allowed_dofs = set(preferred) if preferred else set(selected_dof_ids)
        allowed_dofs -= set(forbidden)
        if selected_dof_ids and not allowed_dofs:
            errors.append("DOF_ALLOWED_SET_EMPTY")
        if preferred or forbidden:
            effects.append("DOF_HARD_FILTER")
    else:
        allowed_dofs = set(selected_dof_ids) - set(forbidden)
        if preferred or forbidden:
            effects.append("DOF_ORDER_ONLY")
    if selected_dof_ids and not allowed_dofs and "DOF_ALLOWED_SET_EMPTY" not in errors:
        errors.append("DOF_ALLOWED_SET_EMPTY")

    effects.extend(["MAX_OPERATION_COUNT", "MAX_TRAJECTORY_DEPTH"])
    report = PlanBindingReport(
        plan_id=plan.plan_id,
        status="FAIL" if errors else "PASS",
        origin=origin,
        errors=errors,
        binding_effects=list(dict.fromkeys(effects)),
        registry_id=registry.get("registry_id") if registry else None,
        registry_schema_version=(
            registry.get("schema_version") if registry else None
        ),
        registry_snapshot_id=(registry.get("snapshot_id") if registry else None),
        identifier_diagnostics=_registry_diagnostics(plan, registry),
    )
    if errors:
        raise PlanBindingError(report)
    bound = BoundKernelPlan(
        plan_id=plan.plan_id,
        origin=origin,
        selected_violation_ids=selected_targets,
        selected_witness_ids=requested_witnesses,
        preferred_participant_ids=list(dict.fromkeys(plan.preferred_participant_ids)),
        preferred_dof_ids=preferred,
        allowed_dof_ids=sorted(allowed_dofs),
        forbidden_dof_ids=forbidden,
        max_operation_count=plan.max_operation_count,
        max_trajectory_depth=min(configured_max_depth, plan.max_trajectory_depth),
        strategy=plan.strategy,
        fallback_strategy=plan.fallback_strategy,
        coordination_request=plan.coordination_request,
        binding_effects=report.binding_effects,
    )
    return bound, report

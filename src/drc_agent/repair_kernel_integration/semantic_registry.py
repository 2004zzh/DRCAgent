from __future__ import annotations

from typing import Any

from pydantic import ConfigDict, Field

from drc_agent.schemas.common import StrictModel, stable_hash


class RegistryParticipant(StrictModel):
    """Current, source-backed meaning of one planner participant ID."""

    participant_id: str
    scene_ids: list[str] = Field(default_factory=list)
    roles: list[str] = Field(default_factory=list)
    geometry_kinds: list[str] = Field(default_factory=list)
    source_object_ids: list[str] = Field(default_factory=list)
    source_anchor_ids: list[str] = Field(default_factory=list)
    instance_anchor_ids: list[str] = Field(default_factory=list)
    physical_geometry_ids: list[str] = Field(default_factory=list)
    connectivity_component_ids: list[str] = Field(default_factory=list)
    editable: bool = False
    edit_authority: str | None = None
    protected_relation_ids: list[str] = Field(default_factory=list)


class RegistryDOF(StrictModel):
    """One selectable DOF and every compiler fact known before EDA."""

    dof_id: str
    scene_id: str
    participant_id: str
    dof_type: str
    axis: str | None = None
    edge: str | None = None
    expected_direction: str | None = None
    source_object_ids: list[str] = Field(default_factory=list)
    allowed_intervals_dbu: list[Any] = Field(default_factory=list)
    manufacturing_grid_dbu: int | None = None
    carrier_kinds: list[str] = Field(default_factory=list)
    atomic_co_dof_ids: list[str] = Field(default_factory=list)
    atomic_source_target_ids: list[str] = Field(default_factory=list)
    protected_relation_ids: list[str] = Field(default_factory=list)
    rejection_codes: list[str] = Field(default_factory=list)
    rejection_reasons: list[str] = Field(default_factory=list)
    executable_variants: list[dict[str, Any]] = Field(default_factory=list)
    compiler_status: str
    required_checks: list[str] = Field(default_factory=list)
    physical_meaning: str = "SYMBOLIC_ONLY_REQUIRES_FRESH_PHYSICAL_VALIDATION"


class RegistryScene(StrictModel):
    scene_id: str
    target_violation_id: str
    target_witness_id: str | None = None
    target_relation_ids: list[str] = Field(default_factory=list)
    participant_ids: list[str] = Field(default_factory=list)
    dof_ids: list[str] = Field(default_factory=list)
    repair_family: str


class PlanSemanticRegistry(StrictModel):
    """Immutable, snapshot-bound registry shared by a plan lifecycle.

    Planner IDs are constrained names, not an encoded complete solution. The
    registry records exact current meaning while leaving parameter selection
    and multi-DOF composition to the planner and deterministic search.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True, frozen=True)

    schema_version: str = "p5v1r2.plan-semantic-registry.v2"
    registry_id: str
    run_id: str
    iteration: int
    snapshot_id: str | None
    formal_context_hash: str
    current_context_fingerprint: str
    scenes: dict[str, RegistryScene]
    participants: dict[str, RegistryParticipant]
    dofs: dict[str, RegistryDOF]
    exact_relation_ids: list[str] = Field(default_factory=list)
    rejection_code_dictionary: list[str] = Field(default_factory=list)
    rejection_reason_dictionary: dict[str, list[str]] = Field(
        default_factory=dict,
    )
    protected_relation_dictionary: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
    )
    contract: str = (
        "Identifiers have only the exact current meaning recorded here; "
        "selection is symbolic and never asserts a physical verdict."
    )


def _dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return dict(value)


def _string(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))


def build_plan_semantic_registry(
    *,
    context: Any,
    scenes: list[Any],
    dofs: list[Any],
    current_context_fingerprint: str,
    run_id: str | None = None,
    iteration: int | None = None,
) -> PlanSemanticRegistry:
    """Build the single canonical registry from current typed scene objects."""

    resolved_run_id = getattr(context, "run_id", run_id)
    resolved_iteration = getattr(context, "iteration", iteration)
    if resolved_run_id is None or resolved_iteration is None:
        raise ValueError("PLAN_SEMANTIC_REGISTRY_RUN_IDENTITY_REQUIRED")

    capability = dict(getattr(context, "dof_capability_summary", {}) or {})
    constraint_dictionary = dict(getattr(
        context, "executable_constraint_dictionary", {},
    ) or {})
    rejection_reason_dictionary = {
        str(code): sorted({str(reason) for reason in reasons})
        for code, reasons in dict(getattr(
            context, "executable_rejection_reason_dictionary", {},
        ) or {}).items()
    }
    participants: dict[str, dict[str, Any]] = {}
    scene_rows: dict[str, RegistryScene] = {}
    dofs_by_scene: dict[str, list[str]] = {}
    exact_relations: set[str] = set()

    for dof in dofs:
        dofs_by_scene.setdefault(str(dof.scene_id), []).append(str(dof.dof_id))

    for scene in scenes:
        scene_raw = _dump(scene)
        scene_id = str(scene.scene_id)
        relation = scene_raw.get("relation") or {}
        relation_ids = sorted({
            str(item) for item in (
                relation.get("relation_kind"),
                relation.get("constraint_kind"),
            ) if item is not None
        })
        exact_relations.update(relation_ids)
        participant_ids: list[str] = []
        for participant in scene.participants:
            raw = _dump(participant)
            participant_id = str(participant.participant_id)
            participant_ids.append(participant_id)
            row = participants.setdefault(participant_id, {
                "participant_id": participant_id,
                "scene_ids": [], "roles": [], "geometry_kinds": [],
                "source_object_ids": [], "source_anchor_ids": [],
                "instance_anchor_ids": [], "physical_geometry_ids": [],
                "connectivity_component_ids": [], "editable": False,
                "edit_authority": None, "protected_relation_ids": [],
            })
            row["scene_ids"].append(scene_id)
            for field, output in (
                ("role", "roles"), ("geometry_kind", "geometry_kinds"),
            ):
                if raw.get(field) is not None:
                    row[output].append(_string(raw[field]))
            for field in (
                "source_object_ids", "source_anchor_ids", "instance_anchor_ids",
                "physical_geometry_ids", "connectivity_component_ids",
            ):
                row[field].extend(str(item) for item in raw.get(field, []) or [])
            row["editable"] = bool(row["editable"] or raw.get("editable", False))
            authority = _string(raw.get("edit_authority"))
            if authority is not None:
                if row["edit_authority"] not in (None, authority):
                    raise ValueError("PLAN_REGISTRY_PARTICIPANT_AUTHORITY_COLLISION")
                row["edit_authority"] = authority

        focus = scene_raw.get("focus") or {}
        scene_rows[scene_id] = RegistryScene(
            scene_id=scene_id,
            target_violation_id=str(focus["primary_violation_id"]),
            target_witness_id=(
                str(focus["target_witness_id"])
                if focus.get("target_witness_id") is not None else None
            ),
            target_relation_ids=relation_ids,
            participant_ids=sorted(set(participant_ids)),
            dof_ids=sorted(set(dofs_by_scene.get(scene_id, []))),
            repair_family=_string(scene_raw.get("repair_family")) or "UNKNOWN",
        )

    dof_rows: dict[str, RegistryDOF] = {}
    rejection_codes: set[str] = set()
    protected_by_participant: dict[str, set[str]] = {}
    for dof in dofs:
        raw = _dump(dof)
        dof_id = str(dof.dof_id)
        cap = dict(capability.get(dof_id, {}) or {})
        variants = [dict(item) for item in cap.get("variants", []) or []]
        variant_protections = {
            str(item) for variant in variants
            for item in variant.get("protected_relation_ids", []) or []
        }
        variant_atomic_dofs = {
            str(item) for variant in variants
            for item in variant.get("atomic_co_dof_ids", []) or []
        }
        variant_targets = {
            str(item) for variant in variants
            for item in variant.get("atomic_source_target_ids", []) or []
        }
        codes = sorted(str(item) for item in cap.get("rejection_codes", []) or [])
        reasons = sorted(str(item) for item in cap.get("rejection_reasons", []) or [])
        rejection_codes.update(codes)
        protected_by_participant.setdefault(str(dof.participant_id), set()).update(
            variant_protections
        )
        dof_rows[dof_id] = RegistryDOF(
            dof_id=dof_id,
            scene_id=str(dof.scene_id),
            participant_id=str(dof.participant_id),
            dof_type=_string(raw.get("dof_type")) or "UNKNOWN",
            axis=_string(raw.get("axis")),
            edge=_string(raw.get("edge")),
            expected_direction=_string(raw.get("expected_direction")),
            source_object_ids=sorted(str(item) for item in raw.get("source_object_ids", []) or []),
            allowed_intervals_dbu=list(raw.get("allowed_intervals", []) or []),
            manufacturing_grid_dbu=raw.get("manufacturing_grid_dbu"),
            carrier_kinds=sorted({
                str(variant["carrier_kind"]) for variant in variants
                if variant.get("carrier_kind") is not None
            }),
            atomic_co_dof_ids=sorted(variant_atomic_dofs),
            atomic_source_target_ids=sorted(variant_targets),
            protected_relation_ids=sorted(variant_protections),
            rejection_codes=codes,
            rejection_reasons=reasons,
            executable_variants=[{
                key: variant[key] for key in (
                    "carrier_kind", "atomic_co_dof_ids",
                    "atomic_source_target_ids", "operation_family",
                    "parameter_kind", "axis", "edge",
                    "legal_intervals_dbu", "grid_dbu",
                    "requires_occurrence_specialization",
                    "protected_relation_ids", "connectivity_risk",
                    "compiler_primitive", "compiler_reason_codes",
                ) if key in variant
            } for variant in sorted(
                variants,
                key=lambda item: stable_hash(item),
            )],
            compiler_status=str(cap.get("status", "CONDITIONAL_PENDING_COMPILER_PREFLIGHT")),
            required_checks=sorted(str(item) for item in cap.get("required_checks", []) or []),
        )

    participant_models: dict[str, RegistryParticipant] = {}
    for participant_id, row in participants.items():
        for field in (
            "scene_ids", "roles", "geometry_kinds", "source_object_ids",
            "source_anchor_ids", "instance_anchor_ids", "physical_geometry_ids",
            "connectivity_component_ids",
        ):
            row[field] = sorted(set(row[field]))
        row["protected_relation_ids"] = sorted(
            protected_by_participant.get(participant_id, set())
        )
        participant_models[participant_id] = RegistryParticipant(**row)

    identity = {
        "schema_version": "p5v1r2.plan-semantic-registry.v2",
        "run_id": str(resolved_run_id),
        "iteration": int(resolved_iteration),
        "snapshot_id": getattr(context, "snapshot_id", None),
        "formal_context_hash": str(getattr(
            context,
            "context_hash",
            stable_hash([current_context_fingerprint, "LEGACY_FORMAL_CONTEXT"]),
        )),
        "current_context_fingerprint": current_context_fingerprint,
        "scenes": {key: value.model_dump(mode="json") for key, value in sorted(scene_rows.items())},
        "participants": {key: value.model_dump(mode="json") for key, value in sorted(participant_models.items())},
        "dofs": {key: value.model_dump(mode="json") for key, value in sorted(dof_rows.items())},
        "exact_relation_ids": sorted(exact_relations),
        "rejection_code_dictionary": sorted(rejection_codes),
        "rejection_reason_dictionary": rejection_reason_dictionary,
        "protected_relation_dictionary": {
            key: value for key, value in sorted(
                constraint_dictionary.items()
            ) if value.get("constraint_class") in {
                "PROTECTED", "COUPLING", "LOCALITY",
            }
        },
    }
    return PlanSemanticRegistry(
        registry_id="plan_registry_" + stable_hash(identity)[:24],
        **identity,
    )


def install_plan_semantic_registry(context: Any, registry: PlanSemanticRegistry) -> None:
    """Install once; any later mismatch is an integrity failure."""

    value = registry.model_dump(mode="json")
    existing = getattr(context, "plan_semantic_registry", None)
    if existing:
        if existing != value:
            raise ValueError("PLAN_SEMANTIC_REGISTRY_IMMUTABILITY_VIOLATION")
        return
    context.plan_semantic_registry = value


def require_plan_semantic_registry(context: Any) -> dict[str, Any]:
    value = getattr(context, "plan_semantic_registry", None)
    if not isinstance(value, dict) or not value.get("registry_id"):
        raise ValueError("PLAN_SEMANTIC_REGISTRY_REQUIRED")
    return value

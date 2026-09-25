from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from drc_agent.llm import LLMClient, LLMCallContext
from drc_agent.agents.evodrc_skill import EvoDRCInitialSkillLibrary
from drc_agent.schemas.common import StrictModel, stable_hash
from .context import dof_capability_summary, planner_semantic_view
from .semantic_registry import (
    build_plan_semantic_registry,
    install_plan_semantic_registry,
    require_plan_semantic_registry,
)


# This is an input-size guard, not a scene-size assumption.  Membership and
# current-scene ownership remain request-specific PlanBinding checks.
MAX_SYMBOLIC_FORBIDDEN_DOF_IDS = 256


class KernelRepairPlan(StrictModel):
    """Symbolic plan emitted by the Region LLM.

    There are intentionally no coordinate, source-span, polygon, or tool
    result fields.  Geometry is selected and verified by deterministic code.
    """

    plan_id: str = ""
    target_violation_ids: list[str] = Field(default_factory=list, max_length=4)
    target_witness_ids: list[str] = Field(default_factory=list, max_length=4)
    target_relation: str = "AUTO"
    preferred_participant_ids: list[str] = Field(default_factory=list, max_length=8)
    preferred_dof_ids: list[str] = Field(default_factory=list, max_length=8)
    forbidden_dof_ids: list[str] = Field(
        default_factory=list, max_length=MAX_SYMBOLIC_FORBIDDEN_DOF_IDS,
    )
    strategy: str = "AUTO"
    fallback_strategy: str | None = None
    max_operation_count: int = Field(default=4, ge=1, le=4)
    max_trajectory_depth: int = Field(default=4, ge=1, le=4)
    rationale_evidence_ids: list[str] = Field(default_factory=list, max_length=16)
    coordination_request: str | None = None
    confidence_milli: int = Field(default=0, ge=0, le=1000)

    @model_validator(mode="after")
    def normalize_identity(self) -> "KernelRepairPlan":
        if not self.plan_id:
            object.__setattr__(
                self, "plan_id", "kernel_plan_" + stable_hash([
                    self.target_violation_ids,
                    self.target_witness_ids,
                    self.target_relation,
                    self.preferred_participant_ids,
                    self.preferred_dof_ids,
                    self.strategy,
                ])[:20],
            )
        return self


def _legacy_prompt_payload(context, scenes, dofs, support_reports, neighbor_messages, blueprint, failure_memory) -> dict[str, Any]:
    return {
        "hard_constraints": [
            "Choose only listed witness, participant, and DOF identifiers.",
            "Do not emit coordinates, polygons, Python, source spans, or tool results.",
            "Exact geometry and DRC/connectivity status are deterministic outputs.",
        ],
        "region": context.region.model_dump(mode="json"),
        "violations": [item.model_dump(mode="json") for item in context.region_violations],
        "support_reports": [item.model_dump(mode="json") for item in support_reports],
        "scenes": [item.model_dump(mode="json") for item in scenes],
        "dofs": [item.model_dump(mode="json") for item in dofs],
        "neighbor_messages": [item.model_dump(mode="json") for item in neighbor_messages],
        "repair_blueprint": blueprint.model_dump(mode="json") if blueprint else {},
        "failure_memory": failure_memory,
    }


def _without_empty(value):
    """Lossless for optional absence; retain False and zero physical values."""
    if isinstance(value, dict):
        return {key: _without_empty(item) for key, item in value.items()
                if key != "schema_version" and item is not None and item != [] and item != {}}
    if isinstance(value, list):
        return [_without_empty(item) for item in value]
    return value


def _dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return dict(value)
    raise TypeError(f"PROMPT_VALUE_NOT_SERIALIZABLE:{type(value).__name__}")


def _select_fields(value: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {key: value[key] for key in fields if key in value}


def _compact_region(context) -> dict[str, Any]:
    raw = _dump(context.region)
    # The planner needs policy, identity, and locality. Target-relevant source
    # IDs remain exact in participants/projections; broad Region-wide sets are
    # content-addressed inventory, not silently sliced lists.
    common = _select_fields(raw, (
        "region_id", "lineage_id", "status", "iteration", "bbox_dbu",
        "edit_halo_dbu", "layers", "violation_ids", "rule_ids",
        "rule_family_histogram", "net_mapping_quality", "resource_context",
        "timing_context", "quality_flags", "hard_dependencies",
    ))
    identity_fields = (
        "editable_object_ids", "frozen_object_ids", "segment_ids", "via_ids",
        "cell_instance_ids", "net_ids", "neighbor_edge_refs",
    )
    if not hasattr(context, "context_hash"):
        # Historical compact-prompt fixtures retain their exact legacy policy.
        common.update(_select_fields(raw, identity_fields))
    else:
        violation_ids = common.pop("violation_ids", [])
        common["region_violation_inventory"] = {
            "count": len(violation_ids),
            "content_hash": stable_hash(violation_ids),
            "selected_ids_are_in_target_admission": True,
            "omitted_targets_remain_pending": True,
        }
        inventory = {
            key: len(raw.get(key, []) or []) for key in identity_fields
            if raw.get(key, [])
        }
        common["region_identity_inventory"] = {
            "counts": inventory,
            "content_hash": stable_hash({
                key: raw.get(key, []) for key in identity_fields
            }),
        }
        resource = common.get("resource_context")
        if isinstance(resource, dict):
            resource_ids = {
                key: resource.get(key, []) for key in (
                    "occupied_track_bins", "free_track_bins", "via_site_bins",
                )
            }
            common["resource_context"] = _without_empty({
                **_select_fields(resource, (
                    "source", "local_whitespace_ratio", "local_density",
                    "congestion_score", "confidence",
                )),
                "bin_counts": {
                    key: len(value) for key, value in resource_ids.items()
                    if value
                },
                "bin_identity_hash": stable_hash(resource_ids),
            })
        common["full_region_context_ref"] = context.context_hash
    return _without_empty(common)


def _compact_violation(value: Any) -> dict[str, Any]:
    return _without_empty(_select_fields(_dump(value), (
        "violation_id", "rule_id", "rule_family", "description",
        "marker_type", "marker_bbox_dbu", "marker_geometry_dbu", "layers",
        "severity", "associated_object_ids", "association_quality",
        "fingerprint",
    )))


def _compact_participant(value: dict[str, Any]) -> dict[str, Any]:
    return _without_empty(_select_fields(value, (
        "role", "layer", "geometry_kind", "source_object_ids",
        "source_anchor_ids", "instance_anchor_ids", "physical_geometry_ids",
        "merged_component_id", "bbox", "editable", "edit_authority",
        "connectivity_component_ids", "connectivity_quality", "frozen_reason",
    )))


def _compact_projection(value: dict[str, Any]) -> dict[str, Any]:
    # Boundary contributors carry the exact source/edge ownership needed for
    # merged-boundary decisions. Duplicate top-level physical geometry lists
    # are retained because they close the physical-to-source mapping.
    return _without_empty(_select_fields(value, (
        "physical_geometry_ids", "source_object_ids", "source_anchor_ids",
        "instance_anchor_ids", "merged_component_id",
        "boundary_contributors", "quality", "unavailable_reason",
    )))


def _compact_scene(value: dict[str, Any]) -> dict[str, Any]:
    row = _select_fields(value, (
        "scene_id", "focus", "repair_family", "relation",
        "local_obstacle_ids", "edit_halo", "manufacturing_grid_dbu",
        "predicate_fidelity", "witness_fidelity", "hierarchy_quality",
        "connectivity_quality", "build_status", "failure_reason",
    ))
    focus = row.get("focus")
    if isinstance(focus, dict):
        row["focus"] = _select_fields(focus, (
            "focus_id", "primary_violation_id", "primary_rule_id",
            "repair_family", "coupled_violation_ids", "region_id",
            "local_bbox", "target_predicate_id", "target_witness_id",
        ))
    return _without_empty(row)


def _compact_neighbor_messages(
    messages,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    access_dictionary: dict[str, Any] = {}
    physical_evidence_dictionary: dict[str, Any] = {}
    identity_inventory_dictionary: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    access_identity_fields = (
        "writable_occurrence_ids", "writable_source_target_ids",
        "potential_write_geometry_ids", "protected_relation_ids",
        "protected_read_geometry_ids", "merged_boundary_contributor_ids",
        "landing_contact_relation_ids",
    )
    for message in messages:
        raw = _dump(message)
        constraints = []
        message_physical_refs: set[str] = set()
        message_access_refs: dict[str, str] = {}
        message_receiver_access_ref = None
        message_neighbor_access_ref = None
        message_neighbor_locality = None
        for constraint in raw.get("constraints", []):
            compact = _dump(constraint)
            parameters = dict(compact.get("parameters", {}))
            access = parameters.pop("potential_physical_access", {}) or {}
            refs = {}
            for endpoint, summary in sorted(access.items()):
                summary = _dump(summary)
                summary_id = str(summary.get("summary_id") or (
                    "access_" + stable_hash(summary)[:20]
                ))
                identity_counts = {
                    field: len(summary.get(field, []) or [])
                    for field in access_identity_fields
                    if summary.get(field, [])
                }
                normalized = _without_empty({
                    **_select_fields(summary, (
                        "potential_write_layers",
                        "connectivity_evidence_quality", "unknown_reasons",
                    )),
                    "identity_counts": identity_counts,
                })
                previous = access_dictionary.get(summary_id)
                if previous is not None and previous != normalized:
                    raise ValueError("PROMPT_ACCESS_SUMMARY_ID_COLLISION")
                access_dictionary[summary_id] = normalized
                refs[endpoint] = summary_id
                if summary.get("region_id") == raw.get("receiver_region_id"):
                    message_receiver_access_ref = summary_id
                elif summary.get("region_id") == raw.get("sender_region_id"):
                    message_neighbor_access_ref = summary_id
            if refs:
                if message_access_refs and message_access_refs != refs:
                    raise ValueError("PROMPT_MESSAGE_ACCESS_REF_COLLISION")
                message_access_refs = refs
                for endpoint, summary in sorted(access.items()):
                    if _dump(summary).get("region_id") == raw.get(
                        "sender_region_id"
                    ):
                        neighbor_halo = parameters.get(f"halo_{endpoint}_dbu")
                        if neighbor_halo is not None:
                            if (message_neighbor_locality is not None
                                    and message_neighbor_locality != neighbor_halo):
                                raise ValueError("PROMPT_NEIGHBOR_LOCALITY_COLLISION")
                            message_neighbor_locality = neighbor_halo
                        break
                parameters.pop("halo_u_dbu", None)
                parameters.pop("halo_v_dbu", None)

            physical_refs = []
            for evidence in parameters.pop(
                "physical_dependency_evidence", []
            ) or []:
                evidence = _dump(evidence)
                evidence_id = str(evidence["evidence_id"])
                value = _without_empty({
                    key: item for key, item in evidence.items()
                    if key != "evidence_id"
                })
                previous = physical_evidence_dictionary.get(evidence_id)
                if previous is not None and previous != value:
                    raise ValueError("PROMPT_PHYSICAL_EVIDENCE_ID_COLLISION")
                physical_evidence_dictionary[evidence_id] = value
                physical_refs.append(evidence_id)
                message_physical_refs.add(evidence_id)
            if physical_refs:
                parameters["physical_dependency_evidence_refs"] = physical_refs

            for field in (
                "shared_editable_object_ids", "shared_proxy_bin_ids",
            ):
                values = parameters.get(field, []) or []
                if values and (
                    physical_refs or not parameters.get("hard", False)
                ):
                    parameters.pop(field, None)
                    inventory = {
                        "count": len(values),
                        "content_hash": stable_hash(values),
                    }
                    inventory_id = (
                        "identity_inventory_"
                        + inventory["content_hash"][:16]
                    )
                    previous = identity_inventory_dictionary.get(inventory_id)
                    if previous is not None and previous != inventory:
                        raise ValueError("PROMPT_IDENTITY_INVENTORY_COLLISION")
                    identity_inventory_dictionary[inventory_id] = inventory
                    parameters[field + "_inventory_ref"] = inventory_id
            constraints.append(_without_empty({
                "relation": raw.get("relation"),
                "kind": compact.get("kind"),
                **parameters,
            }))
        requested_response = raw.get("requested_response", [])
        rows.append(_without_empty({
            "message_id": raw.get("message_id"),
            "round": raw.get("round"),
            "neighbor_region_id": raw.get("sender_region_id"),
            "relation": raw.get("relation"),
            "evidence_ids": [
                item for item in raw.get("evidence_ids", [])
                if item not in message_physical_refs
            ],
            "receiver_potential_access_ref": message_receiver_access_ref,
            "neighbor_potential_access_ref": message_neighbor_access_ref,
            # Legacy messages without endpoint Region identities keep their
            # orientation-preserving map; current messages use role-named refs.
            "potential_physical_access_refs": (
                {} if message_receiver_access_ref and message_neighbor_access_ref
                else message_access_refs
            ),
            "neighbor_locality_bbox": message_neighbor_locality,
            "constraints": constraints,
            "proposed_intents": raw.get("proposed_intents", []),
            "resource_claims": raw.get("resource_claims", []),
            "timing_claims": raw.get("timing_claims", []),
            "requested_response": (
                requested_response
                if requested_response != ["FACT_ACK"] else []
            ),
        }))
    # Multiple edge kinds may connect the same two Regions. Preserve every
    # relation and constraint, but emit one envelope per neighbor/round so the
    # shared access summary and locality are not repeated in paid context.
    grouped: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        key = (int(row.get("round", 0)), str(row.get("neighbor_region_id", "")))
        group = grouped.setdefault(key, {
            "round": row.get("round"),
            "neighbor_region_id": row.get("neighbor_region_id"),
            "relations": [], "evidence_ids": [],
            "constraints": [], "proposed_intents": [],
            "resource_claims": [], "timing_claims": [],
            "requested_response": [],
        })
        group["relations"].append(row.get("relation"))
        for name in (
            "evidence_ids", "constraints", "proposed_intents",
            "resource_claims", "timing_claims", "requested_response",
        ):
            group[name].extend(row.get(name, []) or [])
        for name in (
            "receiver_potential_access_ref", "neighbor_potential_access_ref",
            "potential_physical_access_refs", "neighbor_locality_bbox",
        ):
            value = row.get(name)
            if value is None:
                continue
            if name in group and group[name] != value:
                raise ValueError(f"PROMPT_NEIGHBOR_ENVELOPE_COLLISION:{name}")
            group[name] = value

    merged_rows = []
    constraint_dictionary: dict[str, Any] = {}
    for key in sorted(grouped):
        group = grouped[key]
        for name in (
            "relations", "evidence_ids", "constraints",
            "proposed_intents", "resource_claims", "timing_claims",
            "requested_response",
        ):
            seen: set[str] = set()
            unique = []
            for item in group[name]:
                identity = stable_hash(item)
                if identity in seen:
                    continue
                seen.add(identity)
                unique.append(item)
            group[name] = unique
        constraint_refs = []
        for constraint in group.pop("constraints"):
            constraint_id = (
                "neighbor_constraint_" + stable_hash(constraint)[:16]
            )
            previous = constraint_dictionary.get(constraint_id)
            if previous is not None and previous != constraint:
                raise ValueError("PROMPT_NEIGHBOR_CONSTRAINT_HASH_COLLISION")
            constraint_dictionary[constraint_id] = constraint
            constraint_refs.append(constraint_id)
        group["constraint_refs"] = constraint_refs
        merged_rows.append(_without_empty(group))

    # A receiver prompt normally has one current access summary. Hoist it once
    # rather than repeating it for every neighbor. Legacy disagreement remains
    # represented per envelope, so compaction never invents equivalence.
    receiver_refs = {
        row.get("receiver_potential_access_ref") for row in merged_rows
        if row.get("receiver_potential_access_ref")
    }
    common_receiver_ref = (
        next(iter(receiver_refs)) if len(receiver_refs) == 1 else None
    )
    if common_receiver_ref:
        for row in merged_rows:
            if row.get("receiver_potential_access_ref") == common_receiver_ref:
                row.pop("receiver_potential_access_ref", None)

    return merged_rows, access_dictionary, _without_empty({
        "receiver_potential_access_ref": common_receiver_ref,
        "constraints": constraint_dictionary,
        "physical_dependency_evidence": physical_evidence_dictionary,
        "identity_inventories": identity_inventory_dictionary,
    })

def _compact_failure_memory(failure_memory) -> dict[str, Any]:
    entries: dict[str, dict[str, Any]] = {}
    counts: dict[str, int] = {}
    for raw in failure_memory:
        value = _without_empty(_dump(raw))
        key = stable_hash(value)
        entries[key] = value
        counts[key] = counts.get(key, 0) + 1
    return _without_empty({
        "original_count": len(failure_memory),
        "unique_entries": [
            {"occurrence_count": counts[key], **entries[key]}
            for key in sorted(entries)
        ],
    })


def _planning_target_audit(context) -> dict[str, Any]:
    selected = [item.violation_id for item in context.region_violations]
    omitted = list(getattr(context, "omitted_region_violation_ids", []))
    by_id = {
        item.violation_id: item.rule_id
        for item in getattr(context, "violations", context.region_violations)
    }
    omitted_rules: dict[str, int] = {}
    for violation_id in omitted:
        rule_id = by_id.get(violation_id, "UNKNOWN")
        omitted_rules[rule_id] = omitted_rules.get(rule_id, 0) + 1
    return _without_empty({
        "selected_violation_ids": selected,
        "omitted_violation_count": len(omitted),
        "omitted_rule_histogram": omitted_rules,
        "policy": (
            "EXPLICIT_BOUNDED_TARGET_SUBSET"
            if getattr(context, "planning_target_violation_ids", None) is not None
            else "HISTORICAL_ALL_REGION_TARGETS"
        ),
        "omitted_targets_remain_pending": True,
        "hard_dependency_scope_preserved": True,
    })


def _prompt_payload(
    context, scenes, dofs, support_reports, neighbor_messages, blueprint,
    failure_memory, evodrc_initial_skill: dict[str, Any] | None = None,
) -> dict[str, Any]:
    legacy = _legacy_prompt_payload(
        context, scenes, dofs, support_reports, neighbor_messages,
        blueprint, failure_memory,
    )
    raw_blueprint = blueprint.model_dump(mode="json") if blueprint else {}
    if raw_blueprint.get("generated_by") == "EMPTY":
        blueprint_payload = _without_empty(_select_fields(raw_blueprint, (
            "blueprint_id", "subgraph_id", "planning_scope_id",
            "planning_scope_type", "graph_version", "generated_by",
            "validation_status",
        )))
    else:
        blueprint_payload = _without_empty(raw_blueprint)
    if blueprint is not None:
        from drc_agent.experience.lessons import bind_lessons, bounded_lesson_payload
        # Historical roles and coordinates are never executable current IDs.
        blueprint_payload.pop("symbolic_lessons", None)
        lessons, audit = bounded_lesson_payload(
            bind_lessons(blueprint.symbolic_lessons, scenes, dofs)
        )
        blueprint_payload["current_symbolic_lessons"] = lessons
        blueprint_payload["lesson_budget"] = audit

    participants: dict[str, Any] = {}
    projections: dict[str, Any] = {}
    scene_rows: list[dict[str, Any]] = []
    for scene_value in legacy["scenes"]:
        scene = _dump(scene_value)
        row = _compact_scene(scene)
        row["participant_ids"] = []
        row["projection_ids"] = []
        for participant in scene.get("participants", []):
            participant = _dump(participant)
            key = participant["participant_id"]
            value = _compact_participant(participant)
            if key in participants and participants[key] != value:
                raise ValueError("PROMPT_PARTICIPANT_ID_COLLISION")
            participants[key] = value
            row["participant_ids"].append(key)
        for projection in scene.get("hierarchy_projections", []):
            projection = _dump(projection)
            key = projection["projection_id"]
            value = _compact_projection(projection)
            if key in projections and projections[key] != value:
                raise ValueError("PROMPT_PROJECTION_ID_COLLISION")
            projections[key] = value
            row["projection_ids"].append(key)
        scene_rows.append(_without_empty(row))

    dof_rows: dict[str, Any] = {}
    dof_templates: dict[str, Any] = {}
    dof_bindings: dict[str, Any] = {}
    capability = dof_capability_summary(context)
    domain_fields = {
        "domain_min_dbu", "domain_max_dbu", "allowed_intervals",
    }
    binding_fields = {
        "scene_id", "participant_id", "source_object_ids", "locality_bbox",
        "coupling_group_id", "conflicts_with_dof_ids",
    }
    for dof_value in legacy["dofs"]:
        dof = _dump(dof_value)
        key = dof["dof_id"]
        row = {name: value for name, value in dof.items() if name != "dof_id"}
        domain = {
            name: row.pop(name) for name in tuple(row)
            if name in domain_fields
        }
        binding = _without_empty({
            name: row.pop(name) for name in tuple(row)
            if name in binding_fields
        })
        binding_id = "dof_binding_" + stable_hash(binding)[:16]
        if binding_id in dof_bindings and dof_bindings[binding_id] != binding:
            raise ValueError("PROMPT_DOF_BINDING_HASH_COLLISION")
        dof_bindings[binding_id] = binding
        capability_row = dict(capability.get(key, {
            "status": "CONDITIONAL_PENDING_COMPILER_PREFLIGHT",
            "predicate_solved": False,
            "physical_verdict": "NOT_EVALUATED",
        }))
        # The executable variable ID is deterministic implementation detail;
        # LLM selection is intentionally restricted to the raw current DOF ID.
        capability_row.pop("executable_variable_id", None)
        row["source_executability"] = _select_fields(capability_row, (
            "status", "primitive", "failure_code", "variants",
            "rejection_codes", "required_checks", "predicate_solved",
            "physical_verdict",
        ))
        template = _without_empty(row)
        template_id = "dof_semantics_" + stable_hash(template)[:16]
        if template_id in dof_templates and dof_templates[template_id] != template:
            raise ValueError("PROMPT_DOF_TEMPLATE_HASH_COLLISION")
        dof_templates[template_id] = template
        value = _without_empty({
            "semantics_ref": template_id,
            "binding_ref": binding_id,
            **domain,
        })
        if key in dof_rows and dof_rows[key] != value:
            raise ValueError("PROMPT_DOF_ID_COLLISION")
        dof_rows[key] = value

    (
        message_rows, access_dictionary, neighbor_fact_dictionary,
    ) = _compact_neighbor_messages(neighbor_messages)
    registry: dict[str, Any] = {}
    if hasattr(context, "plan_semantic_registry"):
        if not context.plan_semantic_registry:
            built = build_plan_semantic_registry(
                context=context,
                scenes=scenes,
                dofs=dofs,
                current_context_fingerprint=context.context_hash,
            )
            install_plan_semantic_registry(context, built)
        registry = require_plan_semantic_registry(context)
    payload = _without_empty({
        "hard_constraints": legacy["hard_constraints"],
        "target_admission": _planning_target_audit(context),
        "region": _compact_region(context),
        "violations": [
            _compact_violation(item) for item in context.region_violations
        ],
        "support_reports": [
            _without_empty(_dump(item)) for item in support_reports
        ],
        "scenes": scene_rows,
        "dofs": dof_rows,
        "neighbor_messages": message_rows,
        "repair_blueprint": blueprint_payload,
        "evodrc_initial_skill": evodrc_initial_skill or {},
        "failure_memory": _compact_failure_memory(failure_memory),
        "plan_semantic_registry": registry,
        "current_id_dictionary": {
            "participants": participants,
            "projections": projections,
            "dof_semantics": dof_templates,
            "dof_bindings": dof_bindings,
            "potential_physical_access": access_dictionary,
            "neighbor_facts": neighbor_fact_dictionary,
        },
        "dictionary_contract": (
            "Resolve refs through current_id_dictionary. Only listed IDs are "
            "selectable; source preflight is not physical truth."
        ),
        "payload_version": "p5-formal-registry-evodrc-skill-v1",
    })
    # The generic compactor omits historical nested schema_version fields.
    # Restore the canonical registry verbatim so its hashable schema identity
    # is identical in prompt, follow-ups, binder and root registration.
    if registry:
        payload["plan_semantic_registry"] = registry
    return payload

def validate_lesson_citations(plan, semantic_view):
    lessons=semantic_view.get("repair_blueprint",{}).get("current_symbolic_lessons",[])
    allowed={lesson["lesson_id"] for lesson in lessons}
    cited={identifier for identifier in plan.rationale_evidence_ids
           if identifier.startswith("lesson_")}
    if not cited<=allowed:
        raise ValueError("UNKNOWN_OR_UNBOUND_SYMBOLIC_LESSON_CITATION")
    return sorted(cited)


def symbolic_intent_view(plan, scenes, dofs):
    """Current physical meaning, not guessed coordinates or a DRC verdict."""
    participants = {p.participant_id:p for scene in (scenes or []) for p in scene.participants}
    selected = [d for d in (dofs or []) if d.dof_id in set(plan.preferred_dof_ids)]
    return {"status":"SYMBOLIC_NOT_VERIFIED", "selection_explicit":bool(plan.preferred_dof_ids),
        "dofs":[{"dof_id":d.dof_id,"type":d.dof_type,"axis":d.axis,"edge":d.edge,
            "direction":d.expected_direction,"legal_intervals_dbu":d.allowed_intervals,
            "role":getattr(participants.get(d.participant_id),"role",None),
            "participant_id":d.participant_id,"source_object_ids":d.source_object_ids,
            "occurrences":getattr(participants.get(d.participant_id),"instance_anchor_ids",[]),
            "grid_dbu":d.manufacturing_grid_dbu,"locality":d.locality_bbox.model_dump(mode="json"),
            "risk_flags":d.risk_flags,"declared_predicate_effect_keys":d.satisfies_relation_keys,
            "connectivity_quality":d.connectivity_quality} for d in selected],
        "unknown":["exact landing point", "fresh DRC", "fresh connectivity"]}


def _followup_semantic_view(context) -> dict[str, Any]:
    """Retain selectable current semantics without repeating round-one bulk."""

    raw = planner_semantic_view(context)
    dictionary = dict(raw.get("current_id_dictionary") or {})
    current_ids = _select_fields(dictionary, (
        "participants", "dof_semantics", "dof_bindings",
    ))
    # Projections are not selectable planner IDs. Their complete current
    # mapping remains immutable in the primary payload bound below. Repeat the
    # participant/source restrictions and DOF bindings needed for a follow-up.
    scenes = []
    for raw_scene in raw.get("scenes", []):
        scene = dict(raw_scene)
        scene.pop("projection_ids", None)
        scenes.append(scene)
    return _without_empty({
        "primary_prompt_hash": stable_hash(raw),
        **_select_fields(raw, (
            "hard_constraints", "target_admission", "violations",
            "support_reports", "dofs", "repair_blueprint",
            "evodrc_initial_skill",
            "plan_semantic_registry", "dictionary_contract", "payload_version",
        )),
        "scenes": scenes,
        "current_id_dictionary": current_ids,
        "followup_omissions": (
            "region inventory, hierarchy projections, and round-one messages "
            "remain bound by primary_prompt_hash; current participant/source "
            "restrictions and DOF bindings are repeated; finalize carries "
            "fresh round-two facts"
        ),
    })


def _allowed_identifier_payload(
    allowed_ids: dict[str, list[str]],
) -> dict[str, list[str]]:
    return {
        "allowed_target_violation_ids": allowed_ids["target_violation_ids"],
        "allowed_witness_ids": allowed_ids["witness_ids"],
        "allowed_participant_ids": allowed_ids["participant_ids"],
        "allowed_dof_ids": allowed_ids["dof_ids"],
    }


def revision_prompt_payload(
    context, original_plan: KernelRepairPlan, binding_errors: list[str],
    allowed_ids: dict[str, list[str]],
) -> dict[str, Any]:
    return {
        "current_semantic_view": _followup_semantic_view(context),
        "original_plan": original_plan.model_dump(mode="json"),
        "binding_errors": binding_errors,
        **_allowed_identifier_payload(allowed_ids),
    }


def execution_revision_prompt_payload(
    context, original_plan: KernelRepairPlan,
    execution_feedback: dict[str, Any],
    allowed_ids: dict[str, list[str]],
) -> dict[str, Any]:
    return {
        "current_semantic_view": _followup_semantic_view(context),
        "original_plan": original_plan.model_dump(mode="json"),
        "execution_feedback": execution_feedback,
        **_allowed_identifier_payload(allowed_ids),
    }


def finalize_prompt_payload(
    context, provisional_plan: KernelRepairPlan, round_two_messages,
    allowed_ids: dict[str, list[str]],
) -> dict[str, Any]:
    (
        round_two_rows, round_two_access, round_two_facts,
    ) = _compact_neighbor_messages(round_two_messages)
    return {
        "current_semantic_view": _followup_semantic_view(context),
        "provisional_plan": provisional_plan.model_dump(mode="json"),
        "round_two_neighbor_messages": round_two_rows,
        "round_two_id_dictionary": {
            "potential_physical_access": round_two_access,
            "neighbor_facts": round_two_facts,
        },
        **_allowed_identifier_payload(allowed_ids),
    }


class SymbolicKernelPlanner:
    def __init__(
        self, llm: LLMClient | None, *,
        prompt_version: str = "p5-formal-registry-v1", event_sink=None,
        project_root: Path | None = None,
        evodrc_skill_enabled: bool = False,
        evodrc_skill_root: Path = Path(
            "benchmarks/EvoDRC/agent/knowledge/cla"
        ),
    ):
        self.llm = llm
        self.prompt_version = prompt_version
        self.event_sink = event_sink
        if evodrc_skill_enabled and project_root is None:
            raise ValueError("EVODRC_INITIAL_SKILL_PROJECT_ROOT_REQUIRED")
        self.evodrc_skill = EvoDRCInitialSkillLibrary(
            project_root=project_root or Path.cwd(),
            enabled=evodrc_skill_enabled,
            skill_root=evodrc_skill_root,
        )

    def _event(self, name: str, **details: Any) -> None:
        if self.event_sink is not None:
            self.event_sink(name, **details)

    @staticmethod
    def deterministic_default(context, scenes, dofs) -> KernelRepairPlan:
        scene = scenes[0] if scenes else None
        scene_dofs = [item for item in dofs if item.scene_id == getattr(scene, "scene_id", "")]
        violation_ids = [item.violation_id for item in context.region_violations]
        witness_ids = [
            str((context.design.rule_witnesses.get(item.violation_id) or {}).get("witness_id"))
            for item in context.region_violations
            if (context.design.rule_witnesses.get(item.violation_id) or {}).get("witness_id")
        ]
        return KernelRepairPlan(
            target_violation_ids=violation_ids[:4],
            target_witness_ids=witness_ids[:4],
            target_relation=(scene.relation.relation_kind if scene and scene.relation else "AUTO"),
            preferred_participant_ids=[item.participant_id for item in scene.participants[:2]] if scene else [],
            preferred_dof_ids=[item.dof_id for item in scene_dofs[:4]],
            strategy="DETERMINISTIC_DEFAULT",
            rationale_evidence_ids=violation_ids[:4],
            confidence_milli=500,
        )

    async def plan(self, *, context, scenes, dofs, support_reports, neighbor_messages, blueprint, failure_memory) -> KernelRepairPlan:
        if self.llm is None:
            return self.deterministic_default(context, scenes, dofs)
        skill_payload = self.evodrc_skill.prompt_payload(
            context=context, scenes=scenes,
        )
        payload = _prompt_payload(
            context, scenes, dofs, support_reports, neighbor_messages,
            blueprint, failure_memory, skill_payload,
        )
        semantic_view = planner_semantic_view(context)
        semantic_view.clear()
        semantic_view.update(payload)
        old = _legacy_prompt_payload(context, scenes, dofs, support_reports, neighbor_messages, blueprint, failure_memory)
        compact_json = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        neighbor_json = json.dumps({
            "messages": payload.get("neighbor_messages", []),
            "access": payload.get("current_id_dictionary", {}).get(
                "potential_physical_access", {}
            ),
        }, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        target_admission = payload.get("target_admission", {})
        exposed_lessons = payload.get("repair_blueprint", {}).get(
            "current_symbolic_lessons", []
        )
        exposed_skill = payload.get("evodrc_initial_skill", {})
        self._event("repair_kernel_prompt_audit", region_id=context.region.region_id,
            payload_hash=stable_hash(payload), payload_version=payload["payload_version"],
            plan_semantic_registry_id=(
                payload.get("plan_semantic_registry", {}).get("registry_id")
            ),
            exposed_lesson_ids=sorted(item["lesson_id"] for item in exposed_lessons),
            evodrc_skill_enabled=bool(exposed_skill),
            evodrc_skill_set_hash=exposed_skill.get("skill_set_hash"),
            evodrc_skill_layers=exposed_skill.get("selected_layers", []),
            evodrc_skill_utf8_bytes=sum(
                len(item.get("content", "").encode("utf-8"))
                for item in exposed_skill.get("documents", [])
            ),
            current_role_binding_hash=stable_hash(exposed_lessons),
            formal_prompt_payload_hash=stable_hash(payload),
            compact_utf8_bytes=len(compact_json.encode()),
            neighbor_utf8_bytes=len(neighbor_json.encode()),
            selected_target_violation_ids=target_admission.get(
                "selected_violation_ids", []
            ),
            omitted_target_violation_count=target_admission.get(
                "omitted_violation_count", 0
            ),
            hard_dependency_scope_preserved=target_admission.get(
                "hard_dependency_scope_preserved", False
            ),
            legacy_utf8_bytes=len(json.dumps(old,sort_keys=True,ensure_ascii=True).encode()),
            tokenizer_status="NOT_CALIBRATED_USE_PROVIDER_USAGE", truncated_hard_constraints=False)
        self._event(
            "repair_kernel_plan_requested",
            region_id=context.region.region_id,
            target_violation_ids=target_admission.get(
                "selected_violation_ids", []
            ),
        )
        value = await self.llm.generate_structured(
            system_prompt=(
                "You are the symbolic planner for a verified DRC repair kernel. "
                "Select relations, participants, and DOFs only from the exact "
                "snapshot-bound plan_semantic_registry. IDs constrain choices "
                "but do not encode a complete solution. Respect each current "
                "executable variant's carrier_kind, atomic_co_dof_ids, legal "
                "intervals, protected_relation_dictionary, and recorded "
                "rejection reasons; never invent a current ID. The "
                "evodrc_initial_skill is advisory strategy only: current "
                "RuleWitness and executable semantics override it, and its "
                "historical coordinates or patches are never current evidence. "
                "Return JSON for "
                "KernelRepairPlan. If a current_symbolic_lesson informs your decision, cite its exact lesson_id "
                "in rationale_evidence_ids; its historical outcome is not a current tool verdict. Return "
                "KernelRepairPlan; never emit geometry, source code, coordinates, "
                "or claims about DRC/connectivity."
            ),
            user_prompt=json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")),
            response_model=KernelRepairPlan,
            context=LLMCallContext(
                run_id=context.run_id,
                iteration=context.iteration,
                subgraph_id=context.subgraph_id,
                region_id=context.region.region_id,
                purpose="repair_agent_step",
                prompt_version=self.prompt_version,
            ),
        )
        result = value if isinstance(value, KernelRepairPlan) else KernelRepairPlan.model_validate(value)
        validate_lesson_citations(result, semantic_view)
        self._event("repair_kernel_plan_received", region_id=context.region.region_id,
            plan_id=result.plan_id, rationale_evidence_ids=result.rationale_evidence_ids,
            preferred_dof_ids=result.preferred_dof_ids)
        return result

    async def revise(
        self, *, context, original_plan: KernelRepairPlan,
        binding_errors: list[str], allowed_ids: dict[str, list[str]],
    ) -> KernelRepairPlan:
        if self.llm is None:
            raise RuntimeError("LLM_PLAN_REVISION_REQUIRES_LLM")
        payload = revision_prompt_payload(
            context, original_plan, binding_errors, allowed_ids,
        )
        self._event(
            "repair_kernel_plan_revision_requested",
            region_id=context.region.region_id, plan_id=original_plan.plan_id,
            target_violation_ids=allowed_ids["target_violation_ids"],
        )
        value = await self.llm.generate_structured(
            system_prompt=(
                "Revise one KernelRepairPlan using only the supplied allowed "
                "identifiers. Fix every binding error. Return JSON only; do "
                "not emit geometry, source code, coordinates, or tool claims."
            ),
            user_prompt=json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")),
            response_model=KernelRepairPlan,
            context=LLMCallContext(
                run_id=context.run_id, iteration=context.iteration,
                subgraph_id=context.subgraph_id,
                region_id=context.region.region_id,
                purpose="repair_agent_revision_step",
                prompt_version=self.prompt_version,
            ),
        )
        result = (
            value if isinstance(value, KernelRepairPlan)
            else KernelRepairPlan.model_validate(value)
        )
        validate_lesson_citations(result, planner_semantic_view(context))
        self._event(
            "repair_kernel_plan_revision_received",
            region_id=context.region.region_id, plan_id=result.plan_id,
        )
        return result

    async def revise_after_execution_feedback(
        self,
        *,
        context,
        original_plan: KernelRepairPlan,
        execution_feedback: dict[str, Any],
        allowed_ids: dict[str, list[str]],
    ) -> KernelRepairPlan:
        if self.llm is None:
            raise RuntimeError("LLM_EXECUTION_REVISION_REQUIRES_LLM")
        payload = execution_revision_prompt_payload(
            context, original_plan, execution_feedback, allowed_ids,
        )
        self._event(
            "repair_kernel_execution_revision_requested",
            region_id=context.region.region_id,
            original_plan_id=original_plan.plan_id,
            feedback_kind=execution_feedback["kind"],
            target_violation_ids=allowed_ids["target_violation_ids"],
        )
        value = await self.llm.generate_structured(
            system_prompt=(
                "You are revising an already-bound symbolic repair plan after "
                "deterministic execution feedback. Do not invent geometry, "
                "coordinates, source code, tool results, witness IDs, participant "
                "IDs, or DOF IDs. For PREFERRED_DOF_INFEASIBLE, select exactly "
                "one proved feasible_alternative_dof_set (or clear preferences) "
                "and remove from forbidden_dof_ids only IDs contained in the "
                "proved alternatives. Increase max_operation_count only when "
                "the selected alternative has a supplied minimum bound, and "
                "then use exactly that proved minimum; the deterministic "
                "validator enforces this. "
                "For every other feedback kind forbidden_dof_ids are immutable "
                "hard constraints. For DEPTH_EXTENSION_AVAILABLE, either "
                "keep max_trajectory_depth unchanged to decline, or increase it "
                "within the supplied maximum. For PHYSICAL_CONSTRAINT_FAILURE, "
                "preserve target_violation_ids, target_witness_ids, "
                "target_relation, forbidden_dof_ids, and max_trajectory_depth; "
                "make a novel symbolic choice by changing strategy, fallback, "
                "preferred participant/DOF ordering, coordination, or operation "
                "count using only the snapshot-bound registry. Do not repeat an "
                "attempted physical effect. Preserve every other execution "
                "semantic field. If you change the authorized execution choice, "
                "emit a new plan_id distinct from original_plan.plan_id. Return "
                "KernelRepairPlan JSON only."
            ),
            user_prompt=json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")),
            response_model=KernelRepairPlan,
            context=LLMCallContext(
                run_id=context.run_id,
                iteration=context.iteration,
                subgraph_id=context.subgraph_id,
                region_id=context.region.region_id,
                purpose="repair_agent_execution_feedback_step",
                prompt_version=self.prompt_version,
            ),
        )
        result = (
            value if isinstance(value, KernelRepairPlan)
            else KernelRepairPlan.model_validate(value)
        )
        validate_lesson_citations(result, planner_semantic_view(context))
        self._event(
            "repair_kernel_execution_revision_received",
            region_id=context.region.region_id,
            original_plan_id=original_plan.plan_id,
            revised_plan_id=result.plan_id,
            feedback_kind=execution_feedback["kind"],
        )
        return result

    async def finalize(
        self, *, context, provisional_plan: KernelRepairPlan,
        round_two_messages, allowed_ids: dict[str, list[str]],
    ) -> KernelRepairPlan:
        if not round_two_messages:
            return provisional_plan
        if self.llm is None:
            return provisional_plan
        payload = finalize_prompt_payload(
            context, provisional_plan, round_two_messages, allowed_ids,
        )
        self._event(
            "repair_kernel_final_plan_requested",
            region_id=context.region.region_id,
            provisional_plan_id=provisional_plan.plan_id,
            target_violation_ids=allowed_ids["target_violation_ids"],
        )
        value = await self.llm.generate_structured(
            system_prompt=(
                "Finalize the provisional KernelRepairPlan after one bounded "
                "neighbor-message round. Use only supplied identifiers and "
                "return KernelRepairPlan JSON without geometry or source code."
            ),
            user_prompt=json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")),
            response_model=KernelRepairPlan,
            context=LLMCallContext(
                run_id=context.run_id, iteration=context.iteration,
                subgraph_id=context.subgraph_id,
                region_id=context.region.region_id,
                purpose="repair_agent_revision_step",
                prompt_version=self.prompt_version,
            ),
        )
        result = (
            value if isinstance(value, KernelRepairPlan)
            else KernelRepairPlan.model_validate(value)
        )
        validate_lesson_citations(result, planner_semantic_view(context))
        self._event(
            "repair_kernel_final_plan_received",
            region_id=context.region.region_id,
            provisional_plan_id=provisional_plan.plan_id,
            final_plan_id=result.plan_id,
            changed=(
                result.model_dump(mode="json")
                != provisional_plan.model_dump(mode="json")
            ),
        )
        return result

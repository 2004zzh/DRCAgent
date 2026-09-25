from __future__ import annotations

from collections import Counter
from pathlib import Path

from pydantic import Field

from drc_agent.schemas.common import Box, StrictModel, canonical_json
from drc_agent.schemas.experience import RepairBlueprint
from drc_agent.schemas.state import (
    CandidateAttemptSummary, LayoutObject, NeighborMessage, RegionState,
    RollbackSummary, RuleCatalog, ViolationRecord,
)


class ContextObject(StrictModel):
    object_id: str
    routing_type: str | None = None
    source_anchor_id: str | None
    kind: str
    layer: str
    editability: str
    bbox_dbu: Box
    bbox_relative_dbu: Box
    parent_bbox_dbu: Box | None = None
    local_fragment_ids: list[str] = Field(default_factory=list)
    geometry_dbu: list[dict] | None
    geometry_relative_dbu: list[dict] | None
    net_id: str | None
    net_mapping_quality: str = "unavailable"
    connectivity_component_id: str | None = None
    source_slice: str | None = None


class CompactRegionContext(StrictModel):
    task_and_hard_constraints: list[str]
    region_identity_and_iteration: dict
    violation_summary_by_rule: dict[str, int]
    violations: list[dict]
    editable_objects_with_source_anchors: list[ContextObject]
    frozen_boundary_context_objects: list[ContextObject]
    local_route_fragments: list[dict] = Field(default_factory=list)
    local_route_endpoints: list[dict] = Field(default_factory=list)
    via_landings: list[dict] = Field(default_factory=list)
    via_metal_adjacencies: list[dict] = Field(default_factory=list)
    local_geometry_origin_dbu: dict[str, int]
    neighbor_messages: list[NeighborMessage]
    failure_history: list[CandidateAttemptSummary]
    rollback_history: list[RollbackSummary]
    repair_blueprint: RepairBlueprint
    allowed_layout_edit_ir: list[str]
    allowed_distance_candidates_dbu: list[int]
    available_segment_ids: list[str] = Field(default_factory=list)
    available_via_ids: list[str] = Field(default_factory=list)
    available_net_ids: list[str] = Field(default_factory=list)
    legal_routing_layers: list[str] = Field(default_factory=list)
    rule_constraints: dict[str, dict] = Field(default_factory=dict)
    rule_predicates: dict[str, dict] = Field(default_factory=dict)
    rule_witnesses: list[dict] = Field(default_factory=list)
    rule_knowledge_packs: dict[str, dict] = Field(default_factory=dict)
    geometry_distance_hints_dbu: dict[str, list[int]] = Field(default_factory=dict)
    rule_distance_constraints_dbu: dict[str, int] = Field(default_factory=dict)
    max_candidate_distance_dbu: int
    object_to_violation_ids: dict[str, list[str]] = Field(default_factory=dict)
    required_structured_output_schema: dict
    available_evidence_ids: list[str] = Field(default_factory=list)
    context_budget: dict = Field(default_factory=dict)
    omitted_context_summary: dict[str, int] = Field(default_factory=dict)

    def to_prompt_json(self) -> str:
        excluded = (
            {"repair_blueprint"}
            if self.repair_blueprint.generated_by == "EMPTY" else set()
        )
        return canonical_json(self.model_dump(mode="json", exclude=excluded))


class CompactRegionContextSerializer:
    HARD_CONSTRAINTS = [
        "Preserve connectivity and never mutate frozen Class D objects.",
        "Reference only listed object IDs, violation IDs, source anchors, and evidence IDs.",
        "Propose intent JSON only; do not emit Python, shell commands, diffs, or exact source spans.",
        "Coordinates used for execution remain integer DBU; relative coordinates are reasoning aids.",
        "Every proposed edit must pass deterministic lowering, checking, joint planning, and EDA verification.",
    ]

    def __init__(
        self, max_context_tokens: int = 6000,
        *, allowed_distances_dbu: list[int] | None = None,
        max_candidate_distance_dbu: int = 512,
        dbu_per_um: int = 4000,
        manufacturing_grid_dbu: int = 4,
    ):
        if max_context_tokens < 1800:
            raise ValueError("max_context_tokens must be at least 1800")
        self.max_context_tokens = max_context_tokens
        self.allowed_distances_dbu = sorted(set(
            allowed_distances_dbu or [8, 16, 24, 32, 48, 64]
        ))
        self.max_candidate_distance_dbu = max_candidate_distance_dbu
        self.dbu_per_um = dbu_per_um
        self.manufacturing_grid_dbu = manufacturing_grid_dbu

    @staticmethod
    def _estimate_tokens(value: CompactRegionContext) -> int:
        length = len(value.to_prompt_json())
        return max(1, (length * 2 + 4) // 5)

    @staticmethod
    def _compact_witness(value: dict) -> dict:
        """Keep rule physics while removing duplicated polygon/edge payloads."""
        keys = (
            "schema_version", "witness_id", "violation_id", "rule_id",
            "predicate_id", "offending_relation",
            "participating_physical_geometry_ids", "current_measurements",
            "required_relation", "source_contributor_ids",
            "editable_contributor_ids", "editable_source_object_ids",
            "connectivity_evidence", "evidence_quality",
            "predicate_fidelity", "local_bbox_dbu",
            "predicate_satisfied", "unresolved_reasons", "analyzer_id",
        )
        result = {key: value[key] for key in keys if key in value}
        geometries = value.get("physical_geometries") or []
        result["physical_contributor_summaries"] = [
            {
                key: geometry[key]
                for key in (
                    "geometry_id", "source_object_id", "source_anchor_id",
                    "layer", "bbox_dbu", "ownership_quality", "editable",
                    "instance_anchor_id",
                )
                if geometry.get(key) is not None
            }
            for geometry in geometries
        ]
        if "editable_source_object_ids" not in result and geometries:
            result["editable_source_object_ids"] = sorted({
                geometry["source_object_id"]
                for geometry in geometries
                if geometry.get("editable") and geometry.get("source_object_id")
            })
        return result

    @staticmethod
    def _compact_knowledge_pack(value: dict) -> dict:
        """PredicateIR is carried once in rule_predicates, not duplicated here."""
        result = {
            key: value[key]
            for key in (
                "schema_version", "exact_deck_excerpt",
                "reviewed_explanation", "repair_notes",
                "allowed_program_operations",
            )
            if key in value
        }
        predicate = value.get("predicate") or {}
        if predicate:
            result["predicate_ref"] = {
                key: predicate[key]
                for key in (
                    "rule_id", "analyzer_id", "analyzer_version",
                    "classification", "provenance",
                )
                if key in predicate
            }
        return result

    @staticmethod
    def _source_slice(obj: LayoutObject, script: Path | None) -> str | None:
        if script is None or obj.source_span is None:
            return None
        try:
            if Path(obj.source_span.path).resolve() != script.resolve():
                return None
            lines = script.read_text(encoding="utf-8").splitlines()
            value = "\n".join(
                lines[obj.source_span.start_line - 1:obj.source_span.end_line]
            )
            if len(value) > 2400:
                return value[:2386] + "\n# [truncated]"
            return value
        except (OSError, ValueError):
            return None

    def build(
        self, *, region: RegionState, violations: list[ViolationRecord],
        objects: list[LayoutObject], neighbor_messages: list[NeighborMessage],
        blueprint: RepairBlueprint, allowed_actions: set[str],
        script: Path | None = None,
        rule_catalog: RuleCatalog | None = None,
        legal_layers: set[str] | None = None,
        local_topology: dict | None = None,
        rule_predicates: dict[str, dict] | None = None,
        rule_witnesses: dict[str, dict] | None = None,
        rule_knowledge_packs: dict[str, dict] | None = None,
    ) -> CompactRegionContext:
        violation_by_id = {item.violation_id: item for item in violations}
        object_by_id = {item.object_id: item for item in objects}
        local_violations = [violation_by_id[key] for key in region.violation_ids if key in violation_by_id]
        quality_rank = {
            "EXACT_FLATTENED": 0, "SOURCE_EXACT": 1,
            "GEOMETRIC_INFERRED": 2, "MARKER_PROXY": 3,
            "UNAVAILABLE": 4,
        }

        def violation_priority(item: ViolationRecord) -> tuple[int, int, str]:
            witness = (rule_witnesses or {}).get(item.violation_id) or {}
            editable_sources = witness.get("editable_source_object_ids")
            if editable_sources is None:
                editable_sources = {
                    value.get("source_object_id")
                    for value in witness.get("physical_geometries") or []
                    if value.get("editable") and value.get("source_object_id")
                }
            return (not bool(editable_sources), quality_rank.get(
                witness.get("evidence_quality", "UNAVAILABLE"), 4), item.violation_id)

        local_violations.sort(key=violation_priority)
        ox, oy = region.bbox_dbu.x1, region.bbox_dbu.y1
        topology = local_topology or {}
        all_fragments = list((topology.get("fragments") or {}).values())
        local_violation_ids = set(region.violation_ids)
        local_rule_ids = {item.rule_id for item in local_violations}
        local_witnesses = [
            self._compact_witness((rule_witnesses or {})[violation_id])
            for violation_id in sorted(local_violation_ids)
            if violation_id in (rule_witnesses or {})
        ]
        local_predicates = {
            rule_id: (rule_predicates or {})[rule_id]
            for rule_id in sorted(local_rule_ids)
            if rule_id in (rule_predicates or {})
        }
        local_packs = {
            rule_id: self._compact_knowledge_pack(
                (rule_knowledge_packs or {})[rule_id]
            )
            for rule_id in sorted(local_rule_ids)
            if rule_id in (rule_knowledge_packs or {})
        }
        local_fragments = [
            item for item in all_fragments
            if local_violation_ids & set(item.get("violation_ids") or [])
            and Box.model_validate(item["bbox_dbu"]).intersects(
                region.edit_halo_dbu
            )
        ]
        fragments_by_parent: dict[str, list[dict]] = {}
        for fragment in local_fragments:
            fragments_by_parent.setdefault(
                fragment["parent_object_id"], []
            ).append(fragment)

        def context_object(obj: LayoutObject) -> ContextObject:
            fragments = sorted(
                fragments_by_parent.get(obj.object_id, []),
                key=lambda item: item["fragment_id"],
            )
            use_local = bool(fragments) and max(
                obj.bbox_dbu.width, obj.bbox_dbu.height
            ) >= 2 * self.dbu_per_um
            if use_local:
                box = Box.model_validate(fragments[0]["bbox_dbu"])
                for fragment in fragments[1:]:
                    box = box.union(Box.model_validate(fragment["bbox_dbu"]))
                geometry_points = fragments[0].get("geometry_dbu") or []
                geometry = list(geometry_points) or None
                relative = [
                    {"x": int(point["x"]) - ox, "y": int(point["y"]) - oy}
                    for point in geometry_points
                ] or None
            else:
                box = obj.bbox_dbu
                geometry = [point.model_dump(mode="json") for point in obj.geometry_dbu] if obj.geometry_dbu else None
                relative = [
                    {"x": point.x - ox, "y": point.y - oy}
                    for point in obj.geometry_dbu
                ] if obj.geometry_dbu else None
            return ContextObject(
                object_id=obj.object_id, routing_type=obj.routing_type,
                source_anchor_id=obj.source_anchor_id,
                kind=obj.kind, layer=obj.layer, editability=obj.editability.value,
                bbox_dbu=box,
                bbox_relative_dbu=Box(
                    x1=box.x1 - ox, y1=box.y1 - oy,
                    x2=box.x2 - ox, y2=box.y2 - oy,
                ),
                parent_bbox_dbu=(obj.bbox_dbu if use_local else None),
                local_fragment_ids=[
                    item["fragment_id"] for item in fragments
                ] if use_local else [],
                geometry_dbu=geometry, geometry_relative_dbu=relative,
                net_id=obj.net_id, net_mapping_quality=obj.net_mapping_quality,
                connectivity_component_id=obj.connectivity_component_id,
                source_slice=self._source_slice(obj, script),
            )

        editable = [context_object(object_by_id[key]) for key in region.editable_object_ids if key in object_by_id]
        frozen = [context_object(object_by_id[key]) for key in region.frozen_object_ids if key in object_by_id]
        rule_distances: dict[str, int] = {}
        rule_constraints: dict[str, dict] = {}
        geometry_hints: dict[str, list[int]] = {}
        if rule_catalog is not None:
            for violation in local_violations:
                spec, exact = rule_catalog.lookup(violation.rule_id)
                semantics = rule_catalog.repair_semantics_for(
                    violation.rule_id, violation.description,
                )
                rule_constraints[violation.rule_id] = {
                    "exact_catalog_match": exact,
                    "family": spec.family.value,
                    "description": spec.description or violation.description,
                    "repair_semantics": semantics,
                    "primary_layers": list(spec.primary_layers),
                    "allowed_action_families": list(spec.allowed_action_families),
                    "min_influence_nm": spec.min_influence_nm,
                }
                if not exact or spec.min_influence_nm is None:
                    continue
                raw = (spec.min_influence_nm * self.dbu_per_um + 999) // 1000
                grid = self.manufacturing_grid_dbu
                distance = ((raw + grid - 1) // grid) * grid
                if 0 < distance <= self.max_candidate_distance_dbu:
                    rule_distances[violation.rule_id] = distance
                    associated_objects = [
                        object_by_id[object_id]
                        for object_id in violation.associated_object_ids
                        if object_id in object_by_id
                    ]
                    derived: list[int] = []
                    if semantics == "ENCLOSURE":
                        marker = violation.marker_bbox_dbu
                        for obj in associated_objects:
                            if not obj.layer.startswith("M"):
                                continue
                            box = obj.bbox_dbu
                            current_enclosures = [marker.x1 - box.x1, box.x2 - marker.x2, marker.y1 - box.y1, box.y2 - marker.y2]
                            for current in current_enclosures:
                                shortfall = distance - current
                                if shortfall > 0:
                                    derived.append(((shortfall + grid - 1) // grid) * grid)
                    elif semantics == "VIA_METAL_WIDTH":
                        vias = [obj for obj in associated_objects if obj.kind == "via_stack"]
                        metals = [obj for obj in associated_objects if obj.layer.startswith("M")]
                        for metal in metals:
                            perpendicular_is_y = metal.bbox_dbu.width >= metal.bbox_dbu.height
                            metal_width = metal.bbox_dbu.height if perpendicular_is_y else metal.bbox_dbu.width
                            for via in vias:
                                via_width = via.bbox_dbu.height if perpendicular_is_y else via.bbox_dbu.width
                                shortfall = via_width - metal_width
                                if shortfall > 0:
                                    derived.append(((shortfall + grid - 1) // grid) * grid)
                    # Track alignment needs an authoritative routing-track origin.
                    # A pitch alone is insufficient, so only a manufacturing-grid
                    # rule may derive a coordinate shortfall here.
                    elif semantics == "GRID_ALIGNMENT":
                        description = (spec.description or "").lower()
                        horizontal_edges = "horizontal edges" in description
                        for obj in associated_objects:
                            coordinates = [obj.bbox_dbu.y1, obj.bbox_dbu.y2] if horizontal_edges else [obj.bbox_dbu.x1, obj.bbox_dbu.x2]
                            for coordinate in coordinates:
                                remainder = coordinate % distance
                                shift = min(remainder, distance - remainder) if remainder else 0
                                if shift > 0:
                                    derived.append(((shift + grid - 1) // grid) * grid)
                    derived = [
                        value for value in derived
                        if 0 < value <= self.max_candidate_distance_dbu
                    ]
                    if derived:
                        geometry_hints[violation.rule_id] = sorted(set(derived))
        geometry_distances = {
            value for values in geometry_hints.values() for value in values
            if 0 < value <= self.max_candidate_distance_dbu
        }
        allowed_distances = sorted({
            value for value in self.allowed_distances_dbu
            if value <= self.max_candidate_distance_dbu
        } | set(rule_distances.values()) | geometry_distances)
        evidence_ids = set(region.violation_ids)
        evidence_ids.update(
            item["witness_id"] for item in local_witnesses
            if item.get("witness_id")
        )
        for message in neighbor_messages:
            evidence_ids.update(message.evidence_ids)
        evidence_ids.update(blueprint.supporting_experience_ids)
        rule_counts = Counter(item.rule_id for item in local_violations)
        original_counts = {
            "violations": len(local_violations),
            "editable_objects": len(editable),
            "frozen_objects": len(frozen),
            "neighbor_messages": len(neighbor_messages),
        }
        context = CompactRegionContext(
            task_and_hard_constraints=self.HARD_CONSTRAINTS,
            region_identity_and_iteration={
                "region_id": region.region_id,
                "lineage_id": region.lineage_id,
                "iteration": region.iteration,
                "bbox_dbu": region.bbox_dbu.model_dump(mode="json"),
                "edit_halo_dbu": region.edit_halo_dbu.model_dump(mode="json"),
                "quality_flags": sorted(region.quality_flags),
            },
            violation_summary_by_rule=dict(sorted(rule_counts.items())),
            violations=[{
                "violation_id": item.violation_id,
                "rule_id": item.rule_id,
                "rule_family": item.rule_family.value,
                "marker_type": item.marker_type,
                "layers": list(item.layers),
                "bbox_dbu": item.marker_bbox_dbu.model_dump(mode="json"),
                "bbox_relative_dbu": {
                    "x1": item.marker_bbox_dbu.x1 - ox,
                    "y1": item.marker_bbox_dbu.y1 - oy,
                    "x2": item.marker_bbox_dbu.x2 - ox,
                    "y2": item.marker_bbox_dbu.y2 - oy,
                },
                "description": item.description,
                "associated_object_ids": item.associated_object_ids,
                "association_quality": item.association_quality,
            } for item in local_violations],
            editable_objects_with_source_anchors=editable,
            frozen_boundary_context_objects=frozen,
            local_route_fragments=local_fragments,
            local_route_endpoints=[
                item for item in (topology.get("endpoints") or {}).values()
                if item.get("fragment_id") in {
                    fragment["fragment_id"] for fragment in local_fragments
                }
            ],
            via_landings=[
                item for item in (topology.get("via_landings") or {}).values()
                if item.get("violation_id") in local_violation_ids
            ],
            via_metal_adjacencies=[
                item for item in topology.get("via_metal_adjacencies", [])
                if item.get("route_fragment_id") in {
                    fragment["fragment_id"] for fragment in local_fragments
                }
            ],
            local_geometry_origin_dbu={"x": ox, "y": oy},
            neighbor_messages=neighbor_messages,
            failure_history=region.candidate_history,
            rollback_history=region.rollback_history,
            repair_blueprint=blueprint,
            allowed_layout_edit_ir=sorted(allowed_actions - {"NO_OP"}),
            allowed_distance_candidates_dbu=allowed_distances,
            rule_constraints=dict(sorted(rule_constraints.items())),
            rule_predicates=local_predicates,
            rule_witnesses=local_witnesses,
            rule_knowledge_packs=local_packs,
            geometry_distance_hints_dbu=dict(sorted(geometry_hints.items())),
            rule_distance_constraints_dbu=dict(sorted(rule_distances.items())),
            max_candidate_distance_dbu=self.max_candidate_distance_dbu,
            required_structured_output_schema={},
            available_evidence_ids=sorted(evidence_ids),
            available_segment_ids=sorted(
                item.object_id for item in editable
                if item.kind in {"polygon", "path"}
                and item.routing_type == "routing"
            ),
            available_via_ids=sorted(
                item.object_id for item in editable if item.kind == "via_stack"
            ),
            available_net_ids=sorted({
                item.net_id for item in editable if item.net_id
            }),
            legal_routing_layers=sorted(legal_layers or set()),
        )
        prompt_budget = max(512, self.max_context_tokens - 1200)
        truncation_target = max(512, prompt_budget - 256)
        while self._estimate_tokens(context) > truncation_target:
            source_slices = [
                item for item in (
                    context.editable_objects_with_source_anchors
                    + context.frozen_boundary_context_objects
                )
                if item.source_slice is not None
            ]
            if source_slices:
                source_slices[-1].source_slice = None
            elif context.frozen_boundary_context_objects:
                context.frozen_boundary_context_objects.pop()
            elif context.local_route_endpoints:
                context.local_route_endpoints.pop()
            elif context.via_metal_adjacencies:
                context.via_metal_adjacencies.pop()
            elif context.via_landings:
                context.via_landings.pop()
            elif context.failure_history:
                context.failure_history.pop()
            elif context.rollback_history:
                context.rollback_history.pop()
            elif len(context.neighbor_messages) > 1:
                context.neighbor_messages.pop()
            elif len(context.violations) > 1:
                removed_violation_id = context.violations.pop()["violation_id"]
                context.rule_witnesses = [
                    item for item in context.rule_witnesses
                    if item.get("violation_id") != removed_violation_id
                ]
                remaining_rule_ids = {
                    item["rule_id"] for item in context.violations
                }
                for field_name in (
                    "rule_predicates", "rule_knowledge_packs", "rule_constraints",
                    "geometry_distance_hints_dbu", "rule_distance_constraints_dbu",
                ):
                    value = getattr(context, field_name)
                    setattr(context, field_name, {
                        key: item for key, item in value.items()
                        if key in remaining_rule_ids
                    })
            elif len(context.editable_objects_with_source_anchors) > 1:
                context.editable_objects_with_source_anchors.pop()
            elif context.local_route_fragments:
                context.local_route_fragments.pop()
            else:
                raise ValueError(
                    "minimum compact RegionContext exceeds configured token budget"
                )
        retained_violation_ids = {
            item["violation_id"] for item in context.violations
        }
        retained_rule_ids = {item["rule_id"] for item in context.violations}
        context.rule_witnesses = [
            item for item in context.rule_witnesses
            if item.get("violation_id") in retained_violation_ids
        ]
        context.violation_summary_by_rule = dict(sorted(Counter(
            item["rule_id"] for item in context.violations
        ).items()))
        context.rule_predicates = {
            key: value for key, value in context.rule_predicates.items()
            if key in retained_rule_ids
        }
        context.rule_knowledge_packs = {
            key: value for key, value in context.rule_knowledge_packs.items()
            if key in retained_rule_ids
        }
        for field_name in (
            "rule_constraints", "geometry_distance_hints_dbu",
            "rule_distance_constraints_dbu",
        ):
            value = getattr(context, field_name)
            setattr(context, field_name, {
                key: item for key, item in value.items()
                if key in retained_rule_ids
            })
        included_evidence = {
            item["violation_id"] for item in context.violations
        }
        included_evidence.update(
            item["witness_id"] for item in context.rule_witnesses
            if item.get("witness_id")
        )
        for message in context.neighbor_messages:
            included_evidence.update(message.evidence_ids)
        included_evidence.update(blueprint.supporting_experience_ids)
        context.available_evidence_ids = sorted(included_evidence)
        included_objects = {
            item.object_id
            for item in context.editable_objects_with_source_anchors
        } | {
            item.object_id
            for item in context.frozen_boundary_context_objects
        }
        editable_by_id = {
            item.object_id: item
            for item in context.editable_objects_with_source_anchors
        }
        context.available_segment_ids = sorted(
            object_id for object_id, item in editable_by_id.items()
            if item.kind in {"polygon", "path"}
            and item.routing_type == "routing"
        )
        context.available_via_ids = sorted(
            object_id for object_id, item in editable_by_id.items()
            if item.kind == "via_stack"
        )
        context.available_net_ids = sorted({
            item.net_id for item in editable_by_id.values() if item.net_id
        })
        relation_map: dict[str, list[str]] = {
            object_id: [] for object_id in sorted(editable_by_id)
        }
        witness_by_violation = {
            item["violation_id"]: item for item in context.rule_witnesses
            if item.get("violation_id")
        }
        for violation in context.violations:
            associated = sorted(
                set(violation["associated_object_ids"]) & included_objects
            )
            violation["associated_object_ids"] = associated
            witness = witness_by_violation.get(violation["violation_id"])
            has_witness_authority = (
                witness is not None
                and "editable_source_object_ids" in witness
                and witness.get("evidence_quality") in {
                    "EXACT_FLATTENED", "SOURCE_EXACT",
                }
            )
            if has_witness_authority:
                candidates = sorted(
                    set(witness.get("editable_source_object_ids") or [])
                    & set(editable_by_id)
                )
            else:
                candidates = associated or [
                    object_id for object_id, obj in editable_by_id.items()
                    if (
                        (not violation["layers"] or obj.layer in violation["layers"])
                        and obj.bbox_dbu.intersects(
                            Box.model_validate(violation["bbox_dbu"]).expand(
                                self.max_candidate_distance_dbu
                            )
                        )
                    )
                ]
            for object_id in candidates:
                if object_id in relation_map:
                    relation_map[object_id].append(violation["violation_id"])
        context.object_to_violation_ids = {
            key: sorted(set(values)) for key, values in relation_map.items()
            if values
        }
        grounded_object_ids = set(context.object_to_violation_ids)
        context.available_segment_ids = sorted(
            set(context.available_segment_ids) & grounded_object_ids
        )
        context.available_via_ids = sorted(
            set(context.available_via_ids) & grounded_object_ids
        )
        traceable_routing_ids = {
            object_id for object_id, item in editable_by_id.items()
            if item.connectivity_component_id
        }
        context.available_segment_ids = sorted(
            set(context.available_segment_ids) & traceable_routing_ids
        )
        context.available_via_ids = sorted(
            set(context.available_via_ids) & traceable_routing_ids
        )
        context.required_structured_output_schema = {
            "injected_by_llm_client": True,
            "action_family_enum": context.allowed_layout_edit_ir,
            "target_object_id_enum": sorted(context.object_to_violation_ids),
            "target_violation_id_enum": sorted(
                item["violation_id"] for item in context.violations
            ),
            "target_segment_id_enum": context.available_segment_ids,
            "target_via_id_enum": context.available_via_ids,
            "target_net_id_enum": context.available_net_ids,
            "avoid_object_id_enum": sorted(included_objects),
            "target_layer_enum": context.legal_routing_layers,
            "routing_geometry_is_lowered_deterministically": True,
            "rationale_evidence_id_enum": context.available_evidence_ids,
            "distance_candidates_dbu_enum": (
                context.allowed_distance_candidates_dbu
            ),
            "geometry_distance_hints_dbu": (
                context.geometry_distance_hints_dbu
            ),
            "rule_distance_constraints_dbu": (
                context.rule_distance_constraints_dbu
            ),
            "object_to_violation_ids": context.object_to_violation_ids,
        }
        context.omitted_context_summary = {
            "violations": original_counts["violations"] - len(context.violations),
            "editable_objects": original_counts["editable_objects"] - len(context.editable_objects_with_source_anchors),
            "frozen_objects": original_counts["frozen_objects"] - len(context.frozen_boundary_context_objects),
            "neighbor_messages": original_counts["neighbor_messages"] - len(context.neighbor_messages),
        }
        context.context_budget = {
            "max_tokens": self.max_context_tokens,
            "reserved_schema_tokens": 1200,
            "estimated_context_tokens": self._estimate_tokens(context),
            "truncated": any(context.omitted_context_summary.values()),
        }
        return context

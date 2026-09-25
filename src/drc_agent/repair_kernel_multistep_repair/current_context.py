from __future__ import annotations

from collections import defaultdict
from collections import OrderedDict
from pathlib import Path
from threading import RLock

from drc_agent.config.loader import AppConfig
from drc_agent.regions.builder import RegionBuilder
from drc_agent.regions.connectivity import map_connectivity_components
from drc_agent.regions.parser import (
    SourceObjectMapper,
    ViolationParser,
    load_rule_catalog,
)
from drc_agent.regions.physical import SourceHierarchyPhysicalBuilder
from drc_agent.regions.topology import LocalTopologyBuilder
from drc_agent.rules import RulePredicateRegistry, RuleWitnessBuilder
from drc_agent.schemas.common import Box, file_sha256, stable_hash
from drc_agent.schemas.rules import FlattenedPhysicalGeometry
from drc_agent.schemas.state import LayoutObject, ViolationRecord
from drc_agent.reliability import IntegrityFailure

from .identity import build_debt_marker_groups
from .models import (
    CurrentRulePredicate,
    CurrentRuleWitness,
    CurrentSemanticContext,
    SemanticFidelity,
    SourceLineageRecord,
)


_REVIEWED = {
    "M1.W.1": ("MINIMUM_WIDTH", ("M1",), 72, (561, 561)),
    "M4.S.2": ("SAME_LAYER_SPACING", ("M4",), 160, (852, 852)),
    "M4.S.3": ("OPPOSITE_SIDE_CLEARANCE", ("M4",), 160, (859, 859)),
    "V1.AUX.1": ("VIA_METAL_INSIDE", ("V1", "M1", "M2"), None, (744, 744)),
    "V3.M3.EN.1": ("VIA_ENCLOSURE", ("V3", "M3"), 20, (826, 826)),
    "V3.AUX.1": ("VIA_METAL_INSIDE", ("V3", "M3", "M4"), None, (828, 828)),
}


def _current_required_dbu(predicate) -> int | None:
    """Keep the exact deck threshold needed by current-debt lowering.

    Conditional spacing is cleared by leaving every ``<=`` trigger range;
    the manufacturing-grid step makes that boundary strictly greater without
    inventing a rule-specific constant. Other relations retain their largest
    reviewed distance requirement.
    """
    values = [
        int(clause.required_value_dbu)
        for clause in predicate.actual_predicate
        if clause.required_value_dbu is not None
    ]
    if not values:
        return None
    if predicate.relation_type.value == "CONDITIONAL_PARALLEL_RUN":
        trigger_values = [
            int(clause.required_value_dbu)
            for clause in predicate.actual_predicate
            if clause.required_value_dbu is not None
            and clause.comparator == "<="
        ]
        if trigger_values:
            trigger = max(trigger_values)
            grid = int(
                predicate.measurement_semantics.manufacturing_grid_dbu
            )
            # The condition is active at ``<= trigger``.  Return the first
            # legal manufacturing-grid coordinate strictly above it; adding
            # one grid step directly to an off-grid deck threshold would
            # itself produce an illegal coordinate (for example 129+4=133).
            return ((trigger // grid) + 1) * grid
    return max(values)


def _inside(inner: Box, outer: Box) -> bool:
    return (
        outer.x1 <= inner.x1 <= inner.x2 <= outer.x2
        and outer.y1 <= inner.y1 <= inner.y2 <= outer.y2
    )


def _local_physical(
    violation: ViolationRecord,
    physical: list[FlattenedPhysicalGeometry],
    layers: set[str],
) -> list[FlattenedPhysicalGeometry]:
    halo = violation.marker_bbox_dbu.expand(256)
    return sorted([
        item for item in physical
        if item.layer in layers and item.bbox_dbu.intersects(halo)
    ], key=lambda item: item.geometry_id)


def _target_via(
    violation: ViolationRecord,
    local: list[FlattenedPhysicalGeometry],
    via_layer: str,
) -> tuple[FlattenedPhysicalGeometry | None, list[str]]:
    vias = [item for item in local if item.layer == via_layer]
    hits = [
        item for item in vias
        if item.bbox_dbu.intersection_area(violation.marker_bbox_dbu) > 0
        or item.bbox_dbu == violation.marker_bbox_dbu
    ]
    reasons = []
    if len(hits) != 1:
        reasons.append(f"CURRENT_VIA_MULTIPLICITY_{len(hits)}")
    via = min(
        hits or vias,
        key=lambda item: (
            -item.bbox_dbu.intersection_area(violation.marker_bbox_dbu),
            item.geometry_id,
        ),
        default=None,
    )
    if via is None:
        reasons.append("CURRENT_VIA_UNAVAILABLE")
    return via, reasons


def _current_reviewed_witness(
    violation: ViolationRecord,
    predicate: CurrentRulePredicate,
    physical: list[FlattenedPhysicalGeometry],
) -> CurrentRuleWitness:
    local = _local_physical(
        violation, physical, set(predicate.involved_layers),
    )
    reasons: list[str] = []
    selected = local
    details = {"reviewed_deck_lines": predicate.deck_lines}
    measurements = {}
    if violation.rule_id in {"V3.AUX.1", "V3.M3.EN.1"}:
        via, via_reasons = _target_via(violation, local, "V3")
        reasons.extend(via_reasons)
        selected = [] if via is None else [via]
        if via is not None:
            metal_layers = (
                ("M3", "M4") if violation.rule_id == "V3.AUX.1"
                else ("M3",)
            )
            inside_by_layer = {}
            for layer in metal_layers:
                metals = [
                    item for item in local
                    if item.layer == layer
                    and item.bbox_dbu.intersects(via.bbox_dbu)
                ]
                selected.extend(metals)
                if not metals:
                    reasons.append(f"CURRENT_{layer}_CONTRIBUTOR_UNAVAILABLE")
                    inside_by_layer[layer] = False
                    continue
                union = Box(
                    x1=min(item.bbox_dbu.x1 for item in metals),
                    y1=min(item.bbox_dbu.y1 for item in metals),
                    x2=max(item.bbox_dbu.x2 for item in metals),
                    y2=max(item.bbox_dbu.y2 for item in metals),
                )
                inside_by_layer[layer] = _inside(via.bbox_dbu, union)
                measurements[f"{layer}_enclosure"] = {
                    "left": via.bbox_dbu.x1 - union.x1,
                    "right": union.x2 - via.bbox_dbu.x2,
                    "bottom": via.bbox_dbu.y1 - union.y1,
                    "top": union.y2 - via.bbox_dbu.y2,
                }
            details["inside_by_layer"] = inside_by_layer
    source_anchors = sorted({
        item.source_anchor_id for item in selected if item.source_anchor_id
    })
    instance_anchors = sorted({
        item.instance_anchor_id for item in selected if item.instance_anchor_id
    })
    fidelity = (
        SemanticFidelity.REVIEWED
        if selected and not reasons else
        SemanticFidelity.APPROXIMATE if selected else
        SemanticFidelity.UNAVAILABLE
    )
    return CurrentRuleWitness(
        witness_id="current_witness_" + stable_hash([
            violation.violation_id, predicate.predicate_id,
            [item.geometry_id for item in selected], measurements,
        ])[:20],
        violation_id=violation.violation_id,
        rule_id=violation.rule_id,
        predicate_id=predicate.predicate_id,
        relation_type=predicate.relation_type,
        participating_physical_geometry_ids=[
            item.geometry_id for item in selected
        ],
        source_anchor_ids=source_anchors,
        instance_anchor_ids=instance_anchors,
        physical_geometries=selected,
        current_measurements=measurements,
        relation_details=details,
        fidelity=fidelity,
        unresolved_reasons=reasons,
    )


def _origin_cell(name: str | None) -> str | None:
    if not name or "_drc_agent_spec_" not in name:
        return None
    return name.split("_drc_agent_spec_", 1)[0]


def _origin_cell_aliases(origin: str | None) -> set[str]:
    if not origin:
        return set()
    return {
        origin,
        origin if origin.startswith("cell_") else f"cell_{origin}",
    }


def _source_object_identity(item: LayoutObject) -> str:
    return stable_hash({
        "object_id": item.object_id,
        "source_anchor_id": item.source_anchor_id,
        "source_variable": item.source_variable,
        "source_cell": item.source_cell,
        "layer": item.layer,
        "source_span_sha256": (
            item.source_span.source_hash if item.source_span else None
        ),
    })


def _source_lineage(
    root_objects: list[LayoutObject], current_objects: list[LayoutObject],
    receipts: dict[str, dict] | None = None,
) -> dict[str, SourceLineageRecord]:
    by_anchor = {
        item.source_anchor_id: item for item in root_objects
        if item.source_anchor_id
    }
    by_id = {item.object_id: item for item in root_objects}
    by_cell_layer: dict[tuple[str, str], list[LayoutObject]] = defaultdict(list)
    for item in root_objects:
        if item.source_cell:
            by_cell_layer[(item.source_cell, item.layer)].append(item)
    result = {}
    for current in current_objects:
        receipt = (receipts or {}).get(current.object_id)
        direct = receipt if receipt and receipt.get(
            "old_source_object_id"
        ) else None
        parent = (
            by_id.get(direct["old_source_object_id"])
            if direct else by_anchor.get(current.source_anchor_id)
        )
        origin = _origin_cell(current.source_cell)
        confidence = SemanticFidelity.EXACT
        reasons = ["SOURCE_ANCHOR_STABLE"]
        if direct:
            if (
                parent is None
                or parent.source_anchor_id != direct["old_source_anchor_id"]
                or _source_object_identity(parent)
                    != direct["before_source_identity"]
                or parent.geometry_hash
                    != direct["before_physical_geometry_identity"]
                or current.source_anchor_id
                    != direct["child_source_anchor_id"]
            ):
                raise IntegrityFailure(
                    "DIRECT_LINEAGE_PARENT_IDENTITY_MISMATCH",
                    failure_code="DIRECT_LINEAGE_PARENT_IDENTITY_MISMATCH",
                    failure_stage="CURRENT_CONTEXT_LINEAGE",
                )
            confidence = SemanticFidelity.EXACT
            reasons = ["VERIFIED_COMPILER_DIRECT_MUTATION_RECEIPT"]
        elif parent is not None and (
            parent.object_id != current.object_id
            or parent.geometry_hash != current.geometry_hash
            or _source_object_identity(parent) != _source_object_identity(current)
        ):
            # A stable variable/line anchor is not proof across a source
            # mutation.  Unsupported split/merge and complex rewrites stay
            # UNKNOWN until a compiler receipt supplies correspondence.
            parent = None
            confidence = SemanticFidelity.UNAVAILABLE
            reasons = [
                "CURRENT_SOURCE_CHANGED_WITHOUT_VERIFIED_CORRESPONDENCE"
            ]
        elif parent is None and origin:
            candidates = sorted([
                item for alias in _origin_cell_aliases(origin)
                for item in by_cell_layer.get((alias, current.layer), [])
            ],
                key=lambda item: (
                    abs(item.bbox_dbu.width - current.bbox_dbu.width)
                    + abs(item.bbox_dbu.height - current.bbox_dbu.height),
                    item.object_id,
                ),
            )
            # A cell-name hint is diagnostic, never nearest-bbox authority.
            parent = candidates[0] if len(candidates) == 1 else None
            confidence = SemanticFidelity.REVIEWED
            reasons = ["SPECIALIZED_CELL_ORIGIN_RECONSTRUCTED"]
        elif parent is None:
            confidence = SemanticFidelity.UNAVAILABLE
            reasons = ["PARENT_SOURCE_UNAVAILABLE"]
        record = SourceLineageRecord(
            lineage_id="source_lineage_" + stable_hash([
                parent.object_id if parent else None,
                current.object_id, origin, direct,
            ])[:20],
            parent_snapshot_source_object_id=(
                parent.object_id if parent else None
            ),
            child_snapshot_source_object_id=current.object_id,
            old_source_anchor_id=(parent.source_anchor_id if parent else None),
            new_source_anchor_id=current.source_anchor_id,
            specialized_cell_origin=origin,
            current_specialized_cell=(current.source_cell if origin else None),
            parent_snapshot_id=(
                direct.get("parent_snapshot_id") if direct else None
            ),
            child_snapshot_id=(
                direct.get("child_snapshot_id") if direct else None
            ),
            compiler_operation=(
                direct.get("compiler_operation") if direct else None
            ),
            before_source_identity=(
                direct.get("before_source_identity") if direct else None
            ),
            after_source_identity=(
                direct.get("after_source_identity") if direct else None
            ),
            before_physical_geometry_identity=(
                direct.get("before_physical_geometry_identity")
                if direct else None
            ),
            after_physical_geometry_identity=(
                direct.get("after_physical_geometry_identity")
                if direct else None
            ),
            verification_status=(
                direct.get("verification_status") if direct else None
            ),
            lineage_confidence=confidence,
            reason_codes=reasons,
        )
        result[current.object_id] = record
    return result


def _instance_lineage(
    root_physical: list[FlattenedPhysicalGeometry],
    current_physical: list[FlattenedPhysicalGeometry],
    receipts: dict[str, dict] | None = None,
) -> dict[str, SourceLineageRecord]:
    # Lineage matching is exact on layer and flattened bbox before applying
    # the existing source-cell/origin predicate.  Index that invariant once;
    # scanning every root geometry for every current occurrence makes large
    # large hierarchical single-rule preparation quadratic without changing
    # the answer.
    root_by_layer_bbox: dict[
        tuple[str, int, int, int, int], list[FlattenedPhysicalGeometry]
    ] = defaultdict(list)
    for item in root_physical:
        box = item.bbox_dbu
        root_by_layer_bbox[(item.layer, box.x1, box.y1, box.x2, box.y2)].append(item)
    root_by_instance_anchor = {
        item.instance_anchor_id: item
        for item in root_physical if item.instance_anchor_id
    }
    result = {}
    for current in current_physical:
        if not current.instance_anchor_id:
            continue
        origin = _origin_cell(current.source_cell)
        aliases = _origin_cell_aliases(origin)
        box = current.bbox_dbu
        candidates = [
            item for item in root_by_layer_bbox.get(
                (current.layer, box.x1, box.y1, box.x2, box.y2), []
            ) if (
                item.source_cell == current.source_cell
                or item.source_cell in aliases
            )
        ]
        # Geometry equality alone does not identify two congruent occurrences.
        exact = [item for item in candidates
            if item.occurrence_provenance and current.occurrence_provenance
            and item.occurrence_provenance.statement_ast_hash == current.occurrence_provenance.statement_ast_hash
            and item.occurrence_provenance.parent_cell == current.occurrence_provenance.parent_cell
            and item.occurrence_provenance.transform == current.occurrence_provenance.transform]
        parents = {item.instance_anchor_id for item in exact}
        parent = exact[0] if len(parents) == 1 else None
        receipt = (receipts or {}).get(current.instance_anchor_id)
        if receipt:
            origin = receipt["old_source_cell"]
            parent = root_by_instance_anchor.get(
                receipt["old_instance_anchor_id"]
            )
        result[current.instance_anchor_id] = SourceLineageRecord(
            lineage_id="instance_lineage_" + stable_hash([
                parent.instance_anchor_id if parent else None,
                current.instance_anchor_id, current.geometry_id,
            ])[:20],
            parent_snapshot_source_object_id=(
                parent.source_object_id if parent else None
            ),
            child_snapshot_source_object_id=current.source_object_id or "",
            old_source_anchor_id=(parent.source_anchor_id if parent else None),
            new_source_anchor_id=current.source_anchor_id,
            old_instance_anchor_id=(
                receipt["old_instance_anchor_id"] if receipt else
                parent.instance_anchor_id if parent else None
            ),
            new_instance_anchor_id=current.instance_anchor_id,
            specialized_cell_origin=origin,
            current_specialized_cell=(current.source_cell if origin else None),
            lineage_confidence=(
                SemanticFidelity.EXACT if receipt or parent else
                SemanticFidelity.REVIEWED if origin else
                SemanticFidelity.UNAVAILABLE
            ),
            reason_codes=[
                "VERIFIED_COMPILER_OCCURRENCE_RECEIPT" if receipt else
                "INSTANCE_GEOMETRY_CORRESPONDENCE" if parent else
                "SPECIALIZED_INSTANCE_CURRENT_OCCURRENCE" if origin else
                "PARENT_INSTANCE_UNAVAILABLE"
            ],
        )
    return result


_CONTEXT_CACHE = OrderedDict()
_CONTEXT_CACHE_LOCK = RLock()


class CurrentSemanticContextBuilder:
    def __init__(
        self, project_root: Path, config: AppConfig, *,
        rule_deck_path: Path,
        require_frozen_rule_deck: bool = True,
    ):
        self.root = project_root.resolve()
        self.config = config
        self.rule_deck = rule_deck_path.resolve()
        self.require_frozen_rule_deck = bool(require_frozen_rule_deck)
        self.catalog = load_rule_catalog(
            self.root / "configs/rules/asap7_rule_catalog.yaml"
        )

    def build(self, **kwargs) -> CurrentSemanticContext:
        # Pure source/DRC reconstruction only; this is never an EDA evidence
        # cache. Return a deep copy so another Region cannot mutate cached truth.
        paths = {key:str(Path(value).resolve()) for key,value in kwargs.items()
                 if key.endswith("_path") and value is not None}
        identity = stable_hash({"version":"phase-four-current-context-v1",
            "inputs":{key:[value,file_sha256(Path(value))] for key,value in paths.items()},
            "deck":[str(self.rule_deck),file_sha256(self.rule_deck)],
            "require_frozen_rule_deck": self.require_frozen_rule_deck,
            "catalog":file_sha256(self.root/"configs/rules/asap7_rule_catalog.yaml"),
            "config":self.config.model_dump(mode="json"),
            "arguments":{key:value for key,value in kwargs.items() if key not in paths}})
        with _CONTEXT_CACHE_LOCK:
            value = _CONTEXT_CACHE.get(identity)
            if value is None:
                value = self._build_uncached(**kwargs)
                _CONTEXT_CACHE[identity] = value
                while len(_CONTEXT_CACHE) > 2:
                    _CONTEXT_CACHE.popitem(last=False)
            else:
                _CONTEXT_CACHE.move_to_end(identity)
            return value.model_copy(deep=True)

    def _build_uncached(
        self, *, snapshot_id: str, case_id: str, script_path: Path,
        drc_path: Path, connectivity_path: Path,
        root_script_path: Path | None = None,
        lineage_receipt: dict | None = None,
        lineage_parent_snapshot_id: str | None = None,
        lineage_run_id: str | None = None,
        require_physical_lineage: bool = False,
    ) -> CurrentSemanticContext:
        script = script_path.resolve()
        violations = ViolationParser().parse_dac26_json(
            drc_path, case_id=case_id, rule_catalog=self.catalog,
        )
        objects = SourceObjectMapper().build_map(script)
        objects, _ = map_connectivity_components(objects, connectivity_path)
        registry = RulePredicateRegistry(
            self.rule_deck,
            require_frozen_hash=self.require_frozen_rule_deck,
        )
        layers = set()
        predicates: dict[str, CurrentRulePredicate] = {}
        packs = {}
        for rule_id in sorted({item.rule_id for item in violations}):
            try:
                pack = registry.get(rule_id)
            except KeyError:
                pack = None
            if pack is not None:
                packs[rule_id] = pack
                predicate = pack.predicate
                layers.update(predicate.involved_layers)
                predicates[rule_id] = CurrentRulePredicate(
                    predicate_id=stable_hash(predicate.model_dump(mode="json")),
                    rule_id=rule_id,
                    relation_type=predicate.relation_type.value,
                    involved_layers=list(predicate.involved_layers),
                    required_dbu=_current_required_dbu(predicate),
                    deck_sha256=registry.deck_sha256,
                    deck_lines=[
                        predicate.provenance.start_line,
                        predicate.provenance.end_line,
                    ],
                    reviewed=True,
                )
            elif rule_id in _REVIEWED:
                relation, involved, required, deck_lines = _REVIEWED[rule_id]
                layers.update(involved)
                predicates[rule_id] = CurrentRulePredicate(
                    predicate_id=stable_hash([
                        rule_id, relation, involved, required,
                        registry.deck_sha256, deck_lines,
                    ]),
                    rule_id=rule_id, relation_type=relation,
                    involved_layers=list(involved), required_dbu=required,
                    deck_sha256=registry.deck_sha256,
                    deck_lines=list(deck_lines), reviewed=True,
                )
        physical = SourceHierarchyPhysicalBuilder().build(
            script, objects, layer_filter=layers or None,
        )
        witnesses: dict[str, CurrentRuleWitness] = {}
        witness_builder = RuleWitnessBuilder(
            registry, prefer_current_marker_pair=True,
        )
        for violation in violations:
            predicate = predicates.get(violation.rule_id)
            if predicate is None:
                continue
            if violation.rule_id in packs:
                try:
                    raw = witness_builder.build(violation, physical, objects)
                except (KeyError, ValueError):
                    continue
                witnesses[violation.violation_id] = CurrentRuleWitness(
                    witness_id=raw.witness_id,
                    violation_id=raw.violation_id,
                    rule_id=raw.rule_id,
                    predicate_id=raw.predicate_id,
                    relation_type=raw.offending_relation.relation_type.value,
                    participating_physical_geometry_ids=raw.participating_physical_geometry_ids,
                    source_anchor_ids=sorted({
                        item.source_anchor_id for item in raw.physical_geometries
                        if item.source_anchor_id
                    }),
                    instance_anchor_ids=sorted({
                        item.instance_anchor_id for item in raw.physical_geometries
                        if item.instance_anchor_id
                    }),
                    physical_geometries=raw.physical_geometries,
                    current_measurements={
                        item.name: item.model_dump(mode="json")
                        for item in raw.current_measurements
                    },
                    relation_details=raw.offending_relation.details,
                    fidelity=(
                        SemanticFidelity.EXACT
                        if raw.predicate_fidelity.value == "EXACT_SIGNOFF_PREDICATE"
                        else SemanticFidelity.REVIEWED
                    ),
                    unresolved_reasons=raw.unresolved_reasons,
                )
            else:
                witnesses[violation.violation_id] = _current_reviewed_witness(
                    violation, predicate, physical,
                )
        built = RegionBuilder(self.config.region).build(
            violations, objects, self.catalog,
        )
        topology = LocalTopologyBuilder(
            dbu_per_um=self.config.region.dbu_per_um,
        ).build(violations, objects).model_dump(mode="json")
        root_script = (root_script_path or script).resolve()
        root_objects = SourceObjectMapper().build_map(root_script)
        root_physical = SourceHierarchyPhysicalBuilder().build(
            root_script, root_objects, layer_filter=layers or None,
        )
        receipts = {}
        if lineage_receipt is not None:
            from drc_agent.patching.compiler import verify_lineage_receipt
            if root_script_path is None:
                raise IntegrityFailure(
                    "LINEAGE_PARENT_SCRIPT_REQUIRED",
                    failure_code="LINEAGE_PARENT_SCRIPT_REQUIRED",
                    failure_stage="CURRENT_CONTEXT_LINEAGE",
                )
            receipts = verify_lineage_receipt(
                script,
                lineage_receipt,
                objects,
                parent_script=root_script,
                expected_parent_snapshot_id=lineage_parent_snapshot_id,
                expected_child_snapshot_id=snapshot_id,
                expected_run_id=lineage_run_id,
                require_physical_verified=require_physical_lineage,
            )
        source_lineage = _source_lineage(root_objects, objects, receipts)
        instance_lineage = _instance_lineage(root_physical, physical, receipts)
        groups = build_debt_marker_groups(violations, snapshot_id)
        source_hashes = {
            item.source_anchor_id: item.source_span.source_hash
            for item in objects if item.source_anchor_id and item.source_span
        }
        material = {
            "snapshot": snapshot_id,
            "script": file_sha256(script),
            "drc": file_sha256(drc_path),
            "groups": [item.model_dump(mode="json") for item in groups],
            "witnesses": {
                key: value.model_dump(mode="json")
                for key, value in sorted(witnesses.items())
            },
            "receipt_digest": stable_hash(lineage_receipt) if lineage_receipt else None,
            "lineage": {
                key: value.model_dump(mode="json")
                for key, value in sorted(source_lineage.items())
            },
        }
        return CurrentSemanticContext(
            context_id="current_context_" + stable_hash(material)[:20],
            snapshot_id=snapshot_id, case_id=case_id,
            current_script=str(script), current_drc=str(drc_path.resolve()),
            violations=violations, source_objects=objects,
            physical_geometries=physical, rule_predicates=predicates,
            rule_witnesses=witnesses, regions=built.regions,
            violation_to_region=built.violation_to_region,
            local_topology=topology, source_hashes=source_hashes,
            manufacturing_grid_dbu=self.config.backend.manufacturing_grid_dbu,
            source_lineage_map=source_lineage,
            instance_lineage_map=instance_lineage,
            marker_groups=groups,
            context_fingerprint=stable_hash(material),
        )

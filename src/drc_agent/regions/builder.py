from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import exp

from pydantic import BaseModel, Field

from drc_agent.config.loader import RegionConfig
from drc_agent.schemas.common import Box, EditabilityClass, stable_hash
from drc_agent.schemas.state import (
    AffinityEvidence, LayoutObject, RegionFeatures, RegionState, RegionStatus,
    ResourceContext, RuleCatalog, ViolationRecord,
)


class RegionBuildResult(BaseModel):
    violations: list[ViolationRecord]
    regions: list[RegionState]
    affinity_edges: list[dict]
    violation_to_region: dict[str, str]


class _UnionFind:
    def __init__(self, ids: list[str]):
        self.parent = {item: item for item in ids}

    def find(self, item: str) -> str:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if b < a:
            a, b = b, a
        self.parent[b] = a

    def groups(self) -> list[list[str]]:
        result: dict[str, list[str]] = {}
        for item in sorted(self.parent):
            result.setdefault(self.find(item), []).append(item)
        return list(result.values())


@dataclass(frozen=True)
class _Seed:
    violation: ViolationRecord
    objects: tuple[LayoutObject, ...]
    seed: Box
    halo: Box
    unknown_rule: bool


class RegionBuilder:
    def __init__(self, cfg: RegionConfig):
        self.cfg = cfg

    def _associate(
        self, violation: ViolationRecord, objects: list[LayoutObject],
    ) -> tuple[LayoutObject, ...]:
        layers = set(violation.layers)
        layer_compatible = [
            obj for obj in objects if not layers or obj.layer in layers
        ]
        marker = violation.marker_bbox_dbu
        nearby = [
            obj for obj in layer_compatible
            if marker.expand(self.cfg.affinity_max_gap_dbu).intersects(
                obj.bbox_dbu
            )
        ]
        exact = [obj for obj in nearby if marker.intersects(obj.bbox_dbu)]
        selected = exact or nearby[:16]
        violation.associated_object_ids = sorted(
            obj.object_id for obj in selected
        )
        violation.association_quality = (
            "geometric" if selected else "unknown"
        )
        return tuple(sorted(selected, key=lambda obj: obj.object_id))

    def _seed(self, violation: ViolationRecord, objects: tuple[LayoutObject, ...], catalog: RuleCatalog) -> _Seed:
        seed = violation.marker_bbox_dbu
        for obj in objects:
            local = obj.bbox_dbu
            if max(local.width, local.height) > self.cfg.dbu_per_um * self.cfg.seed_window_um:
                local = Box(
                    x1=max(local.x1, violation.marker_bbox_dbu.x1 - self.cfg.dbu_per_um),
                    y1=max(local.y1, violation.marker_bbox_dbu.y1 - self.cfg.dbu_per_um),
                    x2=min(local.x2, violation.marker_bbox_dbu.x2 + self.cfg.dbu_per_um),
                    y2=min(local.y2, violation.marker_bbox_dbu.y2 + self.cfg.dbu_per_um),
                )
            seed = seed.union(local)
        rule, known = catalog.lookup(violation.rule_id)
        rule_dbu = int((rule.min_influence_nm or 0) * self.cfg.dbu_per_um / 1000 * rule.halo_multiplier)
        marker_extent = max(violation.marker_bbox_dbu.width, violation.marker_bbox_dbu.height)
        halo = max(rule_dbu, marker_extent // 2, self.cfg.halo_min_dbu)
        halo = min(halo, self.cfg.halo_max_dbu)
        return _Seed(violation=violation, objects=objects, seed=seed,
                     halo=seed.expand(halo), unknown_rule=not known)

    @staticmethod
    def _evidence(a: _Seed, b: _Seed, max_gap: int) -> tuple[bool, bool, AffinityEvidence]:
        shared_context = set(a.violation.associated_object_ids) & set(b.violation.associated_object_ids)
        editable_a = {obj.object_id for obj in a.objects if obj.editability in {EditabilityClass.A, EditabilityClass.B}}
        editable_b = {obj.object_id for obj in b.objects if obj.editability in {EditabilityClass.A, EditabilityClass.B}}
        shared_editable = editable_a & editable_b
        hard = bool(shared_editable)
        min_area = min(a.halo.area, b.halo.area)
        overlap = a.halo.intersection_area(b.halo) / min_area if min_area else 0.0
        gap = a.halo.gap(b.halo)
        gap_score = exp(-gap / max(max_gap, 1))
        same_family = a.violation.rule_family == b.violation.rule_family
        layer_relation = 1.0 if set(a.violation.layers) & set(b.violation.layers) else 0.0
        evidence = AffinityEvidence(
            halo_overlap_ratio=overlap, normalized_gap_score=gap_score,
            same_rule=a.violation.rule_id == b.violation.rule_id,
            same_rule_family=same_family, layer_relation=layer_relation,
            shared_object_context=1.0 if shared_context else 0.0,
        )
        locality = a.halo.intersects(b.halo) or gap <= max_gap
        semantic_count = sum([same_family and bool(layer_relation), bool(shared_editable)])
        merge = locality and ((overlap >= 0.10 and semantic_count >= 1) or
                              (semantic_count >= 2 and gap_score >= 0.60))
        return hard, merge, evidence

    def _cap_ok(self, seeds: list[_Seed]) -> bool:
        bbox = seeds[0].seed
        objects = set()
        for seed in seeds:
            bbox = bbox.union(seed.seed)
            objects.update(obj.object_id for obj in seed.objects)
        max_span = int(self.cfg.max_region_span_um * self.cfg.dbu_per_um)
        return (len(seeds) <= self.cfg.max_region_markers and
                len(objects) <= self.cfg.max_region_objects and
                max(bbox.width, bbox.height) <= max_span and
                (len(seeds) * 180 + len(objects) * 80) <= self.cfg.max_context_tokens)

    def _split(self, seeds: list[_Seed]) -> list[list[_Seed]]:
        if self._cap_ok(seeds) or len(seeds) <= 1:
            return [seeds]
        hard_objects = Counter(obj.object_id for seed in seeds for obj in seed.objects)
        if any(count > 1 for count in hard_objects.values()):
            return [seeds]
        bbox = seeds[0].seed
        for seed in seeds[1:]:
            bbox = bbox.union(seed.seed)
        axis = 0 if bbox.width >= bbox.height else 1
        ordered = sorted(seeds, key=lambda seed: (seed.seed.centroid[axis], seed.violation.violation_id))
        candidates = []
        for cut in range(1, len(ordered)):
            left, right = ordered[:cut], ordered[cut:]
            imbalance = abs(len(left) - len(right))
            cut_gap = right[0].seed.centroid[axis] - left[-1].seed.centroid[axis]
            candidates.append((imbalance * 1.0 - cut_gap * 0.001, cut, left, right))
        _, _, left, right = min(candidates, key=lambda item: (item[0], item[1]))
        return self._split(left) + self._split(right)

    def _finalize(self, seeds: list[_Seed], previous: list[RegionState]) -> RegionState:
        violations = sorted((seed.violation for seed in seeds), key=lambda item: item.violation_id)
        objects = {obj.object_id: obj for seed in seeds for obj in seed.objects}
        bbox = seeds[0].seed
        halo = seeds[0].halo
        for seed in seeds[1:]:
            bbox, halo = bbox.union(seed.seed), halo.union(seed.halo)
        identity = stable_hash([item.violation_id for item in violations])
        region_id = f"region_{identity[:16]}"
        old = max(previous, key=lambda item: len(set(item.violation_ids) & {v.violation_id for v in violations}), default=None)
        overlap = len(set(old.violation_ids) & {v.violation_id for v in violations}) if old else 0
        lineage_id = old.lineage_id if old and overlap / max(len(set(old.violation_ids) | {v.violation_id for v in violations}), 1) >= .35 else f"lineage_{identity[:16]}"
        editable = sorted(obj.object_id for obj in objects.values() if obj.source_anchor_id and obj.editability != EditabilityClass.D)
        frozen = sorted(set(objects) - set(editable))
        flags = {"UNKNOWN_RULE" for seed in seeds if seed.unknown_rule}
        oversized = not self._cap_ok(seeds)
        if oversized:
            flags.add("OVERSIZED_HARD_COMPONENT")
        content = {
            "violations": [v.violation_id for v in violations], "objects": sorted(objects),
            "bbox": bbox.model_dump(), "halo": halo.model_dump(),
        }
        bin_size = max(self.cfg.affinity_max_gap_dbu, 1)
        occupied_proxy_bins = sorted({
            f"proxy:{bin_size}:{obj.layer}:{x_index}:{y_index}"
            for obj in objects.values()
            for x_index in range(
                obj.bbox_dbu.x1 // bin_size,
                (max(obj.bbox_dbu.x2 - 1, obj.bbox_dbu.x1) // bin_size) + 1,
            )
            for y_index in range(
                obj.bbox_dbu.y1 // bin_size,
                (max(obj.bbox_dbu.y2 - 1, obj.bbox_dbu.y1) // bin_size) + 1,
            )
        })
        mapped_objects = [obj for obj in objects.values() if obj.net_id]
        mapping_qualities = {obj.net_mapping_quality for obj in mapped_objects}
        net_quality = (
            next(iter(mapping_qualities))
            if mapped_objects and len(mapped_objects) == len(objects)
            and len(mapping_qualities) == 1
            else "partial" if mapped_objects else "unavailable"
        )
        return RegionState(
            region_id=region_id, lineage_id=lineage_id, status=RegionStatus.OVERSIZED if oversized else RegionStatus.ACTIVE,
            iteration=max(v.report_iteration for v in violations), bbox_dbu=bbox, edit_halo_dbu=halo,
            layers=sorted({layer for v in violations for layer in v.layers}),
            violation_ids=[v.violation_id for v in violations], rule_ids=sorted({v.rule_id for v in violations}),
            rule_family_histogram=dict(sorted(Counter(v.rule_family.value for v in violations).items())),
            object_ids=sorted(objects), editable_object_ids=editable, frozen_object_ids=frozen,
            segment_ids=sorted(
                obj.object_id for obj in objects.values()
                if obj.kind in {"polygon", "path"} and obj.routing_type == "routing"
            ),
            via_ids=sorted(obj.object_id for obj in objects.values() if obj.kind == "via_stack"),
            cell_instance_ids=sorted(obj.object_id for obj in objects.values() if obj.kind == "cell_instance"),
            net_ids=sorted({obj.net_id for obj in objects.values() if obj.net_id}),
            net_mapping_quality=net_quality,
            resource_context=ResourceContext(
                source=("geometry_proxy" if occupied_proxy_bins else "unavailable"),
                occupied_track_bins=occupied_proxy_bins,
                free_track_bins=[],
                local_density=min(
                    sum(obj.bbox_dbu.area for obj in objects.values())
                    / max(halo.area, 1), 1.0,
                ),
                confidence=(0.3 if occupied_proxy_bins else 0.0),
            ),
            candidate_history=list(old.candidate_history) if old else [],
            rollback_history=list(old.rollback_history) if old else [],
            derived_features=RegionFeatures(marker_count=len(violations), object_count=len(objects),
                                            token_estimate=len(violations) * 180 + len(objects) * 80,
                                            geometry_signature=stable_hash(content)),
            quality_flags=flags, content_hash=stable_hash(content),
        )

    def build(self, violations: list[ViolationRecord], objects: list[LayoutObject],
              rule_catalog: RuleCatalog, previous: list[RegionState] | None = None) -> RegionBuildResult:
        previous = previous or []
        seeds = {v.violation_id: self._seed(v, self._associate(v, objects), rule_catalog)
                 for v in sorted(violations, key=lambda item: item.violation_id)}
        uf = _UnionFind(list(seeds))
        edges = []
        ordered = list(seeds.values())
        for index, a in enumerate(ordered):
            for b in ordered[index + 1:]:
                if a.halo.gap(b.halo) > self.cfg.affinity_max_gap_dbu:
                    continue
                hard, merge, evidence = self._evidence(a, b, self.cfg.affinity_max_gap_dbu)
                if hard:
                    uf.union(a.violation.violation_id, b.violation.violation_id)
                edges.append({"u": a.violation.violation_id, "v": b.violation.violation_id,
                              "hard": hard, "merge": merge, "evidence": evidence.model_dump()})
        for edge in sorted(edges, key=lambda item: (-item["evidence"]["halo_overlap_ratio"],
                                                     -item["evidence"]["normalized_gap_score"], item["u"], item["v"])):
            if not edge["merge"] or uf.find(edge["u"]) == uf.find(edge["v"]):
                continue
            members = [seeds[item] for item in seeds if uf.find(item) in {uf.find(edge["u"]), uf.find(edge["v"])}]
            if self._cap_ok(members):
                uf.union(edge["u"], edge["v"])
        groups = []
        for ids in uf.groups():
            groups.extend(self._split([seeds[item] for item in ids]))
        regions = sorted([self._finalize(group, previous) for group in groups], key=lambda item: item.region_id)
        ownership = {violation_id: region.region_id for region in regions for violation_id in region.violation_ids}
        if set(ownership) != {v.violation_id for v in violations}:
            raise AssertionError("violation coverage invariant failed")
        return RegionBuildResult(violations=violations, regions=regions, affinity_edges=edges,
                                 violation_to_region=ownership)

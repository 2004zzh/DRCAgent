from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field

from drc_agent.config.loader import AppConfig
from drc_agent.regions.builder import RegionBuilder
from drc_agent.regions.connectivity import map_connectivity_components
from drc_agent.regions.parser import (
    SourceObjectMapper, ViolationParser, load_rule_catalog,
)
from drc_agent.regions.physical import SourceHierarchyPhysicalBuilder
from drc_agent.regions.topology import LocalTopologyBuilder
from drc_agent.rules import RulePredicateRegistry, RuleWitnessBuilder
from drc_agent.schemas.common import StrictModel, stable_hash

from .models import RootTask, SearchSnapshot
from .snapshot import validate_snapshot_artifacts


class NodeContext(StrictModel):
    schema_version: str = "1.0"
    context_id: str
    snapshot_id: str
    case_id: str
    active_rule_ids: list[str]
    violation_count: int = Field(ge=0)
    source_object_count: int = Field(ge=0)
    physical_geometry_count: int = Field(ge=0)
    witness_count: int = Field(ge=0)
    region_count: int = Field(ge=0)
    local_topology_fingerprint: str
    source_map_fingerprint: str
    root_target_present: bool
    rebuilt_from_current_snapshot: bool = True
    artifact_hashes: dict[str, str]


class NodeContextBuilder:
    """Reparse current node artifacts through existing deterministic builders."""

    def __init__(self, project_root: Path, config: AppConfig):
        self.project_root = project_root.resolve()
        self.config = config
        self.catalog = load_rule_catalog(
            self.project_root / "configs/rules/asap7_rule_catalog.yaml"
        )

    def build(self, task: RootTask, snapshot: SearchSnapshot) -> NodeContext:
        validate_snapshot_artifacts(snapshot)
        script = Path(snapshot.script_ref)
        drc = Path(snapshot.drc_ref)
        violations = ViolationParser().parse_dac26_json(
            drc, case_id=task.case_id, rule_catalog=self.catalog,
        )
        objects = SourceObjectMapper().build_map(script)
        objects, _ = map_connectivity_components(
            objects, Path(snapshot.connectivity_reference_ref),
        )
        registry = RulePredicateRegistry(
            Path(task.rule_deck_path), require_frozen_hash=True,
        )
        witness_builder = RuleWitnessBuilder(registry)
        physical_by_id = {}
        witness_ids: list[str] = []
        for rule_id in sorted({item.rule_id for item in violations}):
            try:
                pack = registry.get(rule_id)
            except (KeyError, ValueError):
                continue
            physical = SourceHierarchyPhysicalBuilder().build(
                script, objects,
                layer_filter=set(pack.predicate.involved_layers),
            )
            physical_by_id.update({item.geometry_id: item for item in physical})
            for violation in violations:
                if violation.rule_id != rule_id:
                    continue
                try:
                    witness = witness_builder.build(
                        violation, physical, objects,
                    )
                except (KeyError, ValueError):
                    continue
                witness_ids.append(witness.witness_id)
        regions = RegionBuilder(self.config.region).build(
            violations, objects, self.catalog,
        ).regions
        topology: dict[str, Any] = LocalTopologyBuilder(
            dbu_per_um=self.config.region.dbu_per_um,
        ).build(violations, objects).model_dump(mode="json")
        source_payload = [
            (item.object_id, item.geometry_hash, item.source_anchor_id)
            for item in sorted(objects, key=lambda value: value.object_id)
        ]
        material = {
            "snapshot": snapshot.state_fingerprint,
            "violations": sorted(item.violation_id for item in violations),
            "objects": source_payload,
            "physical": sorted(physical_by_id),
            "witnesses": sorted(witness_ids),
            "topology": topology,
        }
        return NodeContext(
            context_id="p4_context_" + stable_hash(material)[:20],
            snapshot_id=snapshot.snapshot_id,
            case_id=task.case_id,
            active_rule_ids=sorted({item.rule_id for item in violations}),
            violation_count=len(violations),
            source_object_count=len(objects),
            physical_geometry_count=len(physical_by_id),
            witness_count=len(witness_ids),
            region_count=len(regions),
            local_topology_fingerprint=stable_hash(topology),
            source_map_fingerprint=stable_hash(source_payload),
            root_target_present=snapshot.root_target_present,
            artifact_hashes={
                "script": snapshot.script_sha256,
                "gds": snapshot.gds_sha256,
                "drc": snapshot.drc_sha256,
            },
        )


from __future__ import annotations

import fcntl
import json
import os
from collections import Counter
from pathlib import Path
from typing import Literal
from pydantic_core import PydanticSerializationError

from pydantic import Field

from drc_agent.schemas.common import StrictModel
from drc_agent.schemas.state import LayoutObject, RegionState
from drc_agent.schemas.workflow import IterationTransactionStatus


class DRCStatistics(StrictModel):
    total_drv: int
    rules_violated: int
    drc_by_rule: dict[str, int]
    drc_by_marker_type: dict[str, int]
    drc_by_rule_and_type: dict[str, dict[str, int]]


class MappingCoverage(StrictModel):
    object_count: int
    source_anchor_known: int
    source_anchor_unknown: int
    source_span_known: int
    geometry_known: int
    net_known: int
    net_unknown: int
    resource_mapping_source: Literal["exact", "proxy", "unavailable"]
    layer_histogram: dict[str, int]
    editability_histogram: dict[str, int]


def mapping_coverage(objects: list[LayoutObject]) -> MappingCoverage:
    layers = Counter(item.layer for item in objects)
    editability = Counter(item.editability.value for item in objects)
    anchor_known = sum(bool(item.source_anchor_id) for item in objects)
    net_known = sum(bool(item.net_id) for item in objects)
    return MappingCoverage(
        object_count=len(objects),
        source_anchor_known=anchor_known,
        source_anchor_unknown=len(objects) - anchor_known,
        source_span_known=sum(item.source_span is not None for item in objects),
        geometry_known=sum(bool(item.geometry_dbu) for item in objects),
        net_known=net_known,
        net_unknown=len(objects) - net_known,
        resource_mapping_source="unavailable",
        layer_histogram=dict(sorted(layers.items())),
        editability_histogram=dict(sorted(editability.items())),
    )


class PhysicalGroundingCoverage(StrictModel):
    source_object_count: int
    flattened_geometry_count: int
    exact_top_level_geometry_count: int
    exact_shared_cell_geometry_count: int
    editable_flattened_geometry_count: int
    witness_count: int
    exact_flattened_witness_count: int
    source_exact_witness_count: int
    inferred_witness_count: int
    unavailable_witness_count: int
    exact_signoff_predicate_witness_count: int = 0
    exact_geometry_approx_predicate_witness_count: int = 0
    source_geometry_approx_predicate_witness_count: int = 0
    proxy_predicate_witness_count: int = 0
    unavailable_predicate_witness_count: int = 0
    predicate_satisfied_count: int
    fabricated_net_id_count: int = 0
    resource_capacity_claimed: bool = False


def physical_grounding_coverage(objects, geometries, witnesses) -> PhysicalGroundingCoverage:
    quality = Counter(
        (item.get("evidence_quality") if isinstance(item, dict) else item.evidence_quality.value)
        for item in witnesses
    )
    fidelity = Counter(
        (
            item.get("predicate_fidelity", "UNAVAILABLE")
            if isinstance(item, dict) else item.predicate_fidelity.value
        )
        for item in witnesses
    )
    ownership = Counter(
        (item.get("ownership_quality") if isinstance(item, dict) else item.ownership_quality)
        for item in geometries
    )
    return PhysicalGroundingCoverage(
        source_object_count=len(objects),
        flattened_geometry_count=len(geometries),
        exact_top_level_geometry_count=ownership["EXACT_TOP_LEVEL"],
        exact_shared_cell_geometry_count=ownership["EXACT_SHARED_CELL"],
        editable_flattened_geometry_count=sum(
            bool(item.get("editable")) if isinstance(item, dict) else item.editable
            for item in geometries
        ),
        witness_count=len(witnesses),
        exact_flattened_witness_count=quality["EXACT_FLATTENED"],
        source_exact_witness_count=quality["SOURCE_EXACT"],
        inferred_witness_count=quality["GEOMETRIC_INFERRED"] + quality["MARKER_PROXY"],
        unavailable_witness_count=quality["UNAVAILABLE"],
        exact_signoff_predicate_witness_count=(
            fidelity["EXACT_SIGNOFF_PREDICATE"]
        ),
        exact_geometry_approx_predicate_witness_count=(
            fidelity["EXACT_GEOMETRY_APPROX_PREDICATE"]
        ),
        source_geometry_approx_predicate_witness_count=(
            fidelity["SOURCE_GEOMETRY_APPROX_PREDICATE"]
        ),
        proxy_predicate_witness_count=fidelity["PROXY"],
        unavailable_predicate_witness_count=fidelity["UNAVAILABLE"],
        predicate_satisfied_count=sum(
            bool(item.get("predicate_satisfied")) if isinstance(item, dict)
            else item.predicate_satisfied for item in witnesses
        ),
    )


class GraphObservabilityReport(StrictModel):
    object_count: int
    region_count: int
    named_net_object_count: int
    connectivity_component_object_count: int
    unavailable_net_object_count: int
    net_evidence_level: Literal[
        "named_exact", "connectivity_component", "mixed", "unavailable"
    ]
    exact_resource_region_count: int
    proxy_resource_region_count: int
    unavailable_resource_region_count: int
    resource_evidence_level: Literal["exact", "proxy", "mixed", "unavailable"]
    resource_capacity_known: bool = False
    fabricated_named_net_count: int = 0


def graph_observability_report(
    objects: list[LayoutObject], regions: list[RegionState],
) -> GraphObservabilityReport:
    named = sum(
        item.net_id is not None and item.net_mapping_quality == "exact"
        for item in objects
    )
    components = sum(
        item.connectivity_component_id is not None
        and item.net_mapping_quality == "connectivity_component"
        for item in objects
    )
    if named and components:
        net_level = "mixed"
    elif named:
        net_level = "named_exact"
    elif components:
        net_level = "connectivity_component"
    else:
        net_level = "unavailable"
    exact_resource = sum(
        item.resource_context.source == "true_routing_grid" for item in regions
    )
    proxy_resource = sum(
        item.resource_context.source == "geometry_proxy" for item in regions
    )
    if exact_resource and proxy_resource:
        resource_level = "mixed"
    elif exact_resource:
        resource_level = "exact"
    elif proxy_resource:
        resource_level = "proxy"
    else:
        resource_level = "unavailable"
    return GraphObservabilityReport(
        object_count=len(objects), region_count=len(regions),
        named_net_object_count=named,
        connectivity_component_object_count=components,
        unavailable_net_object_count=len(objects) - named - components,
        net_evidence_level=net_level,
        exact_resource_region_count=exact_resource,
        proxy_resource_region_count=proxy_resource,
        unavailable_resource_region_count=(
            len(regions) - exact_resource - proxy_resource
        ),
        resource_evidence_level=resource_level,
        resource_capacity_known=bool(exact_resource),
    )


class IterationMetrics(StrictModel):
    iteration: int
    metrics_semantics_version: Literal["1.0", "2.0"] = "1.0"
    transaction_status: IterationTransactionStatus
    committed_total_drv: int
    attempted_total_drv: int
    removed_original_drv: int = 0
    new_drv: int | None = 0
    connectivity_preserved: bool | None
    drc_by_rule: dict[str, int]
    drc_by_marker_type: dict[str, int]
    drc_by_rule_and_type: dict[str, dict[str, int]]
    attempted_drc_by_rule: dict[str, int] | None = None
    attempted_drc_by_marker_type: dict[str, int] | None = None
    attempted_drc_by_rule_and_type: dict[str, dict[str, int]] | None = None
    region_count: int = 0
    window_count: int = 0
    active_subgraph_count: int = 0
    candidate_count: int = 0
    non_noop_candidate_count: int = 0
    selected_non_noop_count: int = 0
    rollback_reason: str | None = None
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    reasoning_tokens: int = 0
    tool_runtime_seconds: float = 0.0
    raw_agent_edge_count: int = 0
    pruned_agent_edge_count: int = 0
    hard_component_count: int = 0
    largest_hard_component_size: int = 0
    planning_view_count: int = 0
    boundary_dependency_count: int = 0
    candidate_graph_edge_count: int = 0
    candidate_graph_completeness_milli: int = 1000
    unresolved_candidate_pair_count: int = 0
    sandbox_trial_count: int = 0
    sandbox_clean_progress_count: int = 0
    sandbox_cache_hit_count: int = 0
    selected_sandbox_verified_count: int = 0
    selected_proven_count: int = 0
    selected_heuristic_count: int = 0
    transaction_batch_count: int = 0
    deferred_batch_count: int = 0
    attribution_quality: str | None = None
    artifact_paths: dict[str, str] = Field(default_factory=dict)


def drc_statistics(source: Path | dict) -> DRCStatistics:
    data = json.loads(source.read_text(encoding="utf-8")) if isinstance(source, Path) else source
    by_rule: Counter[str] = Counter()
    by_type: Counter[str] = Counter()
    by_rule_type: dict[str, Counter[str]] = {}
    for rule_id, rule_data in sorted((data.get("rules") or {}).items()):
        values = rule_data.get("violations") or []
        declared = rule_data.get("violation_count")
        if declared is not None and int(declared) != len(values):
            raise ValueError(f"rule count mismatch for {rule_id}: {declared} != {len(values)}")
        by_rule[rule_id] += len(values)
        local = by_rule_type.setdefault(rule_id, Counter())
        for violation in values:
            marker_type = str(violation.get("type") or "unknown")
            by_type[marker_type] += 1
            local[marker_type] += 1
    total = sum(by_rule.values())
    if data.get("total_violations") is not None and int(data["total_violations"]) != total:
        raise ValueError(
            f"report total mismatch: {data['total_violations']} != {total}"
        )
    return DRCStatistics(
        total_drv=total,
        rules_violated=sum(count > 0 for count in by_rule.values()),
        drc_by_rule=dict(sorted(by_rule.items())),
        drc_by_marker_type=dict(sorted(by_type.items())),
        drc_by_rule_and_type={
            rule: dict(sorted(counts.items())) for rule, counts in sorted(by_rule_type.items())
        },
    )


class IterationMetricsStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, metrics: IterationMetrics) -> None:
        line = metrics.model_dump_json() + "\n"
        with self.path.open("a", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def read(self) -> list[IterationMetrics]:
        if not self.path.is_file():
            return []
        rows = [
            IterationMetrics.model_validate_json(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        by_iteration = {row.iteration: row for row in rows}
        return [by_iteration[key] for key in sorted(by_iteration)]


def persist_iteration_metrics(
    metrics: IterationMetrics,
    *,
    store: IterationMetricsStore,
    artifact_writer=None,
    artifact_path: str | None = None,
    required: bool = False,
) -> tuple[bool, str | None]:
    """Persist observability without making it repair-loop authority."""
    try:
        store.append(metrics)
        if artifact_writer is not None and artifact_path is not None:
            artifact_writer(
                artifact_path, metrics, producer="commit_or_rollback",
                schema_name="IterationMetrics",
            )
    except (OSError, TypeError, ValueError, PydanticSerializationError) as exc:
        if required:
            raise
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


def llm_usage_for_iteration(path: Path, iteration: int) -> dict[str, int]:
    result = {
        "llm_calls": 0, "input_tokens": 0,
        "output_tokens": 0, "cache_read_tokens": 0,
        "reasoning_tokens": 0,
    }
    if not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if int(record["context"]["iteration"]) != iteration:
            continue
        result["llm_calls"] += 1
        usage = record.get("usage") or {}
        result["input_tokens"] += int(usage.get("input_tokens") or 0)
        result["output_tokens"] += int(usage.get("output_tokens") or 0)
        result["cache_read_tokens"] += int(usage.get("cache_read_tokens") or 0)
        result["reasoning_tokens"] += int(usage.get("reasoning_tokens") or 0)
    return result

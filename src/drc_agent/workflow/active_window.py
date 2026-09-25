from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from drc_agent.config.loader import AppConfig
from drc_agent.coordinator.batching import TransactionBatchScheduler
from drc_agent.coordinator.sandbox import has_complete_physical_truth
from drc_agent.coordinator.solver_v2 import GraphCoordinatorV2
from drc_agent.graphs.candidate import CandidateGraphBuilder
from drc_agent.patching.compiler import PatchCompiler
from drc_agent.patching.repair_program import build_physical_effect_fingerprint
from drc_agent.reliability import FailureCode, InfrastructureFailure
from drc_agent.schemas.action import (
    CandidateGraph, JointRepairBundle, PatchPlan, RepairCandidate,
    SandboxEvidence, SandboxScope, SandboxStatus,
)
from drc_agent.schemas.common import stable_hash
from drc_agent.schemas.state import AgentSubgraph, LayoutObject
from drc_agent.schemas.workflow import (
    ActiveWindow, DesignSnapshotRef, HelperRegionAssignment, TransactionBatch,
)


def _scope_ids(subgraph: AgentSubgraph) -> set[str]:
    return {
        value for value in (
            subgraph.subgraph_id, subgraph.physical_scope_id,
            subgraph.solver_scope_id, subgraph.planning_view_id,
        ) if value
    }


def _dependency_pairs(graph: Any, subgraphs: list[AgentSubgraph]):
    owner = {
        region_id: subgraph.subgraph_id
        for subgraph in subgraphs for region_id in subgraph.region_ids
    }
    scope_owner = {
        scope_id: subgraph.subgraph_id
        for subgraph in subgraphs for scope_id in _scope_ids(subgraph)
    }
    hard: defaultdict[str, set[str]] = defaultdict(set)
    soft: defaultdict[str, set[str]] = defaultdict(set)
    dependency_ids: defaultdict[frozenset[str], set[str]] = defaultdict(set)
    for edge in graph.edges:
        left, right = owner.get(edge.u), owner.get(edge.v)
        if not left or not right or left == right:
            continue
        target = hard if edge.hard else soft
        target[left].add(right)
        target[right].add(left)
    for dependency in graph.boundary_dependencies:
        left = scope_owner.get(dependency.u_scope_id)
        right = scope_owner.get(dependency.v_scope_id)
        if not left or not right or left == right:
            continue
        target = hard if dependency.hard else soft
        target[left].add(right)
        target[right].add(left)
        dependency_ids[frozenset((left, right))].add(
            dependency.dependency_id
        )
    for left in subgraphs:
        for right in subgraphs:
            if left.subgraph_id >= right.subgraph_id:
                continue
            if (
                left.solver_scope_id
                and left.solver_scope_id == right.solver_scope_id
            ):
                hard[left.subgraph_id].add(right.subgraph_id)
                hard[right.subgraph_id].add(left.subgraph_id)
    return hard, soft, dependency_ids


_HARD_PHYSICAL_EVIDENCE_KINDS = {
    "same_writable_occurrence",
    "same_writable_source_target",
    "shared_merged_boundary_contributor",
    "potential_write_protected_read",
    "shared_protected_landing_contact",
}


def classify_window_region_roles(
    graph: Any, region_ids: list[str], frontier_violation_ids: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """Partition a full physical scope without discarding context Regions."""
    region_set = set(region_ids)
    frontier = set(frontier_violation_ids)
    decision = {
        region_id for region_id in region_set
        if frontier.intersection(graph.regions[region_id].violation_ids)
    }
    physical_adjacency: defaultdict[str, set[str]] = defaultdict(set)
    for edge in graph.edges:
        if not edge.hard or edge.u not in region_set or edge.v not in region_set:
            continue
        has_physical_proof = any(
            item.kind in _HARD_PHYSICAL_EVIDENCE_KINDS
            or (
                item.kind == "shared_editable_object"
                and bool(item.details.get("hard_by_occurrence"))
            )
            for item in edge.evidence
        )
        if not has_physical_proof:
            continue
        physical_adjacency[edge.u].add(edge.v)
        physical_adjacency[edge.v].add(edge.u)
    reachable = set(decision)
    queue = deque(sorted(decision))
    while queue:
        current = queue.popleft()
        for neighbor in sorted(physical_adjacency[current]):
            if neighbor not in reachable:
                reachable.add(neighbor)
                queue.append(neighbor)
    helper = reachable - decision
    context_only = region_set - decision - helper
    return sorted(decision), sorted(helper), sorted(context_only)


def build_helper_assignments(
    graph: Any, *, helper_region_ids: list[str],
    decision_region_ids: list[str], frontier_violation_ids: list[str],
    snapshot_id: str,
) -> list[HelperRegionAssignment]:
    """Bind each helper to reachable current targets through proven hard edges."""
    helpers = set(helper_region_ids)
    decisions = set(decision_region_ids)
    frontier = set(frontier_violation_ids)
    adjacency: defaultdict[str, list[tuple[str, list[str]]]] = defaultdict(list)
    for edge in graph.edges:
        if not edge.hard:
            continue
        evidence_ids = [
            item.evidence_id for item in edge.evidence
            if item.kind in _HARD_PHYSICAL_EVIDENCE_KINDS
            or (
                item.kind == "shared_editable_object"
                and bool(item.details.get("hard_by_occurrence"))
            )
        ]
        if not evidence_ids:
            continue
        adjacency[edge.u].append((edge.v, evidence_ids))
        adjacency[edge.v].append((edge.u, evidence_ids))

    assignments = []
    for helper in sorted(helpers):
        queue = deque([(helper, [])])
        visited = {helper}
        assisted: set[str] = set()
        dependency_evidence: set[str] = set()
        while queue:
            current, path_evidence = queue.popleft()
            if current in decisions:
                assisted.add(current)
                dependency_evidence.update(path_evidence)
                continue
            for neighbor, edge_evidence in sorted(adjacency[current]):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append((neighbor, path_evidence + edge_evidence))
        targets = sorted({
            violation_id for region_id in assisted
            for violation_id in graph.regions[region_id].violation_ids
            if violation_id in frontier
        })
        if not targets or not dependency_evidence:
            continue
        summary = graph.potential_access_summaries.get(helper)
        status = "DEPENDENCY_ASSIST"
        if (
            summary is not None
            and summary.schema_version == "2.0"
            and not summary.potential_dof_capability_ids
            and not summary.potential_carrier_ids
        ):
            # Keep the dependency in scope, but do not buy an LLM call when
            # deterministic current evidence proves that no helper mutation
            # carrier is available.
            status = "UNSUPPORTED_WITH_EVIDENCE"
        assignments.append(HelperRegionAssignment(
            helper_region_id=helper,
            assisted_region_ids=sorted(assisted),
            assisted_target_violation_ids=targets,
            dependency_evidence_ids=sorted(dependency_evidence),
            helper_potential_dof_capability_ids=(
                summary.potential_dof_capability_ids if summary else []
            ),
            helper_potential_carrier_ids=(
                summary.potential_carrier_ids if summary else []
            ),
            helper_protected_relation_ids=(
                summary.protected_relation_ids if summary else []
            ),
            snapshot_id=snapshot_id,
            status=status,
        ))
    return assignments


def engineering_scope_views(graph, subgraphs, target_ids: set[str]):
    """Explicit engineering-only planning restriction, not physical cropping.

    Preserve the original graph and every hard/boundary/hierarchical scope.
    Production full profiles never call this function.
    """
    if not target_ids:
        return subgraphs
    selected={r.region_id for r in graph.regions.values() if target_ids.intersection(r.violation_ids)}
    while True:
        expanded=selected | {node for e in graph.edges if e.hard and
            (e.u in selected or e.v in selected) for node in (e.u,e.v)}
        for scope in subgraphs:
            if expanded.intersection(scope.region_ids) and (
                scope.hierarchical or scope.scope_kind!="FLAT" or scope.boundary_dependency_ids):
                expanded.update(scope.region_ids)
        hard,_,_=_dependency_pairs(graph,subgraphs)
        owners={s.subgraph_id:s for s in subgraphs}
        for scope in subgraphs:
            if expanded.intersection(scope.region_ids):
                for neighbor in hard[scope.subgraph_id]:
                    expanded.update(owners[neighbor].region_ids)
        if expanded==selected:
            break
        selected=expanded
    views=[]
    for scope in subgraphs:
        ids=sorted(selected.intersection(scope.region_ids))
        if ids:
            views.append(scope.model_copy(update={"region_ids":ids,
                "edge_ids":[e.edge_id for e in graph.edges if e.edge_id in scope.edge_ids
                    and e.u in ids and e.v in ids]}))
    return views


def select_active_window(
    *, graph: Any, subgraphs: list[AgentSubgraph], iteration: int,
    window_index: int, base_snapshot_id: str,
    iteration_frontier_violation_ids: set[str],
    processed_frontier_violation_ids: set[str],
    current_violation_ids: set[str], max_parallel_subgraphs: int,
    max_active_regions: int, top_k_candidates_including_noop: int,
    max_non_noop_actions_per_batch: int,
    deferred_frontier_violation_ids: set[str] | None = None,
) -> ActiveWindow | None:
    """Select one dependency-connected window from the current graph."""
    # Deferred means not admitted/attempted yet, not exhausted for the entire
    # iteration. Only truly attempted targets enter processed_frontier.
    pending = (
        iteration_frontier_violation_ids
        - processed_frontier_violation_ids
    ) & current_violation_ids
    if not pending:
        return None
    by_id = {item.subgraph_id: item for item in subgraphs}
    claims = {
        item.subgraph_id: sorted({
            violation_id
            for region_id in item.region_ids
            for violation_id in graph.regions[region_id].violation_ids
            if violation_id in pending
        })
        for item in subgraphs
    }
    eligible = [item for item in subgraphs if claims[item.subgraph_id]]
    if not eligible:
        return None
    hard, soft, dependency_ids = _dependency_pairs(graph, subgraphs)

    def hard_degree(item: AgentSubgraph) -> int:
        return len(hard[item.subgraph_id]) + sum(
            edge.hard and (
                edge.u in item.region_ids or edge.v in item.region_ids
            ) for edge in graph.edges
        )

    seed = min(eligible, key=lambda item: (
        -item.priority, -hard_degree(item),
        -len(claims[item.subgraph_id]), len(item.region_ids),
        item.subgraph_id,
    ))
    selected = {seed.subgraph_id}
    queue = deque([seed.subgraph_id])
    while queue:
        current = queue.popleft()
        for dependent in sorted(hard[current]):
            if dependent in by_id and dependent not in selected:
                selected.add(dependent)
                queue.append(dependent)

    per_region_nonnoop = max(top_k_candidates_including_noop - 1, 0)
    candidate_soft_budget = (
        max_non_noop_actions_per_batch * top_k_candidates_including_noop
    )

    def region_count(ids: set[str]) -> int:
        return len({
            region_id for identifier in ids
            for region_id in by_id[identifier].region_ids
        })

    def estimate(ids: set[str]) -> int:
        return region_count(ids) * per_region_nonnoop

    candidates = set().union(*(soft[item] for item in selected)) - selected
    while candidates:
        ordered = sorted(
            (by_id[item] for item in candidates if item in by_id),
            key=lambda item: (
                -item.priority, -len(claims[item.subgraph_id]),
                len(item.region_ids), item.subgraph_id,
            ),
        )
        added = None
        for item in ordered:
            proposal = selected | {item.subgraph_id}
            if len(proposal) > max_parallel_subgraphs:
                continue
            if region_count(proposal) > max_active_regions:
                continue
            if estimate(proposal) > candidate_soft_budget:
                continue
            added = item.subgraph_id
            break
        if added is None:
            break
        selected.add(added)
        candidates = (
            set().union(*(soft[item] for item in selected)) - selected
        )

    chosen = [by_id[item] for item in sorted(selected)]
    region_ids = sorted({
        region_id for item in chosen for region_id in item.region_ids
    })
    frontier = sorted({
        violation_id for item in chosen
        for violation_id in claims[item.subgraph_id]
    })
    boundary_ids = sorted(set().union(*(
        dependency_ids[frozenset((left, right))]
        for left in selected for right in selected if left < right
    ))) if len(selected) > 1 else []
    decision, helper, context_only = classify_window_region_roles(
        graph, region_ids, frontier,
    )
    helper_assignments = build_helper_assignments(
        graph, helper_region_ids=helper, decision_region_ids=decision,
        frontier_violation_ids=frontier, snapshot_id=base_snapshot_id,
    )
    assigned_helpers = {
        item.helper_region_id for item in helper_assignments
    }
    # A Region without a current target binding is context, not a paid helper.
    context_only = sorted(set(context_only) | (set(helper) - assigned_helpers))
    helper = sorted(assigned_helpers)
    identity = [
        iteration, window_index, base_snapshot_id,
        sorted(selected), frontier, decision, helper, context_only,
        [item.model_dump(mode="json") for item in helper_assignments],
    ]
    return ActiveWindow(
        window_id="window_" + stable_hash(identity)[:20],
        iteration=iteration, window_index=window_index,
        base_snapshot_id=base_snapshot_id,
        subgraph_ids=sorted(selected), region_ids=region_ids,
        frontier_violation_ids=frontier,
        claimed_frontier_violation_ids=frontier,
        decision_region_ids=decision,
        helper_region_ids=helper,
        helper_assignments=helper_assignments,
        context_only_region_ids=context_only,
        boundary_dependency_ids=boundary_ids,
        hard_dependency_closure=True,
    )


@dataclass
class WindowCoordination:
    status: str
    candidate_graph: CandidateGraph
    bundle: JointRepairBundle | None
    patch_plan: PatchPlan | None
    sandbox_evidence: SandboxEvidence | None
    master_batches: list[TransactionBatch]
    rounds: int
    no_good_count: int
    sandbox_jobs_used: int
    evidence_reused: bool = False
    failure_codes: list[str] | None = None
    selection_cardinality_caps_tried: list[int] | None = None
    selected_non_noop_counts: list[int] | None = None
    backoff_count: int = 0


def exact_isolated_evidence_matches(
    *, candidate: RepairCandidate, snapshot: DesignSnapshotRef,
    patch_plan: PatchPlan, rule_deck_sha256: str, evaluator_sha256: str,
) -> bool:
    evidence = candidate.benefit_evidence.sandbox
    if evidence is None:
        return False
    return all((
        has_complete_physical_truth(evidence),
        evidence.status == SandboxStatus.CLEAN_PROGRESS,
        evidence.base_snapshot_id == snapshot.snapshot_id,
        evidence.scope.value == "ISOLATED_CANDIDATE",
        evidence.candidate_ids == [candidate.candidate_id],
        evidence.base_script_sha256 == snapshot.script_ref.sha256,
        evidence.physical_effect_fingerprint == (
            build_physical_effect_fingerprint(candidate).sha256
        ),
        evidence.patch_plan_sha256 == stable_hash(
            patch_plan.model_dump(mode="json")
        ),
        evidence.rule_deck_sha256 == rule_deck_sha256,
        evidence.evaluator_sha256 == evaluator_sha256,
        evidence.connectivity_reference_sha256 == (
            snapshot.connectivity_ref.sha256
            if snapshot.connectivity_ref else None
        ),
        evidence.removed_original_count is not None
        and evidence.removed_original_count > 0,
        evidence.new_violation_count == 0,
        evidence.connectivity_preserved is True,
    ))


def coordinate_active_window(
    *, window: ActiveWindow, candidates: list[RepairCandidate],
    source_map: list[LayoutObject], script: Path,
    snapshot: DesignSnapshotRef, config: AppConfig,
    resource_capacities: dict[str, int],
    sandbox: Callable[[list[RepairCandidate], PatchPlan], SandboxEvidence],
    sandbox_jobs_remaining: int, rule_deck_sha256: str,
    evaluator_sha256: str,
) -> WindowCoordination:
    """Solve and exact-verify one window; failures become snapshot-local no-goods."""
    graph = CandidateGraphBuilder(
        interaction_margin_dbu=config.candidate_graph.interaction_margin_dbu,
        fail_on_incomplete_audit=config.candidate_graph.fail_on_incomplete_audit,
    ).build(
        candidates, window.window_id, resource_capacities=resource_capacities,
    )
    by_id = {item.candidate_id: item for item in candidates}
    no_goods = 0
    jobs = 0
    failures: list[str] = []
    caps_tried: list[int] = []
    selected_counts: list[int] = []
    backoff_count = 0
    same_cardinality_retried: set[int] = set()
    selection_cap = min(
        config.transaction.max_non_noop_actions_per_batch,
        config.candidate_evidence.joint_sandbox_max_bundle_size,
    )
    max_rounds = config.candidate_evidence.joint_sandbox_max_rounds
    if max_rounds <= 0:
        return WindowCoordination(
            "NO_SAFE_BUNDLE", graph, None, None, None, [], 0, 0, 0,
            failure_codes=["JOINT_SANDBOX_ROUNDS_EXHAUSTED"],
        )
    for round_index in range(1, max_rounds + 1):
        caps_tried.append(selection_cap)
        bundle = GraphCoordinatorV2().solve(
            graph, config.coordinator,
            max_non_noop_selected=selection_cap,
        )
        selected = [
            by_id[item] for item in bundle.selected_candidate_ids
            if item in by_id and not by_id[item].is_noop
        ]
        selected_counts.append(len(selected))
        if not selected:
            return WindowCoordination(
                "NO_OP", graph, bundle, None, None, [], round_index,
                no_goods, jobs, failure_codes=failures,
                selection_cardinality_caps_tried=caps_tried,
                selected_non_noop_counts=selected_counts,
                backoff_count=backoff_count,
            )
        modified = {
            object_id for candidate in selected
            for object_id in (
                candidate.affected_object_ids or candidate.target_object_ids
            )
        }
        if (
            len(selected) > config.transaction.max_non_noop_actions_per_batch
            or len(modified) > config.transaction.max_modified_objects_per_batch
        ):
            graph.no_good_candidate_sets.append(sorted(
                item.candidate_id for item in selected
            ))
            no_goods += 1
            failures.append("WINDOW_TRANSACTION_CAP_EXCEEDED")
            next_cap = min(selection_cap - 1, len(selected) - 1)
            if next_cap < 1:
                break
            selection_cap = next_cap
            backoff_count += 1
            continue
        try:
            patch_plan = PatchCompiler().compile(
                bundle, candidates, source_map, script,
            )
        except Exception as exc:
            graph.no_good_candidate_sets.append(sorted(
                item.candidate_id for item in selected
            ))
            no_goods += 1
            failures.append(
                "WINDOW_BUNDLE_COMPILE_FAIL:" + type(exc).__name__
            )
            next_cap = min(selection_cap - 1, len(selected) - 1)
            if next_cap < 1:
                break
            selection_cap = next_cap
            backoff_count += 1
            continue
        reused = bool(
            len(selected) == 1
            and config.repair_kernel_integration.reuse_verified_kernel_evidence
            and exact_isolated_evidence_matches(
                candidate=selected[0], snapshot=snapshot,
                patch_plan=patch_plan, rule_deck_sha256=rule_deck_sha256,
                evaluator_sha256=evaluator_sha256,
            )
        )
        if reused:
            evidence = selected[0].benefit_evidence.sandbox
        else:
            if jobs >= sandbox_jobs_remaining:
                failures.append("ITERATION_SANDBOX_BUDGET_EXHAUSTED")
                break
            try:
                evidence = sandbox(selected, patch_plan)
            except InfrastructureFailure:
                # Shared EDA/evidence failures are run-scoped.  They must trip
                # the runtime circuit breaker instead of becoming a local
                # no-safe-bundle result and triggering more paid work.
                raise
            except Exception as exc:
                failures.append(
                    "WINDOW_SANDBOX_INFRASTRUCTURE:" + type(exc).__name__
                )
                return WindowCoordination(
                    "NO_SAFE_BUNDLE", graph, None, None, None, [],
                    round_index, no_goods, jobs, failure_codes=failures,
                    selection_cardinality_caps_tried=caps_tried,
                    selected_non_noop_counts=selected_counts,
                    backoff_count=backoff_count,
                )
            jobs += 1
        if not has_complete_physical_truth(evidence):
            failures.append(
                evidence.failure_code or "WINDOW_SANDBOX_INFRASTRUCTURE"
            )
            return WindowCoordination(
                "NO_SAFE_BUNDLE", graph, None, None, evidence, [],
                round_index, no_goods, jobs, failure_codes=failures,
                selection_cardinality_caps_tried=caps_tried,
                selected_non_noop_counts=selected_counts,
                backoff_count=backoff_count,
            )
        expected_scope = (
            SandboxScope.JOINT_BUNDLE
            if len(selected) >= 2
            else SandboxScope.ISOLATED_CANDIDATE
        )
        expected_connectivity_sha256 = (
            snapshot.connectivity_ref.sha256
            if snapshot.connectivity_ref else None
        )
        identity_matches = (
            evidence.scope == expected_scope
            and evidence.base_snapshot_id == snapshot.snapshot_id
            and evidence.candidate_ids == sorted(c.candidate_id for c in selected)
            and evidence.patch_plan_sha256 == stable_hash(patch_plan.model_dump(mode="json"))
            and evidence.rule_deck_sha256 == rule_deck_sha256
            and evidence.evaluator_sha256 == evaluator_sha256
            and evidence.connectivity_reference_sha256 == expected_connectivity_sha256
        )
        if not identity_matches:
            raise InfrastructureFailure("WINDOW_EVIDENCE_IDENTITY_MISMATCH",
                failure_code=FailureCode.EVIDENCE_OWNERSHIP,retryable=False,
                provider="WindowCoordinator",model="KLayout",failure_stage="JOINT_EVIDENCE_IDENTITY")
        if (
            has_complete_physical_truth(evidence)
            and evidence.status == SandboxStatus.CLEAN_PROGRESS
            and evidence.scope == expected_scope
            and evidence.base_snapshot_id == snapshot.snapshot_id
            and evidence.candidate_ids == sorted(
                item.candidate_id for item in selected
            )
            and evidence.patch_plan_sha256 == stable_hash(
                patch_plan.model_dump(mode="json")
            )
            and evidence.rule_deck_sha256 == rule_deck_sha256
            and evidence.evaluator_sha256 == evaluator_sha256
            and evidence.connectivity_reference_sha256
            == expected_connectivity_sha256
            and evidence.removed_original_count > 0
            and evidence.new_violation_count == 0
            and evidence.connectivity_preserved is True
        ):
            verified = bundle.model_copy(update={
                "sandbox_status": SandboxStatus.CLEAN_PROGRESS,
                "sandbox_verification_ref": evidence.verification_ref,
            })
            batches = TransactionBatchScheduler().schedule(
                bundles=[verified], candidates=candidates,
                snapshot=snapshot, config=config.transaction,
            )
            selected_ids = sorted(item.candidate_id for item in selected)
            if len(batches) == 1 and sorted(batches[0].candidate_ids) == selected_ids:
                return WindowCoordination(
                    "SAFE", graph, verified, patch_plan, evidence, batches,
                    round_index, no_goods, jobs, reused, failures,
                    caps_tried, selected_counts, backoff_count,
                )
            failures.append("WINDOW_MASTER_BATCH_CARDINALITY")
        else:
            failures.append("WINDOW_JOINT_" + evidence.status.value)
        if has_complete_physical_truth(evidence):
            graph.no_good_candidate_sets.append(sorted(
                item.candidate_id for item in selected
            ))
            no_goods += 1
            cardinality = len(selected)
            rounds_remaining = max_rounds - round_index
            # Preserve enough of the existing cardinality backoff budget to
            # reach a singleton.  A same-cardinality alternative is admitted
            # only when the configured round budget has one genuinely spare
            # slot; it must not consume the only path to the old fallback.
            alternate_same_cardinality = False
            if (
                cardinality > 1
                and cardinality not in same_cardinality_retried
                and rounds_remaining >= cardinality
                and jobs < sandbox_jobs_remaining
            ):
                # Solving the updated graph is a zero-EDA preview.  Do not
                # spend a validation round when the exact-set no-good leaves
                # no genuinely different bundle of the same cardinality.
                alternate = GraphCoordinatorV2().solve(
                    graph,
                    config.coordinator,
                    max_non_noop_selected=cardinality,
                )
                alternate_same_cardinality = sum(
                    1
                    for candidate_id in alternate.selected_candidate_ids
                    if candidate_id in by_id and not by_id[candidate_id].is_noop
                ) == cardinality
            if alternate_same_cardinality:
                # A new exact-set no-good may leave another safe combination.
                # Same total round/EDA limits; never infer pairwise conflicts.
                same_cardinality_retried.add(cardinality)
                selection_cap = cardinality
                continue
        next_cap = min(selection_cap - 1, len(selected) - 1)
        if next_cap < 1:
            break
        selection_cap = next_cap
        backoff_count += 1
    return WindowCoordination(
        "NO_SAFE_BUNDLE", graph, None, None, None, [], len(caps_tried),
        no_goods, jobs, failure_codes=failures,
        selection_cardinality_caps_tried=caps_tried,
        selected_non_noop_counts=selected_counts,
        backoff_count=backoff_count,
    )

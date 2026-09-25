from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable

from pydantic import Field

from drc_agent.actions.candidates import CandidateChecker, make_noop_candidate
from drc_agent.actions.vocabulary import (
    normalize_action_names, validate_semantic_actions,
)
from drc_agent.agents import (
    CandidateIntentLowerer, CompactRegionContextSerializer, RegionAgent,
    RegionAgentResult,
)
from drc_agent.agents.repair_programmer import (
    LLMRepairProgrammer, RepairProgrammingController, SandboxPreviewTools,
)
from drc_agent.agents.repair_memory import (
    FAILURE_OUTCOMES, RepairAttemptMemory,
)
from drc_agent.agents.repair_scheduler import RepairabilityScheduler
from drc_agent.config.loader import AppConfig
from drc_agent.coordinator.sandbox import (
    TwoStageSandboxPolicy,
    has_complete_physical_truth,
)
from drc_agent.coordinator.solver import GraphCoordinator, GreedyCoordinator
from drc_agent.experience.blueprint import BlueprintBuilder
from drc_agent.experience.retrieval import ExperienceExpander, HybridRetriever
from drc_agent.experience.signature import ContextSignatureBuilder
from drc_agent.graphs.agent import AgentGraph, MessagePasser
from drc_agent.graphs.candidate import CandidateGraphBuilder
from drc_agent.llm import InfrastructureFailure, LLMClient
from drc_agent.reliability import IntegrityFailure
from drc_agent.patching.repair_program import (
    RepairProgramCompiler, build_physical_effect_fingerprint,
)
from drc_agent.schemas.action import (
    CandidateGraph, DesignState, JointRepairBundle, RepairCandidate,
    SandboxEvidence, SandboxStatus,
)
from drc_agent.schemas.common import ArtifactRef, StrictModel, stable_hash
from drc_agent.schemas.workflow import DesignSnapshotRef, HelperRegionAssignment
from drc_agent.schemas.repair_program import RepairProgrammingTrace
from drc_agent.schemas.experience import (
    ContextSignature, CoordinationRequirement, EvidencePack, ExperienceQuery, RepairBlueprint,
)
from drc_agent.schemas.state import (
    AgentSubgraph, IntentSummary, LayoutObject, NeighborMessage, RuleCatalog,
    ViolationRecord,
)
from drc_agent.agents.schemas import IntentFailure
from drc_agent.utils.artifacts import ArtifactStore
from drc_agent.repair_kernel_integration import FormalRepairKernel
from drc_agent.repair_kernel_integration.facade import PreparedKernelRegion
from drc_agent.repair_kernel_integration.models import KernelRegionResult


def region_worker_count(region_count: int, llm_max_concurrent_requests: int) -> int:
    """Bound Region coroutines by the provider limit, not graph parallelism."""
    return min(max(0, int(region_count)), max(1, int(llm_max_concurrent_requests)))


class PlannedSubgraph(StrictModel):
    subgraph: AgentSubgraph
    signature: ContextSignature
    evidence: EvidencePack
    blueprint: RepairBlueprint
    messages: list[NeighborMessage]
    region_results: list[RegionAgentResult]
    candidates: list[RepairCandidate]
    candidate_graph: CandidateGraph
    bundle: JointRepairBundle
    repair_programming_traces: list[RepairProgrammingTrace] = Field(
        default_factory=list
    )
    decision_region_ids: list[str] = Field(default_factory=list)
    helper_region_ids: list[str] = Field(default_factory=list)
    context_only_region_ids: list[str] = Field(default_factory=list)
    claimed_frontier_violation_ids: list[str] = Field(default_factory=list)
    admitted_frontier_violation_ids: list[str] = Field(default_factory=list)
    llm_started_frontier_violation_ids: list[str] = Field(default_factory=list)
    plan_validated_frontier_violation_ids: list[str] = Field(default_factory=list)
    kernel_attempted_frontier_violation_ids: list[str] = Field(default_factory=list)
    physical_evaluated_frontier_violation_ids: list[str] = Field(default_factory=list)
    attempted_frontier_violation_ids: list[str] = Field(default_factory=list)
    deferred_frontier_violation_ids: list[str] = Field(default_factory=list)
    unsupported_frontier_violation_ids: list[str] = Field(default_factory=list)
    helper_assignments: list[HelperRegionAssignment] = Field(default_factory=list)
    llm_invocation_count: int = 0
    artifact_refs: dict[str, ArtifactRef] = Field(default_factory=dict)
    integration_audit: dict[str, Any] = Field(default_factory=dict)


class SubgraphPlanner:
    def __init__(
        self, *, config: AppConfig, llm: LLMClient, experience_store,
        artifact_store: ArtifactStore, run_id: str,
        coordinator_mode: str = "cp_sat",
        candidate_sandbox: Callable[..., SandboxEvidence] | None = None,
        bundle_sandbox: Callable[..., SandboxEvidence] | None = None,
        event_sink: Callable[..., Any] | None = None,
        project_root: Path | None = None,
    ):
        self.config = config
        self.llm = llm
        self.experience_store = experience_store
        self.artifact_store = artifact_store
        self.run_id = run_id
        self.coordinator_mode = coordinator_mode
        self.candidate_sandbox = candidate_sandbox
        self.bundle_sandbox = bundle_sandbox
        self.event_sink = event_sink
        self.repair_attempt_memory = RepairAttemptMemory()
        integration = config.repair_kernel_integration
        self.formal_kernel = (
            FormalRepairKernel(
                llm=llm, event_sink=event_sink,
                max_candidates=min(
                    config.agent.top_k_candidates_including_noop - 1,
                    integration.max_plans_per_region,
                ),
                mode=integration.mode,
                project_root=project_root,
                config=config,
                artifact_root=artifact_store.root,
            )
            if integration.enabled else None
        )
        self.integration_audit = (
            self.formal_kernel.audit
            if self.formal_kernel is not None else None
        )
        self.region_agent = None if self.formal_kernel is not None else RegionAgent(
            llm=llm,
            context_serializer=CompactRegionContextSerializer(
                config.region.max_context_tokens,
                allowed_distances_dbu=config.agent.allowed_distances_dbu,
                max_candidate_distance_dbu=(
                    config.agent.max_candidate_distance_dbu
                ),
                dbu_per_um=config.region.dbu_per_um,
                manufacturing_grid_dbu=config.backend.manufacturing_grid_dbu,
            ),
            lowerer=CandidateIntentLowerer(config.agent),
            checker=CandidateChecker(),
            prompt_version=config.llm.prompt_version,
        )

    def _integration_event(self, event: str, **details: Any) -> None:
        if self.event_sink is not None:
            self.event_sink(event, **details)

    @staticmethod
    def _allowed_actions(region, catalog: RuleCatalog) -> set[str]:
        configured = {"NO_OP"}
        for rule_id in region.rule_ids:
            spec, _ = catalog.lookup(rule_id)
            configured.update(spec.allowed_action_families)
        values = set(normalize_action_names(configured))
        validate_semantic_actions(
            values, owner=f"Region[{region.region_id}] action space",
        )
        return values

    @staticmethod
    def _kernel_result_to_region_result(result: KernelRegionResult) -> RegionAgentResult:
        """Bridge the kernel result to the existing Graph Action result schema."""
        candidates = [
            item.model_dump(mode="json") for item in result.candidates
        ]
        if result.noop_candidate is not None:
            candidates.append(result.noop_candidate.model_dump(mode="json"))
        failures = [
            IntentFailure(
                intent_id=str(item.get("plan_id") or item.get("violation_id") or "kernel"),
                code=str(item.get("code", "KERNEL_FAILURE")),
                message=str(item.get("message", "kernel planning failed"))[:500],
                semantic_fingerprint=item.get("semantic_fingerprint"),
            )
            for item in result.failures
        ]
        path = (
            "REPAIR_KERNEL"
            if result.supported_rule_ids
            else "UNSUPPORTED_NOOP"
        )
        return RegionAgentResult(
            region_id=result.region_id,
            subgraph_id=result.subgraph_id,
            diagnosis=None,
            planning_path=path,
            candidates=candidates,
            failures=failures,
        )

    def _deterministic_sound_plan(
        self, *, region, subgraph_id: str,
        violations: list[ViolationRecord], objects: list[LayoutObject],
        neighbor_messages: list[NeighborMessage], blueprint: RepairBlueprint,
        design: DesignState, allowed_actions: set[str],
        catalog: RuleCatalog, script: Path,
    ) -> RegionAgentResult:
        context = self.region_agent.context_serializer.build(
            region=region, violations=violations, objects=objects,
            legal_layers=design.legal_layers,
            neighbor_messages=neighbor_messages, blueprint=blueprint,
            allowed_actions=allowed_actions, script=script,
            rule_catalog=catalog, local_topology=design.local_topology,
            rule_predicates=design.rule_predicates,
            rule_witnesses=design.rule_witnesses,
            rule_knowledge_packs=design.rule_knowledge_packs,
        )
        guidance = self.region_agent.guide.build(context)
        noop = make_noop_candidate(region, subgraph_id)
        violation_by_id = {
            item.violation_id: item for item in violations
            if item.violation_id in set(region.violation_ids)
        }
        candidates = [noop]
        failures: list[IntentFailure] = []
        selected_intents = []
        seen_intents = set()
        seen_candidates = set()
        max_non_noop = max(
            0, self.config.agent.top_k_candidates_including_noop - 1
        )
        for hypothesis in guidance.hypotheses:
            intent = hypothesis.intent
            if (
                intent.action_family not in allowed_actions
                or not set(intent.target_violation_ids) <= set(violation_by_id)
            ):
                continue
            intent_key = intent.model_dump_json(
                exclude={"intent_id", "confidence_milli"}
            )
            if intent_key in seen_intents:
                continue
            seen_intents.add(intent_key)
            selected_intents.append(intent)
            target_rules = {
                violation_by_id[item].rule_id
                for item in intent.target_violation_ids
            }
            target_layers = {
                layer for item in intent.target_violation_ids
                for layer in violation_by_id[item].layers
            }
            secondary = max([
                (spec.min_influence_nm * self.config.region.dbu_per_um + 999)
                // 1000
                for rule_id, spec in catalog.rules.items()
                if spec.family.value == "spacing"
                and rule_id not in target_rules
                and set(spec.primary_layers) & target_layers
                and spec.min_influence_nm is not None
            ] or [0])
            minimum = max([
                context.rule_distance_constraints_dbu.get(
                    violation_by_id[item].rule_id, 0
                )
                for item in intent.target_violation_ids
            ] or [0])
            lowered, lowering_failures = (
                self.region_agent.lowerer.lower_variants(
                    intent=intent, region=region, subgraph_id=subgraph_id,
                    design=design, rank=1,
                    failed_fingerprints={
                        item.semantic_fingerprint
                        for item in region.candidate_history
                        if (
                            item.outcome in FAILURE_OUTCOMES
                            or "FAIL" in item.outcome
                            or "ROLLBACK" in item.outcome
                        )
                    },
                    allowed_distances_dbu=set(
                        context.allowed_distance_candidates_dbu
                    ),
                    minimum_clearance_dbu=minimum,
                    secondary_clearance_dbu=int(secondary),
                    max_variants=(
                        self.config.agent.max_geometry_variants_per_hypothesis
                    ),
                )
            )
            for exc in lowering_failures:
                failures.append(IntentFailure(
                    intent_id=intent.intent_id, code=exc.code,
                    message=str(exc), semantic_fingerprint=exc.fingerprint,
                ))
            for value in lowered:
                if len(candidates) - 1 >= max_non_noop:
                    break
                if value.intent_fingerprint in seen_candidates:
                    continue
                seen_candidates.add(value.intent_fingerprint)
                candidate = value.candidate
                candidate.rank_from_agent = len(candidates)
                report = self.region_agent.checker.validate(
                    candidate, region, design, allowed_actions,
                    violation_by_id, rule_catalog=catalog,
                )
                if report.status.value in {
                    "VALID", "VALID_REQUIRES_SANDBOX",
                }:
                    candidates.append(candidate)
                else:
                    failures.append(IntentFailure(
                        intent_id=intent.intent_id,
                        code=(
                            report.issues[0].code if report.issues
                            else report.status.value
                        ),
                        message="; ".join(
                            item.message for item in report.issues
                        ),
                        semantic_fingerprint=value.intent_fingerprint,
                    ))
            if len(candidates) - 1 >= max_non_noop:
                break
        if not selected_intents:
            failures.append(IntentFailure(
                intent_id="deterministic_sound_path",
                code="DETERMINISTIC_SYNTHESIZER_UNSUPPORTED",
                message="reviewed witness has no sound deterministic affordance",
            ))
        return RegionAgentResult(
            region_id=region.region_id, subgraph_id=subgraph_id,
            diagnosis=None, planning_path="DETERMINISTIC_SOUND",
            repair_hypotheses=guidance.hypotheses,
            guidance_audit=guidance.audit,
            intents=selected_intents,
            candidates=[item.model_dump(mode="json") for item in candidates],
            failures=failures,
        )

    async def plan(
        self, *, iteration: int, subgraph: AgentSubgraph,
        graph: AgentGraph, violations: list[ViolationRecord],
        objects: list[LayoutObject], design: DesignState,
        catalog: RuleCatalog, script: Path,
        current_snapshot_id: str | None = None,
        current_snapshot: DesignSnapshotRef | None = None,
        baseline_snapshot: DesignSnapshotRef | None = None,
        rule_deck_path: Path | None = None,
        case_id: str | None = None,
        evaluator_hash: str | None = None,
        artifact_prefix: str | None = None,
        execution_epoch: str = "epoch_initial",
        window_id: str | None = None,
        window_index: int | None = None,
        defer_isolated_sandbox: bool = False,
        defer_final_joint_sandbox: bool = False,
        sandbox_job_limit: int | None = None,
        decision_region_ids: list[str] | None = None,
        helper_region_ids: list[str] | None = None,
        context_only_region_ids: list[str] | None = None,
        claimed_frontier_violation_ids: list[str] | None = None,
        helper_assignments: list[HelperRegionAssignment] | None = None,
    ) -> PlannedSubgraph:
        prefix = (
            f"{artifact_prefix}/planning/subgraphs/{subgraph.subgraph_id}"
            if artifact_prefix else
            f"iterations/iter_{iteration:04d}/planning/subgraphs/"
            f"{subgraph.subgraph_id}"
        )
        full_region_ids = set(subgraph.region_ids)
        roles_supplied = any(value is not None for value in (
            decision_region_ids, helper_region_ids, context_only_region_ids,
        ))
        if roles_supplied:
            decision_ids = set(decision_region_ids or ())
            helper_ids = set(helper_region_ids or ())
            context_ids = set(context_only_region_ids or ())
            if (
                decision_ids & helper_ids
                or decision_ids & context_ids
                or helper_ids & context_ids
                or decision_ids | helper_ids | context_ids != full_region_ids
            ):
                raise ValueError("REGION_ROLE_PARTITION_INVALID")
        else:
            # Historical direct callers retain the old all-Region behavior.
            decision_ids = set(full_region_ids)
            helper_ids = set()
            context_ids = set()
        claimed_frontier = set(claimed_frontier_violation_ids or ())
        if claimed_frontier_violation_ids is None:
            claimed_frontier = {
                violation_id for region_id in decision_ids
                for violation_id in graph.regions[region_id].violation_ids
            }
        if helper_assignments is None and helper_ids:
            from drc_agent.workflow.active_window import build_helper_assignments
            helper_assignments = build_helper_assignments(
                graph, helper_region_ids=sorted(helper_ids),
                decision_region_ids=sorted(decision_ids),
                frontier_violation_ids=sorted(claimed_frontier),
                snapshot_id=current_snapshot_id or "UNKNOWN_SNAPSHOT",
            )
        helper_assignments = list(helper_assignments or [])
        helper_assignment_by_region = {
            item.helper_region_id: item for item in helper_assignments
            if item.helper_region_id in helper_ids
        }
        assigned_helper_ids = set(helper_assignment_by_region)
        context_ids.update(helper_ids - assigned_helper_ids)
        helper_ids = assigned_helper_ids
        unsupported_helper_ids = {
            region_id for region_id, assignment
            in helper_assignment_by_region.items()
            if assignment.status == "UNSUPPORTED_WITH_EVIDENCE"
        }
        planning_region_ids = sorted(
            decision_ids | (helper_ids - unsupported_helper_ids)
        )

        # One deterministic target per paid invocation. A helper consumes the
        # target it is explicitly assisting; it never falls back to one of its
        # own arbitrary markers.
        planning_targets_by_region: dict[str, list[str]] = {}
        planning_roles_by_region: dict[str, str] = {}
        for region_id in sorted(decision_ids):
            claimed = sorted(
                set(graph.regions[region_id].violation_ids) & claimed_frontier
            )
            planning_targets_by_region[region_id] = claimed[:1]
            planning_roles_by_region[region_id] = "TARGET_REPAIR"
        for region_id in sorted(helper_ids):
            assignment = helper_assignment_by_region[region_id]
            assisted = sorted(
                set(assignment.assisted_target_violation_ids)
                & claimed_frontier
            )
            planning_targets_by_region[region_id] = assisted[:1]
            planning_roles_by_region[region_id] = "DEPENDENCY_ASSIST"
        audit_event_start = (
            len(self.integration_audit.events)
            if self.integration_audit is not None else 0
        )
        signature = ContextSignatureBuilder().build(
            subgraph=subgraph, graph=graph,
            violations=violations, objects=objects,
        )
        allowed_by_region = {
            region_id: self._allowed_actions(graph.regions[region_id], catalog)
            for region_id in subgraph.region_ids
        }
        kernel_results_by_region: dict[str, KernelRegionResult] = {}
        from drc_agent.experience.lessons import execution_scope
        from drc_agent.schemas.common import file_sha256
        sandbox_owner=getattr(self.candidate_sandbox,"__self__",None)
        current_backend=getattr(sandbox_owner,"backend",None)
        image_identity=getattr(current_backend,"image_digest",None)
        current_coordination = []
        scope_region_ids = set(subgraph.region_ids)
        for edge in graph.edges:
            if not edge.hard or {edge.u, edge.v} - scope_region_ids:
                continue
            physical = [
                item for item in edge.evidence
                if item.evidence_id.startswith("physical:")
                and item.details.get("evidence_quality") in {
                    "exact", "legacy_exact", "exact_or_proven_physical",
                }
            ]
            if not physical:
                continue
            current_coordination.append(CoordinationRequirement(
                region_ids=sorted([edge.u, edge.v]),
                requirement="PRESERVE_CURRENT_HARD_PHYSICAL_DEPENDENCY:"
                    + ",".join(sorted({item.kind for item in physical})),
                evidence_ids=sorted(item.evidence_id for item in physical),
            ))
        query = ExperienceQuery(
            subgraph_id=subgraph.subgraph_id,
            planning_scope_id=(subgraph.physical_scope_id or subgraph.subgraph_id),
            planning_scope_type=(
                "HIERARCHICAL_VIEW"
                if subgraph.scope_kind == "PLANNING_VIEW"
                else "FLAT_SUBGRAPH"
            ),
            view_ids=([subgraph.planning_view_id] if subgraph.planning_view_id else []),
            current_snapshot_id=current_snapshot_id, pdk="ASAP7",
            knowledge_cutoff=(self.experience_store.knowledge_cutoff()
                if self.config.features.experience_graph else None),
            scope=execution_scope(config=self.config,
                deck_sha256=file_sha256(rule_deck_path) if rule_deck_path else None,
                backend_image_digest=image_identity() if callable(image_identity) else None,
                evaluator_sha256=evaluator_hash),
            backend="klayout_dac26", signature=signature,
            allowed_action_families=set().union(*allowed_by_region.values()),
            current_coordination_requirements=current_coordination,
        )
        if self.config.features.experience_graph:
            seeds = HybridRetriever().retrieve(
                query, self.experience_store, self.config.retrieval,
                enabled=True,
            )
            evidence = ExperienceExpander().expand(
                seeds, self.experience_store, self.config.retrieval, query=query,
            )
            current_evidence_ids = {
                evidence_id for item in current_coordination
                for evidence_id in item.evidence_ids
            }
            if current_evidence_ids:
                sections = dict(evidence.sections)
                sections["current_coordination_evidence"] = sorted(current_evidence_ids)
                evidence = evidence.model_copy(update={"sections": sections,
                    "evidence_ids": evidence.evidence_ids | current_evidence_ids})
            blueprint = await BlueprintBuilder().build(query, evidence)
            blueprint.validate_citations(
                evidence.evidence_ids, query.allowed_action_families,
            )
        else:
            # B5 identity invariant: do not call a store/retriever/expander.
            evidence = EvidencePack(
                query_hash=stable_hash(query), items=[],
                evidence_ids=set(), sections={},
            )
            blueprint = RepairBlueprint.empty(query.subgraph_id).model_copy(
                update={
                    "planning_scope_id": query.planning_scope_id
                    or query.subgraph_id,
                    "planning_scope_type": query.planning_scope_type,
                    "view_ids": query.view_ids,
                    # Current-snapshot physical obligations are method-neutral
                    # facts. B5 gets the same requirements as B6 while still
                    # retrieving no historical experience.
                    "coordination_requirements": current_coordination,
                }
            )
        round_one = (
            MessagePasser().round_one(subgraph, graph)
            if self.config.features.message_passing else []
        )
        region_ids = planning_region_ids
        worker_count = region_worker_count(
            len(region_ids), self.config.llm.max_concurrent_requests,
        )
        prepared_by_region: dict[
            str, PreparedKernelRegion | KernelRegionResult | Exception
        ] = {}
        symbolic_round_two: list[NeighborMessage] = []

        def kernel_kwargs(region_id: str) -> dict[str, Any]:
            region = graph.regions[region_id].model_copy(
                update={"iteration": iteration},
            )
            selected_target_ids = planning_targets_by_region[region_id]
            selected_witnesses = [
                design.rule_witnesses[identifier]
                for identifier in selected_target_ids
                if identifier in design.rule_witnesses
            ]
            witness_signature = self.repair_attempt_memory.witness_signature(
                selected_witnesses
            )
            selected_rules = sorted({
                (design.violations.get(identifier) or {}).get(
                    "rule_id", "UNKNOWN"
                )
                for identifier in selected_target_ids
            } | set(region.rule_ids))
            local_context_fingerprint = stable_hash({
                "snapshot_id": current_snapshot_id,
                "region_id": region_id,
                "bbox_dbu": region.bbox_dbu.model_dump(mode="json"),
                "target_violation_ids": sorted(selected_target_ids),
                "witness_signature": witness_signature,
            })
            inbound = [
                message for message in round_one
                if message.receiver_region_id == region_id
            ]
            return {
                "run_id": self.run_id,
                "iteration": iteration,
                "snapshot_id": current_snapshot_id,
                "current_snapshot": current_snapshot,
                "baseline_snapshot": baseline_snapshot,
                "rule_deck_path": rule_deck_path,
                "evaluator_hash": evaluator_hash,
                "case_id": case_id,
                "subgraph": subgraph,
                "region": region,
                "violations": violations,
                "objects": objects,
                "design": design,
                "rule_catalog": catalog,
                "script": script,
                "neighbor_messages": inbound,
                "blueprint": blueprint,
                "failure_memory": [
                    {
                        "semantic_fingerprint": item.semantic_fingerprint,
                        "outcome": item.outcome,
                        "memory_layer": "CURRENT_REGION_CANDIDATE_HISTORY",
                    }
                    for item in region.candidate_history
                ] + self.repair_attempt_memory.prompt_records(
                    case_id=case_id,
                    snapshot_id=current_snapshot_id,
                    region_id=region_id,
                    rule_ids=selected_rules,
                    witness_signature=witness_signature,
                    local_context_fingerprint=local_context_fingerprint,
                ),
                "execution_epoch": execution_epoch,
                "window_id": window_id,
                "window_index": window_index,
                "artifact_prefix": artifact_prefix,
                "planning_target_violation_ids": planning_targets_by_region[region_id],
                "planning_role": planning_roles_by_region[region_id],
                "assisted_region_ids": (
                    helper_assignment_by_region[region_id].assisted_region_ids
                    if region_id in helper_assignment_by_region else []
                ),
                "assisted_target_violation_ids": (
                    helper_assignment_by_region[
                        region_id
                    ].assisted_target_violation_ids
                    if region_id in helper_assignment_by_region else []
                ),
                "helper_dependency_evidence_ids": (
                    helper_assignment_by_region[
                        region_id
                    ].dependency_evidence_ids
                    if region_id in helper_assignment_by_region else []
                ),
                "helper_potential_dof_capability_ids": (
                    helper_assignment_by_region[
                        region_id
                    ].helper_potential_dof_capability_ids
                    if region_id in helper_assignment_by_region else []
                ),
                "helper_potential_carrier_ids": (
                    helper_assignment_by_region[
                        region_id
                    ].helper_potential_carrier_ids
                    if region_id in helper_assignment_by_region else []
                ),
                "helper_protected_relation_ids": (
                    helper_assignment_by_region[
                        region_id
                    ].helper_protected_relation_ids
                    if region_id in helper_assignment_by_region else []
                ),
            }

        formal_symbolic_split = self.formal_kernel is not None and all((
            current_snapshot is not None,
            baseline_snapshot is not None,
            rule_deck_path is not None,
        ))
        if formal_symbolic_split:
            semaphore = asyncio.Semaphore(worker_count)

            async def prepare_formal(region_id: str) -> None:
                async with semaphore:
                    try:
                        prepared_by_region[region_id] = (
                            await self.formal_kernel.plan_symbolic_region(
                                **kernel_kwargs(region_id)
                            )
                        )
                    except (InfrastructureFailure, IntegrityFailure):
                        raise
                    except Exception as exc:
                        prepared_by_region[region_id] = exc

            preparation_tasks = [
                asyncio.create_task(prepare_formal(region_id))
                for region_id in region_ids
            ]
            try:
                await asyncio.gather(*preparation_tasks)
            except (InfrastructureFailure, IntegrityFailure):
                for task in preparation_tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(
                    *preparation_tasks, return_exceptions=True,
                )
                raise
            plan_summaries: dict[str, list[IntentSummary]] = {}
            for region_id in region_ids:
                prepared = prepared_by_region.get(region_id)
                if not isinstance(prepared, PreparedKernelRegion):
                    plan_summaries[region_id] = []
                    continue
                plan = prepared.provisional_plan
                from drc_agent.repair_kernel_integration.llm_planner import symbolic_intent_view
                plan_summaries[region_id] = [IntentSummary(
                    physical_intent=symbolic_intent_view(plan, prepared.scenes, prepared.dofs),
                    action_family=plan.strategy,
                    plan_id=plan.plan_id,
                    target_violation_ids=plan.target_violation_ids,
                    target_relation=plan.target_relation,
                    preferred_participant_ids=plan.preferred_participant_ids,
                    preferred_dof_ids=plan.preferred_dof_ids,
                    coordination_request=plan.coordination_request,
                    fingerprint=stable_hash(plan.model_dump(mode="json")),
                )]
            if (
                self.config.features.message_passing
                and self.config.agent.message_rounds >= 2
                and subgraph.edge_ids
            ):
                symbolic_round_two = [
                    message for message in MessagePasser().round_two(
                        subgraph, graph, plan_summaries,
                    )
                    if message.proposed_intents
                ]

        async def region_plan(region_id: str) -> RegionAgentResult:
            region = graph.regions[region_id].model_copy(
                update={"iteration": iteration},
            )
            inbound = [
                message for message in round_one
                if message.receiver_region_id == region_id
            ]
            predicates = [
                design.rule_predicates.get(rule_id)
                for rule_id in region.rule_ids
            ]
            witnesses = [
                design.rule_witnesses.get(violation_id)
                for violation_id in region.violation_ids
            ]
            sound_simple = (
                self.config.repair_programming.sound_simple_fast_path
                and bool(predicates)
                and all(
                    item and item.get("classification") == "SOUND_SIMPLE"
                    for item in predicates
                )
                and bool(witnesses)
                and all(
                    item
                    and item.get("predicate_fidelity")
                    == "EXACT_SIGNOFF_PREDICATE"
                    and not item.get("unresolved_reasons")
                    for item in witnesses
                )
            )
            if self.formal_kernel is not None:
                if formal_symbolic_split:
                    prepared = prepared_by_region[region_id]
                    if isinstance(prepared, Exception):
                        raise prepared
                    if isinstance(prepared, KernelRegionResult):
                        kernel_result = prepared
                    else:
                        round_two_inbound = [
                            message for message in symbolic_round_two
                            if message.receiver_region_id == region_id
                        ]
                        kernel_result = (
                            await self.formal_kernel.execute_symbolic_plan(
                                prepared,
                                round_two_messages=round_two_inbound,
                            )
                        )
                else:
                    kernel_result = await self.formal_kernel.plan_region(
                        **kernel_kwargs(region_id)
                    )
                kernel_results_by_region[region_id] = kernel_result
                return self._kernel_result_to_region_result(kernel_result)
            if sound_simple:
                return self._deterministic_sound_plan(
                    region=region, subgraph_id=subgraph.subgraph_id,
                    violations=violations, objects=objects,
                    neighbor_messages=inbound, blueprint=blueprint,
                    design=design,
                    allowed_actions=allowed_by_region[region_id],
                    catalog=catalog, script=script,
                )
            failed = {
                attempt.semantic_fingerprint
                for attempt in region.candidate_history
                if (
                    attempt.outcome in FAILURE_OUTCOMES
                    or "FAIL" in attempt.outcome
                    or "ROLLBACK" in attempt.outcome
                )
            } | self.repair_attempt_memory.failed_fingerprints(region_id)
            return await self.region_agent.plan(
                run_id=self.run_id, subgraph_id=subgraph.subgraph_id,
                region=region, violations=violations, objects=objects,
                neighbor_messages=inbound, blueprint=blueprint,
                design=design, allowed_actions=allowed_by_region[region_id],
                rule_catalog=catalog,
                failed_fingerprints=failed, script=script,
            )

        queue: asyncio.Queue[str] = asyncio.Queue()
        for region_id in region_ids:
            queue.put_nowait(region_id)
        outcome_by_region: dict[str, RegionAgentResult | Exception] = {}

        async def worker() -> None:
            while True:
                try:
                    region_id = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    outcome_by_region[region_id] = await region_plan(region_id)
                except (InfrastructureFailure, IntegrityFailure):
                    raise
                except Exception as exc:
                    outcome_by_region[region_id] = exc
                finally:
                    queue.task_done()

        workers = [
            asyncio.create_task(worker()) for _ in range(worker_count)
        ]
        try:
            await asyncio.gather(*workers)
        except (InfrastructureFailure, IntegrityFailure):
            for task in workers:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            raise
        outcomes = [outcome_by_region[region_id] for region_id in region_ids]
        region_results: list[RegionAgentResult] = []
        for region_id, outcome in zip(region_ids, outcomes, strict=True):
            if not isinstance(outcome, Exception):
                region_results.append(outcome)
                continue
            region = graph.regions[region_id].model_copy(
                update={"iteration": iteration},
            )
            noop = make_noop_candidate(region, subgraph.subgraph_id)
            region_results.append(RegionAgentResult(
                region_id=region_id, subgraph_id=subgraph.subgraph_id,
                diagnosis=None,
                candidates=[noop.model_dump(mode="json")],
                failures=[IntentFailure(
                    intent_id="region_planning",
                    code="REGION_PLANNING_DEGRADED_NO_OP",
                    message=(
                        f"{type(outcome).__name__}: {str(outcome)[:460]}"
                    ),
                )],
            ))
        for region_id in sorted(context_ids):
            region = graph.regions[region_id].model_copy(
                update={"iteration": iteration},
            )
            region_results.append(RegionAgentResult(
                region_id=region_id, subgraph_id=subgraph.subgraph_id,
                diagnosis=None, planning_path="CONTEXT_ONLY_NOOP",
                candidates=[
                    make_noop_candidate(region, subgraph.subgraph_id).model_dump(
                        mode="json"
                    )
                ],
            ))
        for region_id in sorted(unsupported_helper_ids):
            region = graph.regions[region_id].model_copy(
                update={"iteration": iteration},
            )
            assignment = helper_assignment_by_region[region_id]
            region_results.append(RegionAgentResult(
                region_id=region_id, subgraph_id=subgraph.subgraph_id,
                diagnosis=None, planning_path="HELPER_UNSUPPORTED_NOOP",
                candidates=[
                    make_noop_candidate(region, subgraph.subgraph_id).model_dump(
                        mode="json"
                    )
                ],
                failures=[IntentFailure(
                    intent_id="dependency_assist",
                    code="HELPER_ACTION_UNSUPPORTED",
                    message=(
                        "No source-authorized helper DOF/carrier for dependency "
                        + ",".join(assignment.dependency_evidence_ids)
                    )[:500],
                )],
            ))
        region_results.sort(key=lambda item: item.region_id)

        candidates = [
            RepairCandidate.model_validate(candidate)
            for result in region_results for candidate in result.candidates
        ]
        intents_by_region = {}
        for result in region_results:
            summaries = []
            for candidate in candidates:
                if candidate.region_id != result.region_id or candidate.is_noop:
                    continue
                summaries.append(IntentSummary(
                    action_family=candidate.action_family,
                    target_object_ids=candidate.target_object_ids,
                    footprint_dbu=candidate.edit_footprint_dbu,
                    fingerprint=candidate.semantic_fingerprint,
                ))
            intents_by_region[result.region_id] = summaries
        round_two = (
            symbolic_round_two
            if formal_symbolic_split
            else (
                MessagePasser().round_two(
                    subgraph, graph, intents_by_region,
                ) if self.config.features.message_passing and
                self.config.agent.message_rounds >= 2 else []
            )
        )
        messages = round_one + round_two
        candidate_graph = CandidateGraphBuilder(interaction_margin_dbu=self.config.candidate_graph.interaction_margin_dbu, fail_on_incomplete_audit=self.config.candidate_graph.fail_on_incomplete_audit).build(
            candidates, subgraph.subgraph_id,
            resource_capacities=design.resource_capacities,
        )
        programming_traces: list[RepairProgrammingTrace] = []
        programming_errors: list[dict] = []
        sandbox_records: list[dict] = []
        isolated_no_goods: list[list[str]] = []
        if (
            self.config.candidate_evidence.sandbox_enabled
            and not defer_isolated_sandbox
        ):
            isolated = TwoStageSandboxPolicy.select_isolated(
                candidates,
                min(
                    self.config.candidate_evidence.sandbox_top_m_per_view,
                    self.config.candidate_evidence.sandbox_max_jobs_per_iteration,
                    (
                        sandbox_job_limit
                        if sandbox_job_limit is not None
                        else self.config.candidate_evidence.sandbox_max_jobs_per_iteration
                    ),
                ),
            )
            for candidate in isolated:
                sandbox_kwargs = {
                    "candidate": candidate,
                    "iteration": iteration,
                    "subgraph": subgraph,
                    "source_map": objects,
                    "script": script,
                    "execution_epoch": execution_epoch,
                    "window_id": window_id,
                    "window_index": window_index,
                }
                if artifact_prefix:
                    sandbox_kwargs["artifact_prefix"] = artifact_prefix
                    sandbox_kwargs["job_run_id"] = (
                        f"{self.run_id}-i{iteration:04d}-"
                        f"{artifact_prefix.rsplit('/', 1)[-1]}-sandbox-isolated"
                    )
                evidence_result = (
                    self.candidate_sandbox(**sandbox_kwargs)
                    if self.candidate_sandbox is not None
                    else SandboxEvidence(
                        status=SandboxStatus.EXECUTION_FAIL,
                        base_snapshot_id=current_snapshot_id,
                    )
                )
                candidate.benefit_evidence.sandbox = evidence_result
                sandbox_records.append({
                    "candidate_ids": [candidate.candidate_id],
                    "evidence": evidence_result.model_dump(mode="json"),
                })
                if (
                    not candidate.is_noop
                    and has_complete_physical_truth(evidence_result)
                ):
                    effect = build_physical_effect_fingerprint(candidate)
                    target_witnesses = [
                        design.rule_witnesses[identifier]
                        for identifier in candidate.target_violation_ids
                        if identifier in design.rule_witnesses
                    ]
                    target_rules = sorted({
                        (design.violations.get(identifier) or {}).get(
                            "rule_id", "UNKNOWN"
                        )
                        for identifier in candidate.target_violation_ids
                    })
                    self.repair_attempt_memory.record(
                        case_id=design.case_id,
                        snapshot_id=current_snapshot_id,
                        region_id=candidate.region_id,
                        rule_ids=target_rules,
                        violation_ids=candidate.target_violation_ids,
                        witness_signature=(
                            self.repair_attempt_memory.witness_signature(
                                target_witnesses
                            )
                        ),
                        physical_effect_fingerprint=effect.sha256,
                        outcome=evidence_result.status.value,
                        semantic_fingerprint=candidate.semantic_fingerprint,
                        checker_outcome=candidate.validation_status.value,
                        sandbox_outcome=evidence_result.status.value,
                        removed_ids=(
                            evidence_result.removed_original_violation_ids
                        ),
                        introduced_rules=[
                            item.get("rule_id", "UNKNOWN")
                            for item in evidence_result.new_violation_records
                        ],
                        connectivity=evidence_result.connectivity_preserved,
                        failure_reason=evidence_result.failure_message,
                        iteration=iteration,
                        failure_class=(
                            "PHYSICAL_VERDICT"
                            if evidence_result.status.value in {
                                "NO_PROGRESS", "OFF_TARGET_EFFECT", "REGRESSION",
                                "CONNECTIVITY_FAIL",
                            }
                            else None
                        ),
                        evidence_validity="VALID",
                        fresh_physical_evidence=True,
                        retry_trigger={
                            "NO_PROGRESS": "PARAMETER_OR_CARRIER_MUST_CHANGE",
                            "OFF_TARGET_EFFECT": "TARGET_BINDING_MUST_CHANGE",
                            "REGRESSION": "PROTECTED_FOOTPRINT_OR_CARRIER_MUST_CHANGE",
                            "CONNECTIVITY_FAIL": "TOPOLOGY_OR_CARRIER_MUST_CHANGE",
                        }.get(evidence_result.status.value),
                        local_context_fingerprint=stable_hash({
                            "snapshot_id": current_snapshot_id,
                            "region_id": candidate.region_id,
                            "witness_signature": (
                                self.repair_attempt_memory.witness_signature(
                                    target_witnesses
                                )
                            ),
                        }),
                        canonical_program_signature=(
                            candidate.semantic_fingerprint
                        ),
                        topology_fingerprint=stable_hash({
                            "connectivity_reference_sha256": (
                                evidence_result.connectivity_reference_sha256
                            ),
                            "target_source_object_ids": (
                                effect.target_source_object_ids
                            ),
                        }),
                        write_footprint=effect.target_source_object_ids,
                    )
                if (
                    evidence_result.status != SandboxStatus.CLEAN_PROGRESS
                    and has_complete_physical_truth(evidence_result)
                ):
                    isolated_no_goods.append([candidate.candidate_id])
            # Graph Action v3 escalation is deliberately bounded.  The LLM
            # proposes only RepairProgram JSON; compiler/checker/sandbox remain
            # the authority for object ownership, geometry, and success.
            program_cfg = self.config.repair_programming
            if (
                self.formal_kernel is None
                and program_cfg.enabled
                and self.candidate_sandbox is not None
                and program_cfg.max_regions_per_subgraph > 0
                and program_cfg.max_calls_per_subgraph > 0
            ):
                scheduled_region_ids = RepairabilityScheduler().rank(
                    planning_region_ids, graph=graph, design=design,
                    memory=self.repair_attempt_memory,
                )

                calls_used = 0
                regions_used = 0
                for region_id in scheduled_region_ids:
                    if regions_used >= program_cfg.max_regions_per_subgraph:
                        break
                    remaining_calls = program_cfg.max_calls_per_subgraph - calls_used
                    if remaining_calls <= 0:
                        break
                    region = graph.regions[region_id].model_copy(
                        update={"iteration": iteration}
                    )
                    witness_ids = [
                        violation_id for violation_id in region.violation_ids
                        if violation_id in design.rule_witnesses
                    ]
                    if program_cfg.require_rule_witness and not witness_ids:
                        continue
                    classifications = {
                        (design.rule_predicates.get(rule_id) or {}).get(
                            "classification", "UNSUPPORTED"
                        )
                        for rule_id in region.rule_ids
                    }
                    local_candidates = [
                        item for item in candidates
                        if item.region_id == region_id and not item.is_noop
                    ]
                    evaluated = [
                        item for item in local_candidates
                        if item.benefit_evidence.sandbox is not None
                    ]
                    contextual = (
                        program_cfg.escalate_contextual
                        and "CONTEXTUAL" in classifications
                    )
                    unsupported = (
                        program_cfg.escalate_unsupported
                        and not local_candidates
                    )
                    all_no_progress = (
                        program_cfg.escalate_after_no_progress
                        and bool(local_candidates)
                        and len(evaluated) == len(local_candidates)
                        and all(
                            item.benefit_evidence.sandbox.status
                            != SandboxStatus.CLEAN_PROGRESS
                            for item in evaluated
                        )
                    )
                    if not (contextual or unsupported or all_no_progress):
                        continue
                    inbound = [
                        message for message in round_one
                        if message.receiver_region_id == region_id
                    ]
                    context = self.region_agent.context_serializer.build(
                        region=region, violations=violations, objects=objects,
                        legal_layers=design.legal_layers,
                        neighbor_messages=inbound, blueprint=blueprint,
                        allowed_actions=allowed_by_region[region_id],
                        script=script, rule_catalog=catalog,
                        local_topology=design.local_topology,
                        rule_predicates=design.rule_predicates,
                        rule_witnesses=design.rule_witnesses,
                        rule_knowledge_packs=design.rule_knowledge_packs,
                    )
                    if not context.object_to_violation_ids:
                        programming_errors.append({
                            "region_id": region_id,
                            "stage": "repair_programming_preflight",
                            "error_type": "ProgrammerCapabilityUnavailable",
                            "message": (
                                "no witness-grounded editable source object"
                            ),
                        })
                        regions_used += 1
                        continue
                    revisions = min(
                        program_cfg.max_revisions, remaining_calls - 1
                    )
                    controller = RepairProgrammingController(
                        programmer=LLMRepairProgrammer(self.llm, tool_loop_enabled=True),
                        compiler=RepairProgramCompiler(),
                        checker=CandidateChecker(),
                        preview_tools=SandboxPreviewTools(
                            candidate_sandbox=self.candidate_sandbox,
                            design=design, region=region,
                            violations=violations,
                        ),
                        max_revisions=revisions,
                        attempt_memory=self.repair_attempt_memory,
                    )
                    try:
                        candidate, trace = await controller.run(
                            context=context, run_id=self.run_id,
                            subgraph_id=subgraph.subgraph_id,
                            region=region, design=design,
                            allowed_action_families=allowed_by_region[region_id],
                            violations=violations, rule_catalog=catalog,
                            sandbox_kwargs={
                                "iteration": iteration,
                                "subgraph": subgraph,
                                "source_map": objects,
                                "script": script,
                            },
                            snapshot_id=current_snapshot_id,
                        )
                    except (InfrastructureFailure, IntegrityFailure):
                        raise
                    except Exception as exc:
                        programming_errors.append({
                            "region_id": region_id,
                            "stage": "repair_programming",
                            "error_type": type(exc).__name__,
                            "message": str(exc)[:500],
                        })
                        regions_used += 1
                        calls_used += revisions + 1
                        continue
                    programming_traces.append(trace)
                    regions_used += 1
                    calls_used += trace.llm_calls
                    if candidate is None:
                        continue
                    candidates.append(candidate)
                    result = next(
                        item for item in region_results
                        if item.region_id == region_id
                    )
                    result.candidates.append(candidate.model_dump(mode="json"))
                    evidence_result = candidate.benefit_evidence.sandbox
                    sandbox_records.append({
                        "candidate_ids": [candidate.candidate_id],
                        "path": "LLM_REPAIR_PROGRAM_PREVIEW",
                        "evidence": evidence_result.model_dump(mode="json"),
                    })
            candidate_graph = CandidateGraphBuilder(
                interaction_margin_dbu=(
                    self.config.candidate_graph.interaction_margin_dbu
                ),
                fail_on_incomplete_audit=(
                    self.config.candidate_graph.fail_on_incomplete_audit
                ),
            ).build(
                candidates, subgraph.subgraph_id,
                resource_capacities=design.resource_capacities,
            )
            candidate_graph.no_good_candidate_sets.extend(isolated_no_goods)
            candidate_graph.no_good_candidate_sets = sorted(
                {tuple(item) for item in candidate_graph.no_good_candidate_sets}
            )
        if self.integration_audit is not None:
            kernel_graph_candidates = [
                item for item in candidate_graph.candidates
                if any(
                    edit.provenance.generator == "formal-repair-kernel-v1"
                    for edit in item.edits
                )
            ]
            self.integration_audit.candidate_graph_kernel_candidate_count += len(
                kernel_graph_candidates
            )
            for item in kernel_graph_candidates:
                self._integration_event(
                    "repair_kernel_candidate_entered_gc",
                    subgraph_id=subgraph.subgraph_id,
                    candidate_id=item.candidate_id,
                )
        coordinator = (
            GraphCoordinator() if self.coordinator_mode == "cp_sat"
            else GreedyCoordinator()
        )
        bundle = coordinator.solve(candidate_graph, self.config.coordinator)
        if self.integration_audit is not None:
            selected = set(bundle.selected_candidate_ids)
            kernel_selected = [
                item for item in candidates
                if item.candidate_id in selected
                and any(
                    edit.provenance.generator == "formal-repair-kernel-v1"
                    for edit in item.edits
                )
            ]
            self.integration_audit.cp_sat_selected_kernel_candidate_count += len(
                kernel_selected
            )
            for item in kernel_selected:
                self._integration_event(
                    "repair_kernel_candidate_selected_by_cp_sat",
                    subgraph_id=subgraph.subgraph_id,
                    candidate_id=item.candidate_id,
                )
        if (
            not defer_final_joint_sandbox
            and self.config.candidate_evidence.sandbox_enabled
            and self.config.candidate_evidence.joint_sandbox_enabled
            and TwoStageSandboxPolicy.needs_joint_sandbox(
                bundle.selected_candidate_ids,
                candidate_graph,
                self.config.candidate_evidence.joint_sandbox_max_bundle_size,
            )
        ):
            selected_non_noop = [
                candidate for candidate in candidates
                if candidate.candidate_id in set(bundle.selected_candidate_ids)
                and not candidate.is_noop
            ]
            joint_evidence = (
                self.bundle_sandbox(
                    candidates=selected_non_noop,
                    iteration=iteration,
                    subgraph=subgraph,
                    source_map=objects,
                    script=script,
                )
                if self.bundle_sandbox is not None
                else SandboxEvidence(
                    status=SandboxStatus.EXECUTION_FAIL,
                    base_snapshot_id=current_snapshot_id,
                )
            )
            sandbox_records.append({
                "candidate_ids": sorted(
                    item.candidate_id for item in selected_non_noop
                ),
                "evidence": joint_evidence.model_dump(mode="json"),
            })
            TwoStageSandboxPolicy.apply_joint_result(
                candidate_graph,
                [item.candidate_id for item in selected_non_noop],
                joint_evidence,
            )
            if joint_evidence.status == SandboxStatus.CLEAN_PROGRESS:
                bundle = bundle.model_copy(update={
                    "sandbox_status": joint_evidence.status,
                    "sandbox_verification_ref": joint_evidence.verification_ref,
                })
            else:
                bundle = coordinator.solve(
                    candidate_graph, self.config.coordinator,
                )
        local_kernel_events = (
            self.integration_audit.events[audit_event_start:]
            if self.integration_audit is not None else []
        )
        llm_request_events = [
            item for item in local_kernel_events
            if item.get("event") in {
                "repair_kernel_plan_requested",
                "repair_kernel_plan_revision_requested",
                "repair_kernel_final_plan_requested",
                "repair_kernel_execution_revision_requested",
            }
            and item.get("region_id") in set(planning_region_ids)
        ]
        admitted_frontier = {
            target for targets in planning_targets_by_region.values()
            for target in targets
        } & claimed_frontier
        llm_started_frontier = {
            target
            for item in local_kernel_events
            if item.get("event") in {
                "repair_kernel_plan_received",
                "repair_kernel_plan_revision_received",
                "repair_kernel_final_plan_received",
                "repair_kernel_execution_revision_received",
            }
            and item.get("region_id") in set(planning_region_ids)
            for target in planning_targets_by_region.get(
                str(item.get("region_id")), []
            )
        } & claimed_frontier
        plan_validated_frontier = {
            str(target) for item in local_kernel_events
            if item.get("event") == "repair_kernel_plan_binding_passed"
            for target in item.get("selected_violation_ids", [])
        } & claimed_frontier
        kernel_attempted_frontier = {
            str(item.get("violation_id")) for item in local_kernel_events
            if item.get("event") == "repair_kernel_root_proposal_attempted"
            and item.get("violation_id")
        } & claimed_frontier
        physical_evaluated_frontier = {
            str(item.get("violation_id")) for item in local_kernel_events
            if (
                item.get("event") == "repair_kernel_trajectory_evaluated"
                and item.get("violation_id")
                and bool(item.get("fresh_physical_evaluated"))
            )
        } & claimed_frontier
        unsupported_statuses = {
            "INSUFFICIENT_FIDELITY", "RULE_NOT_LIVE_QUALIFIED",
            "UNSUPPORTED_REPAIR_FAMILY",
        }
        unsupported_frontier = {
            str(violation_id)
            for kernel_result in kernel_results_by_region.values()
            for report in kernel_result.support_reports
            if report.status.value in unsupported_statuses
            for violation_id in report.violation_ids
        } & claimed_frontier
        # Context construction and a pre-dispatch request event do not make a
        # target attempted. A returned LLM plan, binding/root progress, or a
        # fresh physical evaluation does.
        attempted_frontier: set[str] = (
            llm_started_frontier
            | plan_validated_frontier
            | kernel_attempted_frontier
            | physical_evaluated_frontier
        ) - unsupported_frontier
        for result in region_results:
            if result.region_id not in planning_region_ids:
                continue
            if (
                self.formal_kernel is None
                and (
                    result.llm_invocation_count > 0
                    or result.planning_path == "DETERMINISTIC_SOUND"
                )
                and result.planning_path not in {
                    "DEGRADED_NOOP", "UNSUPPORTED_NOOP",
                    "HELPER_UNSUPPORTED_NOOP", "CONTEXT_ONLY_NOOP",
                }
            ):
                selected_targets = {
                    target for intent in result.intents
                    for target in intent.target_violation_ids
                }
                attempted_frontier.update(
                    (selected_targets or set(planning_targets_by_region.get(
                        result.region_id, []
                    ))) & claimed_frontier
                )
        deferred_frontier = (
            claimed_frontier - attempted_frontier - unsupported_frontier
        )
        role_audit = {
            "total_regions": len(full_region_ids),
            "decision_region_ids": sorted(decision_ids),
            "helper_region_ids": sorted(helper_ids),
            "context_only_region_ids": sorted(context_ids),
            "claimed_frontier_violation_ids": sorted(claimed_frontier),
            "planning_target_violation_ids_by_region": planning_targets_by_region,
            "planning_roles_by_region": planning_roles_by_region,
            "helper_assignments": [
                item.model_dump(mode="json") for item in helper_assignments
            ],
            "planning_target_policy": (
                "FIRST_STABLE_TARGET_OR_EXPLICIT_DEPENDENCY_ASSIST"
            ),
            "admitted_frontier_violation_ids": sorted(admitted_frontier),
            "llm_started_frontier_violation_ids": sorted(llm_started_frontier),
            "plan_validated_frontier_violation_ids": sorted(
                plan_validated_frontier
            ),
            "kernel_attempted_frontier_violation_ids": sorted(
                kernel_attempted_frontier
            ),
            "physical_evaluated_frontier_violation_ids": sorted(
                physical_evaluated_frontier
            ),
            "attempted_frontier_violation_ids": sorted(attempted_frontier),
            "deferred_frontier_violation_ids": sorted(deferred_frontier),
            "unsupported_frontier_violation_ids": sorted(
                unsupported_frontier
            ),
            "llm_invocation_count": (
                len(llm_request_events)
                + sum(item.llm_invocation_count for item in region_results)
            ),
        }
        from drc_agent.experience.adoption import build_adoption_trace
        adoption_trace = build_adoption_trace(
            subgraph_id=subgraph.subgraph_id,
            current_snapshot_id=current_snapshot_id,
            evidence=evidence,
            blueprint=blueprint,
            kernel_results_by_region=kernel_results_by_region,
            audit_events=local_kernel_events,
        )
        refs = {
            "subgraph": self.artifact_store.write_json(
                f"{prefix}/subgraph.json", {
                    "subgraph": subgraph.model_dump(mode="json"),
                    "signature": signature.model_dump(mode="json"),
                    "region_ids": subgraph.region_ids,
                    "edge_ids": subgraph.edge_ids,
                }, producer="prepare_subgraph_context",
                schema_name="SubgraphSnapshot",
            ),
            "evidence": self.artifact_store.write_json(
                f"{prefix}/evidence_pack.json", evidence,
                producer="retrieve_experience", schema_name="EvidencePack",
            ),
            "blueprint": self.artifact_store.write_json(
                f"{prefix}/blueprint.json", blueprint,
                producer="build_blueprint", schema_name="RepairBlueprint",
            ),
            "messages": self.artifact_store.write_json(
                f"{prefix}/neighbor_messages.json",
                [item.model_dump(mode="json") for item in messages],
                producer="message_pass", schema_name="NeighborMessage[]",
            ),
            "repair_hypotheses": self.artifact_store.write_json(
                f"{prefix}/repair_hypotheses.json",
                {
                    "artifact_role": (
                        "non_exhaustive_deterministic_affordances_and_fallbacks"
                    ),
                    "llm_candidate_space_is_not_limited_to_these_hints": True,
                    "regions": [
                        {
                            "region_id": result.region_id,
                            "affordance_hints": [
                                item.model_dump(mode="json")
                                for item in result.repair_hypotheses
                            ],
                            "guidance_audit": [
                                item.model_dump(mode="json")
                                for item in result.guidance_audit
                            ],
                        }
                        for result in region_results
                    ],
                },
                producer="rule_aware_affordance_provider",
                schema_name="RuleAffordanceArtifact",
            ),
            "intents": self.artifact_store.write_json(
                f"{prefix}/candidate_intents.json",
                [
                    intent.model_dump(mode="json")
                    for result in region_results for intent in result.intents
                ],
                producer="generate_candidates", schema_name="CandidateIntent[]",
            ),
            "candidates": self.artifact_store.write_json(
                f"{prefix}/candidates.json",
                [item.model_dump(mode="json") for item in candidates],
                producer="validate_candidates", schema_name="RepairCandidate[]",
            ),
            "sandbox_evidence": self.artifact_store.write_json(
                f"{prefix}/sandbox_evidence.json", sandbox_records,
                producer="candidate_sandbox", schema_name="SandboxEvidence[]",
            ),
            "repair_programming": self.artifact_store.write_json(
                f"{prefix}/repair_programming.json", {
                    "traces": [
                        item.model_dump(mode="json")
                        for item in programming_traces
                    ],
                    "errors": programming_errors,
                    "budget": {
                        "max_regions_per_subgraph": (
                            self.config.repair_programming.max_regions_per_subgraph
                        ),
                        "max_calls_per_subgraph": (
                            self.config.repair_programming.max_calls_per_subgraph
                        ),
                    },
                }, producer="llm_repair_programmer",
                schema_name="RepairProgrammingArtifact",
            ),
            "candidate_graph": self.artifact_store.write_json(
                f"{prefix}/candidate_graph.json", candidate_graph,
                producer="build_candidate_graph", schema_name="CandidateGraph",
            ),
            "bundle": self.artifact_store.write_json(
                f"{prefix}/bundle.json", bundle,
                producer="coordinate_actions", schema_name="JointRepairBundle",
            ),
            "experience_adoption_trace": self.artifact_store.write_json(
                f"{prefix}/experience_adoption_trace.json", adoption_trace,
                producer="experience_adoption_audit",
                schema_name="ExperienceAdoptionTrace",
            ),
            "agent_results": self.artifact_store.write_json(
                f"{prefix}/region_agent_results.json",
                [item.model_dump(mode="json") for item in region_results],
                producer="region_agent", schema_name="RegionAgentResult[]",
            ),
            "region_roles": self.artifact_store.write_json(
                f"{prefix}/region_roles.json", role_audit,
                producer="subgraph_planner",
                schema_name="RegionRoleAndFrontierAudit",
            ),
        }
        if self.formal_kernel is not None:
            refs["repair_kernel"] = self.artifact_store.write_json(
                f"{prefix}/repair_kernel.json",
                {
                    "planning_path": "REPAIR_KERNEL",
                    "results": [
                        item.model_dump(mode="json")
                        for _, item in sorted(kernel_results_by_region.items())
                    ],
                    "audit": self.integration_audit.model_dump(mode="json"),
                },
                producer="formal_repair_kernel",
                schema_name="FormalRepairKernelArtifact",
            )
        return PlannedSubgraph(
            subgraph=subgraph, signature=signature, evidence=evidence,
            blueprint=blueprint, messages=messages,
            region_results=region_results, candidates=candidates,
            candidate_graph=candidate_graph, bundle=bundle,
            repair_programming_traces=programming_traces,
            decision_region_ids=sorted(decision_ids),
            helper_region_ids=sorted(helper_ids),
            context_only_region_ids=sorted(context_ids),
            claimed_frontier_violation_ids=sorted(claimed_frontier),
            admitted_frontier_violation_ids=sorted(admitted_frontier),
            llm_started_frontier_violation_ids=sorted(llm_started_frontier),
            plan_validated_frontier_violation_ids=sorted(
                plan_validated_frontier
            ),
            kernel_attempted_frontier_violation_ids=sorted(
                kernel_attempted_frontier
            ),
            physical_evaluated_frontier_violation_ids=sorted(
                physical_evaluated_frontier
            ),
            attempted_frontier_violation_ids=sorted(attempted_frontier),
            deferred_frontier_violation_ids=sorted(deferred_frontier),
            unsupported_frontier_violation_ids=sorted(unsupported_frontier),
            helper_assignments=helper_assignments,
            llm_invocation_count=(
                len(llm_request_events)
                + sum(item.llm_invocation_count for item in region_results)
            ),
            artifact_refs=refs,
            integration_audit=(
                self.integration_audit.model_dump(mode="json")
                if self.integration_audit is not None else {}
            ),
        )

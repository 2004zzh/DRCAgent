from __future__ import annotations

import asyncio
from functools import partial
from pathlib import Path
from typing import Any, Callable

from drc_agent.backends.execution import (
    EXECUTION_PROTOCOL_VERSION,
    ExecutionControl,
    atomic_write_json,
)
from drc_agent.config.loader import AppConfig
from drc_agent.llm import LLMStructuredResponseError
from drc_agent.reliability import FailureCode, InfrastructureFailure
from drc_agent.repair_kernel.models import RepairDOF, RepairScene
from drc_agent.schemas.action import (
    RepairCandidate, SandboxEvidence, SandboxScope, SandboxStatus,
)
from drc_agent.schemas.common import ArtifactRef, stable_hash
from drc_agent.schemas.tools import (
    EvidenceValidity,
    ExecutionPurpose,
    ExecutionTraceContext,
)

from .candidate_adapter import deduplicate_candidates, ensure_noop
from .current_semantic_adapter import (
    FormalCurrentSemanticAdapter,
    FormalCurrentSemanticResult,
)
from .equivalence import CondensationEquivalenceGate
from .llm_planner import KernelRepairPlan, SymbolicKernelPlanner
from .models import (
    FormalKernelExecutionContext,
    KernelRegionResult,
    KernelSupportReport,
    SupportStatus,
)
from .phase3_context_adapter import FormalPhase3ContextAdapter
from .plan_binding import (
    BoundKernelPlan,
    ExecutionFeedbackRevisionError,
    PlanOrigin,
    PlanBindingError,
    allowed_binding_ids,
    bind_kernel_plan,
    validate_execution_feedback_revision,
)
from .root_proposal_adapter import FormalRootProposalAdapter
from .semantic_registry import (
    build_plan_semantic_registry,
    install_plan_semantic_registry,
)
from .trajectory_engine import (
    FormalTrajectoryEngine,
    SharedLiveEvaluationBudget,
)


_SUPPORTED = {
    SupportStatus.SUPPORTED_DIRECT,
    SupportStatus.SUPPORTED_TRAJECTORY,
    SupportStatus.SUPPORTED_REVIEWED_RULE,
    SupportStatus.SUPPORTED_EXACT,
    SupportStatus.SUPPORTED_REVIEWED,
}


def _reserved_verification_stop(exc: InfrastructureFailure) -> bool:
    """Only search escrow exhaustion is a soft stop; real infra still aborts."""
    return (
        exc.failure_code == FailureCode.BUDGET_EXHAUSTED
        and exc.provider == "P4_VALIDATION_LEDGER"
        and exc.failure_stage == "VALIDATION_BUDGET"
        and str(exc) == "DEFERRED_BUDGET_RESERVED_FOR_REQUIRED_VERIFICATION"
    )


class FormalProductionTrajectoryEngine:
    """Compose the frozen repair capabilities for one formal Region.

    The engine consumes only the formal current snapshot and current DesignState.
    Development manifests and historical successful candidates are never read.
    Temporary debt remains inside the isolated trajectory sandbox; only a
    fresh-equivalent root-to-final candidate is returned to Graph Action.
    """

    def __init__(
        self,
        *,
        project_root: Path,
        config: AppConfig,
        artifact_root: Path,
        llm=None,
        max_candidates: int = 3,
        event_sink: Callable[..., Any] | None = None,
        eda_semaphore: asyncio.Semaphore | None = None,
    ):
        self.project_root = project_root.resolve()
        self.config = config
        self.artifact_root = artifact_root.resolve()
        self.llm = llm
        self.max_candidates = max(1, min(4, int(max_candidates)))
        self.event_sink = event_sink
        self.eda_semaphore = eda_semaphore or asyncio.Semaphore(
            config.backend.max_concurrent_eda_jobs
        )
        self.live_evaluation_budget = SharedLiveEvaluationBudget(
            config.repair_kernel_integration.max_kernel_live_calls_per_region
        )

    def _event(self, name: str, **details: Any) -> None:
        if self.event_sink is not None:
            self.event_sink(name, **details)

    async def _run_eda(
        self,
        *,
        kind: str,
        region_id: str,
        violation_id: str,
        fn: Callable[[], Any],
        proposal_id: str | None = None,
        trajectory_id: str | None = None,
        execution_control: ExecutionControl | None = None,
        claim_region_budget: bool = False,
    ) -> Any:
        control = execution_control or ExecutionControl()
        if (
            claim_region_budget
            and not self.live_evaluation_budget.claim()
        ):
            raise RuntimeError("REGION_LIVE_BUDGET_EXCEEDED")
        details = {
            "kind": kind,
            "region_id": region_id,
            "violation_id": violation_id,
        }
        if proposal_id is not None:
            details["proposal_id"] = proposal_id
        if trajectory_id is not None:
            details["trajectory_id"] = trajectory_id
        async with self.eda_semaphore:
            self._event("repair_kernel_eda_started", **details)
            worker = asyncio.create_task(asyncio.to_thread(fn))
            outcome = "UNKNOWN"
            try:
                value = await asyncio.shield(worker)
                outcome = "RETURNED"
                return value
            except asyncio.CancelledError:
                outcome = "CANCEL_REQUESTED"
                control.cancel("ASYNC_CALLER_CANCELLED")
                done, _ = await asyncio.wait(
                    {worker}, timeout=control.cancellation_grace_seconds,
                )
                grace_expired = not done
                if grace_expired:
                    self._event(
                        "repair_kernel_eda_cancel_grace_expired", **details,
                    )
                # Do not release the shared EDA slot while the worker can still
                # mutate its owned workspace.  The command runner observes the
                # same control and terminates only this attempt's process/container.
                try:
                    await asyncio.shield(worker)
                except Exception:
                    pass
                if grace_expired:
                    raise InfrastructureFailure(
                        "EDA worker exceeded cancellation grace period",
                        failure_code=FailureCode.EDA_CANCEL_TIMEOUT,
                        retryable=False,
                        provider="KLayoutBackend",
                        model=self.config.backend.image,
                        original_exception_type="CancellationGraceExpired",
                        failure_stage=kind,
                    )
                raise
            except Exception:
                outcome = "RAISED"
                raise
            finally:
                self._event(
                    "repair_kernel_eda_terminal", **details, outcome=outcome,
                    worker_done=worker.done(),
                    cancellation_requested=control.cancelled,
                )
                # Compatibility event: completion means wrapper termination,
                # not proof that DRC was evaluated.
                self._event(
                    "repair_kernel_eda_completed", **details,
                    wrapper_outcome=outcome,
                )

    def _proposal_evidence_dir(
        self,
        *,
        execution: FormalKernelExecutionContext,
        bound_plan: BoundKernelPlan,
        violation,
        proposal,
    ) -> Path:
        relative = Path(
            execution.artifact_prefix
            or f"iterations/iter_{execution.iteration:04d}"
        )
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("INVALID_REPAIR_KERNEL_ARTIFACT_PREFIX")
        return (
            self.artifact_root
            / relative
            / "planning/subgraphs"
            / stable_hash(execution.subgraph.subgraph_id)[:20]
            / "repair_kernel/regions"
            / stable_hash(execution.region.region_id)[:20]
            / "plans"
            / stable_hash(bound_plan.plan_id)[:20]
            / "targets"
            / stable_hash([violation.rule_id, violation.violation_id])[:20]
            / "proposals"
            / stable_hash(proposal.proposal_id)[:20]
        )

    @staticmethod
    def _trace_context(
        *,
        execution: FormalKernelExecutionContext,
        bound_plan: BoundKernelPlan,
        violation,
        proposal_id: str,
        parent_snapshot_id: str,
        purpose: ExecutionPurpose,
        depth: int,
        plan_revision: int = 0,
        invocation_suffix: str = "",
    ) -> ExecutionTraceContext:
        material = {
            "formal_context": execution.context_fingerprint,
            "plan": bound_plan.plan_id,
            "revision": plan_revision,
            "target": violation.violation_id,
            "proposal": proposal_id,
            "parent": parent_snapshot_id,
            "purpose": purpose.value,
            "depth": depth,
            "suffix": invocation_suffix,
        }
        return ExecutionTraceContext(
            formal_run_id=execution.run_id,
            execution_epoch=execution.execution_epoch,
            formal_iteration=execution.iteration,
            window_id=execution.window_id,
            window_index=execution.window_index,
            subgraph_id=execution.subgraph.subgraph_id,
            region_id=execution.region.region_id,
            plan_id=bound_plan.plan_id,
            plan_revision=plan_revision,
            target_violation_id=violation.violation_id,
            root_proposal_id=proposal_id,
            proposal_id=proposal_id,
            parent_snapshot_id=parent_snapshot_id,
            depth=depth,
            purpose=purpose,
            invocation_id="invocation_" + stable_hash(material)[:20],
        )

    @staticmethod
    def _persist_root_registration(
        *,
        path: Path,
        execution: FormalKernelExecutionContext,
        bound_plan: BoundKernelPlan,
        proposal,
        violation,
        root_diagnostics: list[dict[str, Any]],
        plan_revision: int,
        plan_semantic_registry: dict[str, Any],
    ) -> None:
        atomic_write_json(path / "lesson_binding.json", {
            "proposal_id":proposal.proposal_id,
            "target_violation_id":proposal.target_violation_id,
            "rule_id":proposal.rule_id,
            "action_family":proposal.candidate.action_family,
            "binding":proposal.executable_binding,
            "physical_effect_fingerprint":proposal.physical_effect_fingerprint,
            "parent_script_sha256":proposal.patch_plan.base_script_sha256,
            "plan_id":bound_plan.plan_id,
            "status":"UNVERIFIED_BEFORE_EDA",
        })
        atomic_write_json(path / "root_registration.json", {
            "execution_protocol_version": EXECUTION_PROTOCOL_VERSION,
            "status": "REGISTERED_BEFORE_EDA",
            "execution_context": execution.model_dump(mode="json"),
            "bound_plan": bound_plan.model_dump(mode="json"),
            "plan_semantic_registry": plan_semantic_registry,
            "plan_revision": plan_revision,
            "target": violation.model_dump(mode="json"),
            "root_diagnostics": root_diagnostics,
            "proposal": proposal.model_dump(mode="json"),
            "candidate": proposal.candidate.model_dump(mode="json"),
            "patch_plan": proposal.patch_plan.model_dump(mode="json"),
            "physical_effect_fingerprint": (
                proposal.physical_effect_fingerprint
            ),
            "parent_snapshot": execution.current_snapshot.model_dump(
                mode="json"
            ),
        })

    def prepare_region_inputs(
        self,
        *,
        formal_context,
        execution: FormalKernelExecutionContext,
        reports: list[KernelSupportReport],
    ) -> tuple[
        FormalCurrentSemanticResult,
        list[RepairScene],
        list[RepairDOF],
        list[dict[str, Any]],
        set[str],
    ]:
        current = FormalCurrentSemanticAdapter(
            self.project_root, self.config,
        ).build(execution)
        supported_violation_ids = {
            violation_id
            for item in reports if item.status in _SUPPORTED
            for violation_id in item.violation_ids
        }
        scenes: list[RepairScene] = []
        dofs: list[RepairDOF] = []
        scene_failures: list[dict[str, Any]] = []
        for violation in formal_context.region_violations:
            if violation.violation_id not in supported_violation_ids:
                continue
            try:
                public, scene, values = FormalPhase3ContextAdapter.build_scene(
                    execution, current.context, violation,
                )
            except (KeyError, ValueError, RuntimeError) as exc:
                scene_failures.append({
                    "code": "FORMAL_PHASE3_CONTEXT_FAIL",
                    "violation_id": violation.violation_id,
                    "message": str(exc),
                })
                continue
            scenes.append(scene)
            dofs.extend(values)
            from drc_agent.repair_kernel_closure.edit_authority import resolve_edit_authority
            from .executable_model_builder import ExecutableRepairModelBuilder
            model=ExecutableRepairModelBuilder().build(scene=scene,raw_dofs=values,
                execution=execution,authority=resolve_edit_authority(scene,public),context=public)
            from .context import compatibility_metadata, dof_capability_summary
            capabilities = dof_capability_summary(formal_context)
            constraint_dictionary = compatibility_metadata(
                formal_context, "executable_constraint_dictionary",
            )
            for constraint in (
                model.primary_constraints
                + model.protected_constraints
                + model.coupling_constraints
                + model.locality_constraints
                + model.compiler_constraints
            ):
                value = constraint.model_dump(mode="json")
                existing = constraint_dictionary.get(constraint.constraint_id)
                if existing is not None and existing != value:
                    raise ValueError("EXECUTABLE_CONSTRAINT_ID_COLLISION")
                constraint_dictionary[constraint.constraint_id] = value
            rejection_reasons = compatibility_metadata(
                formal_context, "executable_rejection_reason_dictionary",
            )
            for failure in model.rejected_dofs:
                reasons = rejection_reasons.setdefault(
                    failure.code.value, [],
                )
                if failure.reason not in reasons:
                    reasons.append(failure.reason)
                    reasons.sort()
            for dof in values:
                variants=[v for v in model.variables if dof.dof_id in v.raw_dof_ids]
                dof_failures = [
                    failure for failure in model.rejected_dofs
                    if dof.dof_id in failure.raw_dof_ids
                ]
                capabilities[dof.dof_id]={
                    "status":"SOURCE_PREFLIGHT_PASS" if variants else "EXECUTABLE_DOF_UNAVAILABLE",
                    "predicate_solved":False,"physical_verdict":"NOT_EVALUATED",
                    "variants":[{
                        "executable_variable_id":v.variable_id,
                        "carrier_kind":v.carrier_kind,
                        "atomic_co_dof_ids":v.raw_dof_ids,
                        "atomic_source_target_ids":v.atomic_source_target_ids,
                        "physical_participant_ids":v.physical_participant_ids,
                        "source_object_ids":v.source_object_ids,
                        "source_anchor_ids":v.source_anchor_ids,
                        "instance_anchor_ids":v.instance_anchor_ids,
                        "operation_family":v.operation_family,
                        "parameter_kind":v.parameter_kind,
                        "axis":v.axis,
                        "edge":v.edge,
                        "authority_status":v.authority_status,
                        "compiler_status":v.compiler_status.value,
                        "compiler_capability_id":v.capability.capability_id,
                        "compiler_primitive":v.capability.primitive,
                        "compiler_reason_codes":v.capability.reason_codes,
                        "requires_occurrence_specialization":v.requires_instance_specialization,
                        "protected_relation_ids":v.protected_relation_ids,
                        "connectivity_risk":v.connectivity_risk,
                        "legal_intervals_dbu":v.legal_intervals_dbu,
                        "grid_dbu":v.manufacturing_grid_dbu,
                    } for v in variants],
                    "rejection_codes":sorted({
                        failure.code.value for failure in dof_failures
                    }),
                    "rejection_reasons":sorted({
                        failure.reason for failure in dof_failures
                    }),
                    "required_checks":["PREFERRED_FORBIDDEN_REBIND","PRIMARY_AND_PROTECTED_SOLVE",
                        "ATOMIC_COMPILATION","FRESH_DRC_SANITY_CONNECTIVITY"],
                }
        registry = build_plan_semantic_registry(
            context=formal_context,
            scenes=scenes,
            dofs=dofs,
            current_context_fingerprint=current.context.context_fingerprint,
            run_id=execution.run_id,
            iteration=execution.iteration,
        )
        install_plan_semantic_registry(formal_context, registry)
        self._event(
            "repair_kernel_region_prepared",
            region_id=execution.region.region_id,
            supported_violation_count=len(supported_violation_ids),
            scene_count=len(scenes),
            dof_count=len(dofs),
            scene_failure_count=len(scene_failures),
            plan_semantic_registry_id=registry.registry_id,
        )
        return current, scenes, dofs, scene_failures, supported_violation_ids

    async def plan_symbolic(
        self,
        *,
        formal_context,
        execution: FormalKernelExecutionContext,
        reports: list[KernelSupportReport],
        current_semantic: FormalCurrentSemanticResult | None = None,
        scenes: list[RepairScene] | None = None,
        dofs: list[RepairDOF] | None = None,
    ) -> KernelRepairPlan:
        if current_semantic is None or scenes is None or dofs is None:
            current_semantic, scenes, dofs, _, _ = self.prepare_region_inputs(
                formal_context=formal_context,
                execution=execution,
                reports=reports,
            )
        if self.config.llm.enabled and self.llm is None:
            raise RuntimeError(
                "LLM_REQUIRED_BUT_UNAVAILABLE: formal LLM run is fail-closed"
            )
        planner = SymbolicKernelPlanner(
            self.llm,
            prompt_version="p5-formal-registry-evodrc-skill-v1",
            event_sink=self._event,
            project_root=self.project_root,
            evodrc_skill_enabled=(
                self.config.llm.enabled
                and self.config.llm.provider != "fake"
                and self.config.llm.evodrc_skill_enabled
            ),
            evodrc_skill_root=self.config.llm.evodrc_skill_root,
        )
        try:
            return await planner.plan(
                context=formal_context, scenes=scenes, dofs=dofs,
                support_reports=reports,
                neighbor_messages=execution.neighbor_messages,
                blueprint=execution.blueprint,
                failure_memory=execution.failure_memory,
            )
        except Exception as exc:
            self._event(
                "repair_kernel_plan_failed",
                region_id=execution.region.region_id,
                error_type=type(exc).__name__,
            )
            if self.config.llm.enabled or self.llm is not None:
                raise
            plan = planner.deterministic_default(formal_context, scenes, dofs)
            self._event(
                "repair_kernel_deterministic_plan_created",
                region_id=execution.region.region_id,
                plan_id=plan.plan_id,
            )
            return plan

    async def plan(
        self,
        *,
        formal_context,
        execution: FormalKernelExecutionContext,
        reports: list[KernelSupportReport],
        symbolic_plan: KernelRepairPlan | None = None,
        round_two_messages: list | None = None,
        current_semantic: FormalCurrentSemanticResult | None = None,
        scenes: list[RepairScene] | None = None,
        dofs: list[RepairDOF] | None = None,
        scene_failures: list[dict[str, Any]] | None = None,
        supported_violation_ids: set[str] | None = None,
    ) -> KernelRegionResult:
        noop = ensure_noop(execution.region, execution.subgraph.subgraph_id)
        if any(item is None for item in (
            current_semantic, scenes, dofs, scene_failures,
            supported_violation_ids,
        )):
            (
                current_semantic, scenes, dofs, scene_failures,
                supported_violation_ids,
            ) = self.prepare_region_inputs(
                formal_context=formal_context,
                execution=execution,
                reports=reports,
            )
        assert current_semantic is not None
        assert scenes is not None
        assert dofs is not None
        assert scene_failures is not None
        assert supported_violation_ids is not None
        current = current_semantic
        if not isinstance(
            getattr(formal_context, "plan_semantic_registry", None), dict,
        ):
            registry = build_plan_semantic_registry(
                context=formal_context,
                scenes=scenes,
                dofs=dofs,
                current_context_fingerprint=(
                    current.context.context_fingerprint
                ),
                run_id=execution.run_id,
                iteration=execution.iteration,
            )
            install_plan_semantic_registry(formal_context, registry)
        self._event(
            "repair_kernel_current_context_built",
            region_id=execution.region.region_id,
            context_fingerprint=current.context.context_fingerprint,
        )
        supported_reports = [item for item in reports if item.status in _SUPPORTED]
        supported_rule_ids = sorted({item.rule_id for item in supported_reports})
        violations = [
            item for item in formal_context.region_violations
            if item.violation_id in supported_violation_ids
        ]
        planner = SymbolicKernelPlanner(
            self.llm,
            prompt_version="p5-formal-registry-evodrc-skill-v1",
            event_sink=self._event,
            project_root=self.project_root,
            evodrc_skill_enabled=(
                self.config.llm.enabled
                and self.config.llm.provider != "fake"
                and self.config.llm.evodrc_skill_enabled
            ),
            evodrc_skill_root=self.config.llm.evodrc_skill_root,
        )
        declared_origin = getattr(self.llm, "plan_origin", None)
        if declared_origin is not None and declared_origin not in {
            "REAL_LLM", "FROZEN_REAL_LLM_REPLAY", "FAKE_LLM",
            "DETERMINISTIC_TEST",
        }:
            raise ValueError("INVALID_DECLARED_PLAN_ORIGIN")
        origin: PlanOrigin = declared_origin or (
            "DETERMINISTIC_TEST" if self.llm is None
            else "FAKE_LLM" if self.config.llm.provider == "fake"
            else "REAL_LLM"
        )

        def schema_rejected(
            exc: LLMStructuredResponseError,
            attempted_plans: list[KernelRepairPlan],
        ) -> KernelRegionResult:
            self._event(
                "repair_kernel_plan_schema_rejected",
                region_id=execution.region.region_id,
                error_type=type(exc).__name__,
            )
            return KernelRegionResult(
                region_id=execution.region.region_id,
                subgraph_id=execution.subgraph.subgraph_id,
                candidates=[], noop_candidate=noop,
                failures=[{
                    "code": "PLAN_SCHEMA_REJECTED",
                    "message": str(exc)[:500],
                }],
                supported_rule_ids=supported_rule_ids,
                unsupported_rule_ids=sorted({
                    item.rule_id for item in reports
                    if item.status not in _SUPPORTED
                }),
                support_reports=reports,
                repair_focus_refs=[item.focus.focus_id for item in scenes],
                repair_scene_refs=[item.scene_id for item in scenes],
                repair_dof_refs=[item.dof_id for item in dofs],
                symbolic_plan_refs=[item.plan_id for item in attempted_plans],
                final_plan_id=(
                    attempted_plans[-1].plan_id if attempted_plans else None
                ),
                plan_origin=origin,
                scenes=[item.model_dump(mode="json") for item in scenes],
                dofs=[item.model_dump(mode="json") for item in dofs],
                plans=[item.model_dump(mode="json") for item in attempted_plans],
                planning_context_hash=formal_context.context_hash,
            )
        if symbolic_plan is None:
            try:
                plan = await self.plan_symbolic(
                    formal_context=formal_context,
                    execution=execution,
                    reports=reports,
                    current_semantic=current,
                    scenes=scenes,
                    dofs=dofs,
                )
            except LLMStructuredResponseError as exc:
                return schema_rejected(exc, [])
        else:
            plan = symbolic_plan
        if round_two_messages:
            try:
                plan = await planner.finalize(
                    context=formal_context,
                    provisional_plan=plan,
                    round_two_messages=round_two_messages,
                    allowed_ids=allowed_binding_ids(scenes, dofs),
                )
            except LLMStructuredResponseError as exc:
                return schema_rejected(exc, [plan])
        binding_report = None
        plans = [plan]
        try:
            bound_plan, binding_report = bind_kernel_plan(
                plan,
                origin=origin,
                supported_violation_ids=supported_violation_ids,
                scenes=scenes,
                dofs=dofs,
                configured_max_depth=(
                    self.config.repair_kernel_integration.max_trajectory_depth
                ),
                witness_relation_by_violation={
                    violation_id: witness.relation_type
                    for violation_id, witness
                    in current.context.rule_witnesses.items()
                },
                registry=formal_context.plan_semantic_registry,
            )
        except PlanBindingError as first_error:
            binding_report = first_error.report
            self._event(
                "repair_kernel_plan_binding_failed",
                region_id=execution.region.region_id,
                plan_id=plan.plan_id,
                origin=origin,
                errors=binding_report.errors,
            )
            if self.llm is not None:
                try:
                    revised = await planner.revise(
                        context=formal_context,
                        original_plan=plan,
                        binding_errors=binding_report.errors,
                        allowed_ids=allowed_binding_ids(scenes, dofs),
                    )
                except LLMStructuredResponseError as exc:
                    return schema_rejected(exc, [plan])
                plans.append(revised)
                plan = revised
                try:
                    bound_plan, binding_report = bind_kernel_plan(
                        plan,
                        origin=origin,
                        supported_violation_ids=supported_violation_ids,
                        scenes=scenes,
                        dofs=dofs,
                        configured_max_depth=(
                            self.config.repair_kernel_integration.max_trajectory_depth
                        ),
                        witness_relation_by_violation={
                            violation_id: witness.relation_type
                            for violation_id, witness
                            in current.context.rule_witnesses.items()
                        },
                        registry=formal_context.plan_semantic_registry,
                    )
                except PlanBindingError as second_error:
                    binding_report = second_error.report
                    self._event(
                        "repair_kernel_plan_binding_failed",
                        region_id=execution.region.region_id,
                        plan_id=plan.plan_id,
                        origin=origin,
                        errors=binding_report.errors,
                        final=True,
                    )
                    return KernelRegionResult(
                        region_id=execution.region.region_id,
                        subgraph_id=execution.subgraph.subgraph_id,
                        candidates=[], noop_candidate=noop,
                        failures=[{
                            "code": "LLM_PLAN_UNBINDABLE",
                            "message": "; ".join(binding_report.errors),
                        }],
                        supported_rule_ids=supported_rule_ids,
                        unsupported_rule_ids=sorted({
                            item.rule_id for item in reports
                            if item.status not in _SUPPORTED
                        }),
                        support_reports=reports,
                        repair_focus_refs=[item.focus.focus_id for item in scenes],
                        repair_scene_refs=[item.scene_id for item in scenes],
                        repair_dof_refs=[item.dof_id for item in dofs],
                        symbolic_plan_refs=[item.plan_id for item in plans],
                        final_plan_id=plan.plan_id,
                        plan_origin=origin,
                        plan_binding_report=binding_report.model_dump(mode="json"),
                        scenes=[item.model_dump(mode="json") for item in scenes],
                        dofs=[item.model_dump(mode="json") for item in dofs],
                        plans=[item.model_dump(mode="json") for item in plans],
                        planning_context_hash=formal_context.context_hash,
                    )
            else:
                return KernelRegionResult(
                    region_id=execution.region.region_id,
                    subgraph_id=execution.subgraph.subgraph_id,
                    candidates=[], noop_candidate=noop,
                    failures=[{
                        "code": "LLM_PLAN_UNBINDABLE",
                        "message": "; ".join(binding_report.errors),
                    }],
                    supported_rule_ids=supported_rule_ids,
                    support_reports=reports,
                    final_plan_id=plan.plan_id,
                    plan_origin=origin,
                    plan_binding_report=binding_report.model_dump(mode="json"),
                    plans=[item.model_dump(mode="json") for item in plans],
                    planning_context_hash=formal_context.context_hash,
                )
        self._event(
            "repair_kernel_final_plan_created",
            region_id=execution.region.region_id,
            plan_id=plan.plan_id,
            origin=origin,
        )
        self._event(
            "repair_kernel_plan_binding_passed",
            region_id=execution.region.region_id,
            plan_id=bound_plan.plan_id,
            origin=origin,
            selected_violation_ids=bound_plan.selected_violation_ids,
            filtered_out_violation_count=(
                len(violations) - len(bound_plan.selected_violation_ids)
            ),
            binding_effects=bound_plan.binding_effects,
        )
        violations = [
            item for item in violations
            if item.violation_id in set(bound_plan.selected_violation_ids)
        ]

        execution_feedback: list[dict[str, Any]] = []

        def build_root_proposals(target, selected_bound_plan):
            diagnostics: list[dict[str, Any]] = []
            try:
                target_scene = next(
                    item for item in scenes
                    if item.focus.primary_violation_id == target.violation_id
                )
                target_dofs = [
                    item for item in dofs if item.scene_id == target_scene.scene_id
                ]
                values = FormalRootProposalAdapter(max_proposals=12).build(
                    execution,
                    current.context,
                    target,
                    selected_bound_plan,
                    scene=target_scene,
                    dofs=target_dofs,
                    diagnostics=diagnostics,
                )
                return values, diagnostics, target_scene, target_dofs, None
            except (KeyError, StopIteration, ValueError, RuntimeError) as exc:
                return [], diagnostics, None, [], exc

        root_cache = {
            item.violation_id: build_root_proposals(item, bound_plan)
            for item in violations
        }
        if plan.preferred_dof_ids or plan.forbidden_dof_ids:
            for target in violations:
                strict, diagnostics, target_scene, target_dofs, error = root_cache[
                    target.violation_id
                ]
                if strict or error is not None or target_scene is None:
                    continue
                all_dof_ids = sorted({item.dof_id for item in target_dofs})
                probe_bound = bound_plan.model_copy(update={
                    "allowed_dof_ids": all_dof_ids,
                    "forbidden_dof_ids": [],
                })
                probe, _, _, _, probe_error = build_root_proposals(
                    target, probe_bound
                )
                operation_bound_expanded = False
                if (
                    probe_error is None
                    and not probe
                    and plan.max_operation_count < 4
                    and any(
                        "EXEC_OPERATION_BOUND_EXCEEDED"
                        in item.get("compile_failure_codes", [])
                        for item in diagnostics
                    )
                ):
                    probe_bound = probe_bound.model_copy(update={
                        "max_operation_count": 4,
                    })
                    probe, _, _, _, probe_error = build_root_proposals(
                        target, probe_bound
                    )
                    operation_bound_expanded = bool(probe)
                if probe_error is not None or not probe:
                    continue
                current_allowed = set(bound_plan.allowed_dof_ids)
                alternative_operation_bounds: dict[tuple[str, ...], int] = {}
                for proposal in probe:
                    option = tuple(sorted(
                        str(item) for item in proposal.executable_binding.get(
                            "selected_dof_ids", []
                        )
                    ))
                    if not option or (
                        not operation_bound_expanded
                        and set(option) <= current_allowed
                    ):
                        continue
                    required = int(proposal.executable_binding.get(
                        "atomic_operation_count",
                        probe_bound.max_operation_count,
                    ))
                    alternative_operation_bounds[option] = min(
                        required,
                        alternative_operation_bounds.get(option, required),
                    )
                alternative_sets = sorted(alternative_operation_bounds)
                if not alternative_sets:
                    continue
                feasible_alternatives = sorted({
                    dof_id for option in alternative_sets for dof_id in option
                })
                registry = formal_context.plan_semantic_registry
                registry_payload = (
                    registry.model_dump(mode="json")
                    if hasattr(registry, "model_dump") else dict(registry or {})
                )
                registry_dofs = registry_payload.get("dofs", {})
                alternative_facts = {
                    dof_id: {
                        key: registry_dofs[dof_id][key]
                        for key in (
                            "carrier_kinds", "atomic_co_dof_ids",
                            "atomic_source_target_ids", "allowed_intervals_dbu",
                            "manufacturing_grid_dbu", "protected_relation_ids",
                            "compiler_status", "required_checks",
                        )
                        if key in registry_dofs.get(dof_id, {})
                    }
                    for dof_id in feasible_alternatives
                    if dof_id in registry_dofs
                }
                feedback = {
                    "kind": "PREFERRED_DOF_INFEASIBLE",
                    "original_plan_id": plan.plan_id,
                    "target_violation_id": target.violation_id,
                    "preferred_dof_ids": list(plan.preferred_dof_ids),
                    "forbidden_dof_ids": list(plan.forbidden_dof_ids),
                    "strict_root_proposal_count": 0,
                    "counterfactual_root_proposal_count": len(probe),
                    "feasible_alternative_dof_ids": feasible_alternatives,
                    "feasible_alternative_dof_sets": [
                        list(item) for item in alternative_sets
                    ],
                    "feasible_alternative_operation_bounds": [
                        {
                            "dof_ids": list(item),
                            "minimum_max_operation_count": (
                                alternative_operation_bounds[item]
                            ),
                        }
                        for item in alternative_sets
                    ],
                    "feasible_alternative_facts": alternative_facts,
                    "reason_codes": [
                        "STRICT_PREFERRED_SET_NOT_EXECUTABLE",
                        "COUNTERFACTUAL_LEGAL_ALTERNATIVE_EXISTS",
                        *(
                            ["COUNTERFACTUAL_OPERATION_BOUND_EXPANSION_REQUIRED"]
                            if operation_bound_expanded else []
                        ),
                    ],
                    "strict_root_diagnostics": diagnostics,
                }
                execution_feedback.append(feedback)
                self._event(
                    "repair_kernel_execution_feedback_generated",
                    region_id=execution.region.region_id,
                    violation_id=target.violation_id,
                    kind=feedback["kind"],
                    strict_root_proposal_count=0,
                    counterfactual_root_proposal_count=(
                        len(probe)
                    ),
                )
                if self.llm is None:
                    break
                original_plan = plan
                try:
                    revised = await planner.revise_after_execution_feedback(
                        context=formal_context,
                        original_plan=original_plan,
                        execution_feedback=feedback,
                        allowed_ids=allowed_binding_ids(scenes, dofs),
                    )
                    plans.append(revised)
                    decision = validate_execution_feedback_revision(
                        original_plan,
                        revised,
                        kind="PREFERRED_DOF_INFEASIBLE",
                        allowed_ids=allowed_binding_ids(scenes, dofs),
                        configured_max_depth=(
                            self.config.repair_kernel_integration.max_trajectory_depth
                        ),
                        feasible_alternative_dof_sets=[
                            list(item) for item in alternative_sets
                        ],
                        feasible_alternative_operation_bounds=feedback[
                            "feasible_alternative_operation_bounds"
                        ],
                    )
                    if decision == "DECLINED":
                        self._event(
                            "repair_kernel_execution_revision_declined",
                            region_id=execution.region.region_id,
                            original_plan_id=original_plan.plan_id,
                            revised_plan_id=revised.plan_id,
                            kind=feedback["kind"],
                        )
                        raise ExecutionFeedbackRevisionError(
                            "EXECUTION_REVISION_PREFERRED_DOF_UNCHANGED"
                        )
                    revised_bound, revised_report = bind_kernel_plan(
                        revised,
                        origin=origin,
                        supported_violation_ids=supported_violation_ids,
                        scenes=scenes,
                        dofs=dofs,
                        configured_max_depth=(
                            self.config.repair_kernel_integration.max_trajectory_depth
                        ),
                        witness_relation_by_violation={
                            violation_id: witness.relation_type
                            for violation_id, witness
                            in current.context.rule_witnesses.items()
                        },
                        registry=formal_context.plan_semantic_registry,
                    )
                except LLMStructuredResponseError as exc:
                    return schema_rejected(exc, plans)
                except (
                    ExecutionFeedbackRevisionError,
                    PlanBindingError,
                    KeyError,
                    ValueError,
                    RuntimeError,
                ) as exc:
                    self._event(
                        "repair_kernel_execution_revision_rejected",
                        region_id=execution.region.region_id,
                        original_plan_id=original_plan.plan_id,
                        kind=feedback["kind"],
                        reason=str(exc),
                    )
                    return KernelRegionResult(
                        region_id=execution.region.region_id,
                        subgraph_id=execution.subgraph.subgraph_id,
                        candidates=[], noop_candidate=noop,
                        failures=[{
                            "code": "INVALID_EXECUTION_REVISION",
                            "message": str(exc),
                        }],
                        supported_rule_ids=supported_rule_ids,
                        support_reports=reports,
                        repair_focus_refs=[item.focus.focus_id for item in scenes],
                        repair_scene_refs=[item.scene_id for item in scenes],
                        repair_dof_refs=[item.dof_id for item in dofs],
                        symbolic_plan_refs=[item.plan_id for item in plans],
                        final_plan_id=original_plan.plan_id,
                        plan_origin=origin,
                        plan_binding_report=binding_report.model_dump(mode="json"),
                        scenes=[item.model_dump(mode="json") for item in scenes],
                        dofs=[item.model_dump(mode="json") for item in dofs],
                        plans=[item.model_dump(mode="json") for item in plans],
                        execution_feedback=execution_feedback,
                        planning_context_hash=formal_context.context_hash,
                    )
                plan = revised
                bound_plan = revised_bound
                binding_report = revised_report
                root_cache = {
                    item.violation_id: build_root_proposals(item, bound_plan)
                    for item in violations
                }
                self._event(
                    "repair_kernel_execution_revision_accepted",
                    region_id=execution.region.region_id,
                    original_plan_id=original_plan.plan_id,
                    revised_plan_id=plan.plan_id,
                    kind=feedback["kind"],
                )
                break

        execution_revision_counts: dict[tuple[str, str], int] = {}
        candidates: list[RepairCandidate] = []
        trajectory_refs: list[str] = []
        evidence_refs: list[ArtifactRef] = []
        failures = list(scene_failures)
        verification_deferred = False
        for violation in violations:
            proposals, root_diagnostics, _, _, root_error = root_cache[
                violation.violation_id
            ]
            if root_error is not None:
                exc = root_error
                failures.append({
                    "code": "FORMAL_TRAJECTORY_EXECUTION_FAIL",
                    "violation_id": violation.violation_id,
                    "message": str(exc),
                })
                continue
            self._event(
                "repair_kernel_root_proposals_built",
                region_id=execution.region.region_id,
                violation_id=violation.violation_id,
                proposal_count=len(proposals),
            )
            if not proposals:
                failures.append({
                    "code": "ROOT_PROPOSAL_UNAVAILABLE",
                    "violation_id": violation.violation_id,
                    "message": "no deterministic root proposal",
                    "failure_stage": "ROOT_PROPOSAL",
                    "failure_reason": (
                        root_diagnostics[-1].get("failure_reason")
                        if root_diagnostics else "ROOT_NO_SCENE"
                    ),
                    "root_diagnostics": root_diagnostics,
                })
                self._event(
                    "repair_kernel_trajectory_skipped",
                    region_id=execution.region.region_id,
                    violation_id=violation.violation_id,
                    reason="ROOT_PROPOSAL_UNAVAILABLE",
                )
                continue

            for proposal in proposals:
                plan_revision = max(0, len(plans) - 1)
                proposal_dir = self._proposal_evidence_dir(
                    execution=execution,
                    bound_plan=bound_plan,
                    violation=violation,
                    proposal=proposal,
                )
                self._persist_root_registration(
                    path=proposal_dir,
                    execution=execution,
                    bound_plan=bound_plan,
                    proposal=proposal,
                    violation=violation,
                    root_diagnostics=root_diagnostics,
                    plan_revision=plan_revision,
                    plan_semantic_registry=formal_context.plan_semantic_registry,
                )
                self._event(
                    "repair_kernel_root_proposal_attempted",
                    region_id=execution.region.region_id,
                    violation_id=violation.violation_id,
                    plan_id=bound_plan.plan_id,
                    proposal_id=proposal.proposal_id,
                )
                trajectory_dir = (
                    proposal_dir / "trajectory" / "search_00"
                )
                trajectory_control = ExecutionControl()
                try:
                    outcome = await self._run_eda(
                        kind="TRAJECTORY",
                        region_id=execution.region.region_id,
                        violation_id=violation.violation_id,
                        proposal_id=proposal.proposal_id,
                        execution_control=trajectory_control,
                        fn=partial(
                            FormalTrajectoryEngine(
                                self.project_root, self.config,
                            ).search,
                            execution=execution,
                            violation=violation,
                            root_proposals=[proposal],
                            bound_plan=bound_plan,
                            output_dir=trajectory_dir,
                            max_trajectories=1,
                            root_semantic=current.context,
                            execution_control=trajectory_control,
                            shared_live_budget=self.live_evaluation_budget,
                        ),
                    )
                except InfrastructureFailure as exc:
                    if _reserved_verification_stop(exc):
                        verification_deferred = True
                        failures.append({"code":"DEFERRED_BUDGET", "physical_outcome":None,
                            "violation_id":violation.violation_id,
                            "message":"Search stopped; required verification capacity preserved."})
                        self._event("repair_kernel_search_deferred_budget",
                            region_id=execution.region.region_id, retained_candidates=len(candidates))
                        break
                    raise
                except (KeyError, ValueError, RuntimeError) as exc:
                    failures.append({
                        "code": "FORMAL_TRAJECTORY_EXECUTION_FAIL",
                        "violation_id": violation.violation_id,
                        "message": str(exc),
                    })
                    continue
                for event in outcome.events:
                    if event.get("phase4_classification") == "ADMITTED_TEMPORARY_DEBT":
                        self._event(
                            "repair_kernel_temporary_debt_admitted",
                            region_id=execution.region.region_id,
                            violation_id=violation.violation_id,
                            plan_id=bound_plan.plan_id,
                            depth=event.get("depth"),
                            proposal_id=event.get("proposal_id"),
                        )
                    if (
                        event.get("stage") == "ORACLE"
                        and int(event.get("depth", 0)) > 0
                        and event.get("status") == "PASS"
                    ):
                        self._event(
                            "repair_kernel_current_debt_rebound",
                            region_id=execution.region.region_id,
                            violation_id=violation.violation_id,
                            plan_id=bound_plan.plan_id,
                            depth=event.get("depth"),
                            binding_count=event.get("binding_count", 0),
                        )
                self._event(
                    "repair_kernel_trajectory_started",
                    region_id=execution.region.region_id,
                    violation_id=violation.violation_id,
                    plan_id=bound_plan.plan_id,
                    proposal_id=proposal.proposal_id,
                    sandbox_calls=outcome.sandbox_calls,
                )
                if outcome.sandbox_calls:
                    self._event(
                        "repair_kernel_trajectory_evaluated",
                        region_id=execution.region.region_id,
                        violation_id=violation.violation_id,
                        plan_id=bound_plan.plan_id,
                        proposal_id=proposal.proposal_id,
                        sandbox_calls=outcome.sandbox_calls,
                        physical_verdict=outcome.status,
                        deepest_failure=outcome.deepest_failure,
                        fresh_physical_evaluated=True,
                    )
                execution_retry_performed = False
                feedback = getattr(outcome, "execution_feedback", None)
                if feedback is not None:
                    feedback_kind = str(feedback["kind"])
                    revision_key = (violation.violation_id, feedback_kind)
                    revision_limit = (
                        2 if feedback_kind == "PHYSICAL_CONSTRAINT_FAILURE"
                        else 1
                    )
                    execution_feedback.append(feedback)
                    self._event(
                        "repair_kernel_execution_feedback_generated",
                        region_id=execution.region.region_id,
                        violation_id=violation.violation_id,
                        plan_id=bound_plan.plan_id,
                        proposal_id=proposal.proposal_id,
                        kind=feedback["kind"],
                        debt_binding_count=feedback.get("debt_binding_count", 0),
                        debt_proposal_count=feedback.get("debt_proposal_count", 0),
                    )
                    if (
                        execution_revision_counts.get(revision_key, 0)
                        >= revision_limit
                        or self.llm is None
                    ):
                        failures.append({
                            "code": (
                                "PHYSICAL_REPLAN_LIMIT_REACHED"
                                if feedback_kind == "PHYSICAL_CONSTRAINT_FAILURE"
                                else "DEPTH_EXTENSION_APPROVAL_REQUIRED"
                            ),
                            "violation_id": violation.violation_id,
                            "message": (
                                "LLM execution revision unavailable or bounded "
                                "revision limit reached"
                            ),
                        })
                        break
                    execution_revision_counts[revision_key] = (
                        execution_revision_counts.get(revision_key, 0) + 1
                    )
                    original_plan = plan
                    try:
                        revised = await planner.revise_after_execution_feedback(
                            context=formal_context,
                            original_plan=original_plan,
                            execution_feedback=feedback,
                            allowed_ids=allowed_binding_ids(scenes, dofs),
                        )
                        plans.append(revised)
                        decision = validate_execution_feedback_revision(
                            original_plan,
                            revised,
                            kind=feedback_kind,
                            allowed_ids=allowed_binding_ids(scenes, dofs),
                            configured_max_depth=(
                                self.config.repair_kernel_integration.max_trajectory_depth
                            ),
                        )
                        if decision == "DECLINED":
                            self._event(
                                "repair_kernel_execution_revision_declined",
                                region_id=execution.region.region_id,
                                original_plan_id=original_plan.plan_id,
                                revised_plan_id=revised.plan_id,
                                kind=feedback["kind"],
                            )
                            failures.append({
                                "code": (
                                    "PHYSICAL_REPLAN_DECLINED"
                                    if feedback_kind
                                    == "PHYSICAL_CONSTRAINT_FAILURE"
                                    else "DEPTH_EXTENSION_DECLINED"
                                ),
                                "violation_id": violation.violation_id,
                                "message": "LLM retained the original execution semantics",
                            })
                            break
                        revised_bound, revised_report = bind_kernel_plan(
                            revised,
                            origin=origin,
                            supported_violation_ids=supported_violation_ids,
                            scenes=scenes,
                            dofs=dofs,
                            configured_max_depth=(
                                self.config.repair_kernel_integration.max_trajectory_depth
                            ),
                            witness_relation_by_violation={
                                violation_id: witness.relation_type
                                for violation_id, witness
                                in current.context.rule_witnesses.items()
                            },
                            registry=formal_context.plan_semantic_registry,
                        )
                    except LLMStructuredResponseError as exc:
                        self._event(
                            "repair_kernel_plan_schema_rejected",
                            region_id=execution.region.region_id,
                            violation_id=violation.violation_id,
                            error_type=type(exc).__name__,
                        )
                        failures.append({
                            "code": "PLAN_SCHEMA_REJECTED",
                            "violation_id": violation.violation_id,
                            "message": str(exc)[:500],
                        })
                        break
                    except (
                        ExecutionFeedbackRevisionError,
                        PlanBindingError,
                        KeyError,
                        ValueError,
                        RuntimeError,
                    ) as exc:
                        self._event(
                            "repair_kernel_execution_revision_rejected",
                            region_id=execution.region.region_id,
                            original_plan_id=original_plan.plan_id,
                            kind=feedback["kind"],
                            reason=str(exc),
                        )
                        failures.append({
                            "code": "INVALID_EXECUTION_REVISION",
                            "violation_id": violation.violation_id,
                            "message": str(exc),
                        })
                        break
                    plan = revised
                    bound_plan = revised_bound
                    binding_report = revised_report
                    rebuilt, rebuilt_diagnostics, _, _, rebuilt_error = (
                        build_root_proposals(violation, bound_plan)
                    )
                    root_cache[violation.violation_id] = (
                        rebuilt, rebuilt_diagnostics, None, [], rebuilt_error
                    )
                    if rebuilt_error is not None or not rebuilt:
                        failures.append({
                            "code": "ROOT_PROPOSAL_UNAVAILABLE_AFTER_REVISION",
                            "violation_id": violation.violation_id,
                            "message": str(rebuilt_error or "no revised root proposal"),
                        })
                        break
                    proposal = rebuilt[0]
                    plan_revision = max(0, len(plans) - 1)
                    proposal_dir = self._proposal_evidence_dir(
                        execution=execution,
                        bound_plan=bound_plan,
                        violation=violation,
                        proposal=proposal,
                    )
                    self._persist_root_registration(
                        path=proposal_dir,
                        execution=execution,
                        bound_plan=bound_plan,
                        proposal=proposal,
                        violation=violation,
                        root_diagnostics=rebuilt_diagnostics,
                        plan_revision=plan_revision,
                        plan_semantic_registry=formal_context.plan_semantic_registry,
                    )
                    self._event(
                        "repair_kernel_execution_revision_accepted",
                        region_id=execution.region.region_id,
                        original_plan_id=original_plan.plan_id,
                        revised_plan_id=plan.plan_id,
                        kind=feedback["kind"],
                    )
                    execution_retry_performed = True
                    trajectory_control = ExecutionControl()
                    try:
                        outcome = await self._run_eda(
                            kind="TRAJECTORY",
                            region_id=execution.region.region_id,
                            violation_id=violation.violation_id,
                            proposal_id=proposal.proposal_id,
                            execution_control=trajectory_control,
                            fn=partial(
                                FormalTrajectoryEngine(
                                    self.project_root, self.config,
                                ).search,
                                execution=execution,
                                violation=violation,
                                root_proposals=[proposal],
                                bound_plan=bound_plan,
                                output_dir=(
                                    proposal_dir / "trajectory" / "search_01"
                                ),
                                max_trajectories=1,
                                    root_semantic=current.context,
                                    execution_control=trajectory_control,
                                    shared_live_budget=(
                                        self.live_evaluation_budget
                                    ),
                                ),
                        )
                    except InfrastructureFailure as exc:
                        if _reserved_verification_stop(exc):
                            verification_deferred = True
                            failures.append({"code":"DEFERRED_BUDGET", "physical_outcome":None,
                                "violation_id":violation.violation_id})
                            break
                        raise
                    except (KeyError, ValueError, RuntimeError) as exc:
                        failures.append({
                            "code": "FORMAL_TRAJECTORY_EXECUTION_FAIL",
                            "violation_id": violation.violation_id,
                            "message": str(exc),
                        })
                        break
                    self._event(
                        "repair_kernel_trajectory_started",
                        region_id=execution.region.region_id,
                        violation_id=violation.violation_id,
                        plan_id=bound_plan.plan_id,
                        proposal_id=proposal.proposal_id,
                        sandbox_calls=outcome.sandbox_calls,
                        execution_revision=True,
                    )
                    if outcome.sandbox_calls:
                        self._event(
                            "repair_kernel_trajectory_evaluated",
                            region_id=execution.region.region_id,
                            violation_id=violation.violation_id,
                            plan_id=bound_plan.plan_id,
                            proposal_id=proposal.proposal_id,
                            sandbox_calls=outcome.sandbox_calls,
                            physical_verdict=outcome.status,
                            deepest_failure=outcome.deepest_failure,
                            fresh_physical_evaluated=True,
                            execution_revision=True,
                        )
                if verification_deferred:
                    break
                if outcome.status != "FINAL_CLEAN" or not outcome.trajectories:
                    failures.append({
                        "code": outcome.deepest_failure or "KERNEL_NO_CANDIDATE",
                        "violation_id": violation.violation_id,
                        "message": "bounded trajectory did not reach strict clean",
                    })
                    if execution_retry_performed:
                        break
                    continue
                trajectory = outcome.trajectories[0]
                self._event(
                    "repair_kernel_condensation_started",
                    trajectory_id=trajectory.trajectory_id,
                )
                condensation_dir = (
                    proposal_dir / "condensation"
                    / stable_hash(trajectory.trajectory_id)[:20]
                )
                condensation_control = ExecutionControl()
                trajectory_root = getattr(trajectory, "root_snapshot", None)
                condensation_trace = self._trace_context(
                    execution=execution,
                    bound_plan=bound_plan,
                    violation=violation,
                    proposal_id=proposal.proposal_id,
                    parent_snapshot_id=(
                        trajectory_root.snapshot_id
                        if trajectory_root is not None
                        else execution.current_snapshot.snapshot_id
                    ),
                    purpose=ExecutionPurpose.CONDENSATION,
                    depth=len(trajectory.steps),
                    plan_revision=plan_revision,
                    invocation_suffix=trajectory.trajectory_id,
                )
                try:
                    record = await self._run_eda(
                        kind="CONDENSATION",
                        region_id=execution.region.region_id,
                        violation_id=violation.violation_id,
                        proposal_id=proposal.proposal_id,
                        trajectory_id=trajectory.trajectory_id,
                        execution_control=condensation_control,
                        claim_region_budget=True,
                        fn=partial(
                            CondensationEquivalenceGate(
                                self.project_root, self.config,
                            ).verify,
                            execution=execution,
                            trajectory=trajectory,
                            output_dir=condensation_dir,
                            run_id=(
                                f"{execution.run_id}-formal-condensation-"
                                f"{stable_hash([violation.violation_id, proposal.proposal_id])[:12]}"
                            ),
                            trace_context=condensation_trace,
                            execution_control=condensation_control,
                        ),
                    )
                except InfrastructureFailure as exc:
                    if _reserved_verification_stop(exc):
                        verification_deferred = True
                        failures.append({"code":"DEFERRED_BUDGET", "physical_outcome":None,
                            "violation_id":violation.violation_id})
                        break
                    raise
                except (KeyError, ValueError, RuntimeError) as exc:
                    failures.append({
                        "code": "TRAJECTORY_CONDENSATION_FAIL",
                        "violation_id": violation.violation_id,
                        "message": str(exc),
                    })
                    continue
                candidate = RepairCandidate.model_validate(record["candidate"])
                candidates.append(candidate)
                artifact = condensation_dir / "condensation_equivalence.json"
                evidence_ref = ArtifactRef.from_path(
                    artifact,
                    producer="formal-repair-kernel-v2",
                    media_type="application/json",
                    schema_name="CondensationEquivalence",
                )
                evidence_refs.append(evidence_ref)
                if all((
                    record.get("rule_deck_sha256"),
                    record.get("evaluator_sha256"),
                    record.get("connectivity_reference_sha256"),
                )):
                    sandbox_evidence = SandboxEvidence(
                        status=SandboxStatus.CLEAN_PROGRESS,
                        scope=SandboxScope.ISOLATED_CANDIDATE,
                        base_snapshot_id=execution.current_snapshot.snapshot_id,
                        target_violation_removed_ids=(
                            list(candidate.target_violation_ids)
                            if int(record["target_removed"]) > 0 else []
                        ),
                        removed_original_violation_ids=list(
                            record.get("removed_original_violation_ids", [])
                        ),
                        removed_original_marker_fingerprints=list(
                            record.get(
                                "removed_original_marker_fingerprints", [],
                            )
                        ),
                        removed_original_count=int(record["removed_original"]),
                        new_violation_ids=list(record.get("new_violation_ids", [])),
                        new_marker_fingerprints=list(
                            record.get("new_marker_fingerprints", [])
                        ),
                        new_violation_count=int(record["new"]),
                        connectivity_preserved=bool(record["connectivity"]),
                        verification_ref=evidence_ref,
                        attempted_script_sha256=record[
                            "condensed_attempt_script_sha256"
                        ],
                        attempted_gds_sha256=record["attempted_gds_sha256"],
                        attempted_drc_sha256=record["attempted_drc_sha256"],
                        candidate_ids=[candidate.candidate_id],
                        base_script_sha256=record["root_script_sha256"],
                        physical_effect_fingerprint=record[
                            "candidate_physical_effect_fingerprint"
                        ],
                        patch_plan_sha256=record["patch_plan_sha256"],
                        rule_deck_sha256=record["rule_deck_sha256"],
                        evaluator_sha256=record["evaluator_sha256"],
                        connectivity_reference_sha256=record[
                            "connectivity_reference_sha256"
                        ],
                        execution_protocol_version=record.get(
                            "execution_protocol_version", "legacy-v1"
                        ),
                        attempt_id=record.get("attempt_id"),
                        logical_evidence_key=record.get(
                            "logical_evidence_key"
                        ),
                        terminal_status=record.get("terminal_status"),
                        drc_validity=EvidenceValidity(record.get(
                            "drc_validity", EvidenceValidity.NOT_EVALUATED.value,
                        )),
                        sanity_validity=EvidenceValidity(
                            record.get(
                                "sanity_validity",
                                EvidenceValidity.NOT_EVALUATED.value,
                            )
                        ),
                        connectivity_validity=EvidenceValidity(
                            record.get(
                                "connectivity_validity",
                                EvidenceValidity.NOT_EVALUATED.value,
                            )
                        ),
                        fresh_evidence_valid=bool(
                            record.get("fresh_evidence_valid", False)
                        ),
                    )
                    candidate = candidate.model_copy(update={
                        "benefit_evidence": candidate.benefit_evidence.model_copy(
                            update={"sandbox": sandbox_evidence}
                        )
                    })
                    candidates[-1] = candidate
                trajectory_refs.append(trajectory.trajectory_id)
                self._event(
                    "repair_kernel_trajectory_final_clean",
                    region_id=execution.region.region_id,
                    violation_id=violation.violation_id,
                    plan_id=bound_plan.plan_id,
                    proposal_id=proposal.proposal_id,
                    trajectory_id=trajectory.trajectory_id,
                    depth=len(trajectory.steps),
                )
                self._event(
                    "repair_kernel_condensation_equivalence_passed",
                    trajectory_id=trajectory.trajectory_id,
                    candidate_id=candidate.candidate_id,
                )
                self._event(
                    "repair_kernel_target_candidate_accepted",
                    region_id=execution.region.region_id,
                    violation_id=violation.violation_id,
                    plan_id=bound_plan.plan_id,
                    proposal_id=proposal.proposal_id,
                    trajectory_id=trajectory.trajectory_id,
                    candidate_id=candidate.candidate_id,
                )
                break
            if verification_deferred or len(candidates) >= self.max_candidates:
                break
        candidates = deduplicate_candidates(candidates)[: self.max_candidates]
        for candidate in candidates:
            self._event(
                "repair_kernel_llm_bound_candidate_created",
                region_id=execution.region.region_id,
                plan_id=bound_plan.plan_id,
                origin=origin,
                candidate_id=candidate.candidate_id,
            )
        candidates = [
            item.model_copy(update={"rank_from_agent": index})
            for index, item in enumerate(candidates, start=1)
        ]
        if not candidates and not verification_deferred:
            failures.append({
                "code": "KERNEL_NO_CANDIDATE",
                "message": "no fresh-equivalent strict-clean trajectory",
            })
        return KernelRegionResult(
            region_id=execution.region.region_id,
            subgraph_id=execution.subgraph.subgraph_id,
            candidates=candidates,
            noop_candidate=noop,
            trajectory_refs=trajectory_refs,
            kernel_evidence_refs=evidence_refs,
            failures=failures,
            supported_rule_ids=supported_rule_ids,
            unsupported_rule_ids=sorted({
                item.rule_id for item in reports if item.status not in _SUPPORTED
            }),
            support_reports=reports,
            repair_focus_refs=[item.focus.focus_id for item in scenes],
            repair_scene_refs=[item.scene_id for item in scenes],
            repair_dof_refs=[item.dof_id for item in dofs],
            symbolic_plan_refs=[item.plan_id for item in plans],
            final_plan_id=bound_plan.plan_id,
            plan_origin=origin,
            plan_binding_report=binding_report.model_dump(mode="json"),
            scenes=[item.model_dump(mode="json") for item in scenes],
            dofs=[item.model_dump(mode="json") for item in dofs],
            plans=[item.model_dump(mode="json") for item in plans],
            execution_feedback=execution_feedback,
            planning_context_hash=formal_context.context_hash,
        )

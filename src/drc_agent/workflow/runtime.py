from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import BaseModel

from drc_agent.actions.candidates import make_noop_candidate
from drc_agent.agents.guidance import RuleAwareCandidateGuide
from drc_agent.backends.dac26 import DAC26ReportAdapter
from drc_agent.backends.klayout import KLayoutBackend
from drc_agent.backends.execution import EXECUTION_PROTOCOL_VERSION
from drc_agent.backends.transaction import TransactionExecutionResult, TransactionExecutor
from drc_agent.config.loader import AppConfig
from drc_agent.config.methods import (
    MethodPreset, resolve_method, validate_method_identity,
)
from drc_agent.experience.episode import EpisodeBuilder
from drc_agent.coordinator.batching import (
    FailureAttributor, TransactionBatchScheduler,
)
from drc_agent.experience.store import ExperienceStore, NullExperienceStore
from drc_agent.graphs.agent import AgentGraphBuilder, AgentGraphPruner, DesignContext
from drc_agent.graphs.potential_access import PotentialPhysicalAccessBuilder
from drc_agent.graphs.candidate import CandidateGraphBuilder
from drc_agent.llm.providers import resolve_base_url
from drc_agent.llm import (
    FakeLLMClient, LLMClient, LLMAuditLogger, OpenAICompatibleClient,
)
from drc_agent.observability.progress import RunProgressLogger
from drc_agent.observability.metrics import (
    IterationMetrics, IterationMetricsStore, drc_statistics,
    graph_observability_report, llm_usage_for_iteration, mapping_coverage,
    physical_grounding_coverage,
    persist_iteration_metrics,
)
from drc_agent.observability.production_metrics import (
    collect_production_execution_metrics,
)
from drc_agent.patching.compiler import PatchCompiler
from drc_agent.patching.repair_program import build_physical_effect_fingerprint
from drc_agent.regions.builder import RegionBuilder
from drc_agent.regions.connectivity import map_connectivity_components
from drc_agent.regions.parser import SourceObjectMapper, ViolationParser, load_rule_catalog
from drc_agent.regions.physical import SourceHierarchyPhysicalBuilder
from drc_agent.regions.topology import LocalTopologyBuilder
from drc_agent.reliability import (
    FailureCode, InfrastructureFailure, IntegrityFailure, failure_metadata,
)
from drc_agent.repair_kernel_integration.current_semantic_adapter import (
    FormalCurrentSemanticAdapter,
)
from drc_agent.rules import RulePredicateRegistry, RuleWitnessBuilder
from drc_agent.schemas.action import (
    DesignState, JointRepairBundle, RepairCandidate,
)
from drc_agent.schemas.common import (
    ArtifactRef, file_sha256, stable_hash, utc_now,
)
from drc_agent.schemas.state import (
    AgentSubgraph, CandidateAttemptSummary, RegionState, RollbackSummary,
)
from drc_agent.schemas.workflow import (
    ActiveWindow, AgentSubgraphRef, DesignSnapshotRef, StoppingStatus,
    FrontierState, SubgraphPlanResult, TransactionBatch,
    IterationFinalEvidencePointer, IterationTransactionStatus, WindowResult,
)
from drc_agent.schemas.continuation import (
    ContinuationVerification,
    ExecutionEnvelope,
)
from drc_agent.schemas.tools import (
    AttemptTerminalStatus,
    EvidenceValidity,
    ExecutionPurpose,
    ExecutionTraceContext,
)
from drc_agent.utils.artifacts import ArtifactStore
from .checkpoints import CheckpointStore
from .continuation import (
    MAX_CONTINUATION_ITERATIONS,
    append_continuation_record,
    apply_execution_envelope,
    authorized_envelope,
    continuation_lock,
    continuation_nonce,
    cumulative_usage,
    executable_digest,
    execution_envelope,
    initial_continuation_manifest,
    initialize_continuation_files,
    make_continuation_record,
    official_input_identity,
    read_continuation_records,
    run_identity,
    scientific_config_hash,
    state_hashes as continuation_state_hashes,
    verification_result as continuation_verification_result,
    verify_experience_cutoff,
    verify_frontier_state,
    verify_repair_attempt_memory,
    verify_snapshot,
)
from .orchestration import build_workflow, workflow_recursion_limit
from .planning import PlannedSubgraph, SubgraphPlanner
from .sandbox_runtime import RuntimeSandboxService
from .active_window import coordinate_active_window, select_active_window
from drc_agent.repair_kernel_integration.gate import verify_formal_integration_gate
from drc_agent.repair_kernel_integration.audit import (
    formal_integration_runtime_digest,
    integration_code_digest,
    p4_v2_source_digest,
)


class RunResult(BaseModel):
    run_id: str
    run_dir: Path
    status: str
    iteration: int
    region_count: int
    subgraph_count: int
    selected_candidate_ids: list[str]


def stopping_condition(
    *, residual_count: int, connectivity_preserved: bool,
    iteration: int, max_iterations: int, no_progress: int,
    max_no_progress: int, all_noop: bool,
    budget_exhausted: bool = False, integrity_failure: bool = False,
) -> StoppingStatus:
    if integrity_failure:
        return StoppingStatus.INTEGRITY_FAILURE
    if residual_count == 0 and connectivity_preserved:
        return StoppingStatus.DRC_CLEAN
    if all_noop:
        return StoppingStatus.ALL_NOOP
    if no_progress >= max_no_progress:
        return StoppingStatus.NO_PROGRESS
    if iteration >= max_iterations:
        return StoppingStatus.MAX_ITERATIONS
    if budget_exhausted:
        return StoppingStatus.BUDGET_EXHAUSTED
    return StoppingStatus.CONTINUE


def _git_commit(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path,
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


def _source_revision(root: Path) -> str:
    commit = _git_commit(root)
    if commit != "UNAVAILABLE":
        return commit
    files = sorted(
        path for path in (root / "src").rglob("*")
        if path.is_file()
        and (path.suffix == ".py" or ".so" in path.suffixes)
    )
    return "TREE:" + stable_hash({
        str(path.relative_to(root)): file_sha256(path) for path in files
    })


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(value, sort_keys=True, ensure_ascii=True) + "\n"
        for value in values
    )
    partial = path.with_name(path.name + ".partial")
    partial.write_text(payload, encoding="utf-8")
    with partial.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(partial, path)


def partition_supported_pending_frontier(
    *, violations: list, pending: set[str],
    integration_enabled: bool, supported_rule_ids: set[str],
) -> tuple[set[str], set[str], dict[str, str]]:
    """Filter only repair targets; graph nodes remain available as context."""
    if not integration_enabled or not supported_rule_ids:
        return set(pending), set(), {}
    rule_by_violation = {
        item.violation_id: item.rule_id for item in violations
    }
    skipped = {
        violation_id for violation_id in pending
        if rule_by_violation.get(violation_id) not in supported_rule_ids
    }
    return (
        set(pending) - skipped,
        skipped,
        {
            violation_id: rule_by_violation[violation_id]
            for violation_id in skipped
            if violation_id in rule_by_violation
        },
    )


@dataclass
class _Session:
    run_id: str
    case_id: str
    experiment_id: str
    seed: int
    method: str
    preset: MethodPreset
    config: AppConfig
    paths: dict[str, Path]
    run_dir: Path
    store: ArtifactStore
    checkpoints: CheckpointStore
    test_mode: bool = False
    execution_epoch: str = field(
        default_factory=lambda: "epoch_" + uuid.uuid4().hex
    )
    hashes: dict[str, str] = field(default_factory=dict)
    manifest: dict = field(default_factory=dict)
    baseline_snapshot: DesignSnapshotRef | None = None
    current_snapshot: DesignSnapshotRef | None = None
    best_snapshot: DesignSnapshotRef | None = None
    best_total: int = 0
    violations: list = field(default_factory=list)
    objects: list = field(default_factory=list)
    physical_geometries: list = field(default_factory=list)
    rule_predicates: dict = field(default_factory=dict)
    rule_witnesses: dict = field(default_factory=dict)
    rule_knowledge_packs: dict = field(default_factory=dict)
    design: DesignState | None = None
    potential_access_builder: PotentialPhysicalAccessBuilder = field(
        default_factory=PotentialPhysicalAccessBuilder
    )
    potential_access_summaries: dict = field(default_factory=dict)
    regions: list = field(default_factory=list)
    previous_regions: list = field(default_factory=list)
    raw_graph: object | None = None
    graph: object | None = None
    subgraphs: list[AgentSubgraph] = field(default_factory=list)
    plans: dict[str, PlannedSubgraph] = field(default_factory=dict)
    degraded_candidates: dict[str, list[RepairCandidate]] = field(default_factory=dict)
    aggregate_candidates: list[RepairCandidate] = field(default_factory=list)
    aggregate_candidate_graph: object | None = None
    aggregate_bundle: JointRepairBundle | None = None
    scheduled_batches: list[TransactionBatch] = field(default_factory=list)
    patch_plan: object | None = None
    transaction_result: TransactionExecutionResult | None = None
    all_noop: bool = False
    selected_all_noop: bool = False
    explored_non_noop: bool = False
    planning_had_failures: bool = False
    fatal_failure: bool = False
    no_progress: int = 0
    completed_iteration: int = 0
    selected_candidate_ids: list[str] = field(default_factory=list)
    experience_store: object | None = None
    planner: SubgraphPlanner | None = None
    sandbox_service: RuntimeSandboxService | None = None
    transaction_executor: TransactionExecutor | None = None
    llm_log: Path | None = None
    llm_logger: LLMAuditLogger | None = None
    progress: RunProgressLogger | None = None
    active_nodes: set[str] = field(default_factory=set)
    interrupted_node: str | None = None
    final_status: str = "RUNNING"
    run_started_monotonic: float = field(default_factory=time.monotonic)
    active_run_seconds_before: float = 0.0
    resume_state: dict | None = None
    iteration_start_snapshot: DesignSnapshotRef | None = None
    iteration_start_total_drv: int = 0
    iteration_frontier_violation_ids: set[str] = field(default_factory=set)
    processed_frontier_violation_ids: set[str] = field(default_factory=set)
    deferred_frontier_violation_ids: set[str] = field(default_factory=set)
    window_admitted_frontier_violation_ids: set[str] = field(
        default_factory=set
    )
    window_llm_started_frontier_violation_ids: set[str] = field(
        default_factory=set
    )
    window_plan_validated_frontier_violation_ids: set[str] = field(
        default_factory=set
    )
    window_kernel_attempted_frontier_violation_ids: set[str] = field(
        default_factory=set
    )
    window_physical_evaluated_frontier_violation_ids: set[str] = field(
        default_factory=set
    )
    window_attempted_frontier_violation_ids: set[str] = field(
        default_factory=set
    )
    window_unsupported_frontier_violation_ids: set[str] = field(
        default_factory=set
    )
    window_llm_invocation_count: int = 0
    skipped_frontier_violation_ids: set[str] = field(default_factory=set)
    skipped_frontier_by_rule: dict[str, str] = field(default_factory=dict)
    window_index: int = 0
    active_window: ActiveWindow | None = None
    window_results: list[WindowResult] = field(default_factory=list)
    iteration_sandbox_jobs_used: int = 0
    window_coordination: object | None = None
    iteration_exit_reason: str = "FRONTIER_EXHAUSTED"


def _window_checkpoint_state(
    session: _Session, *, in_progress_iteration: int,
) -> dict:
    """Return the minimal JSON state needed for window-boundary recovery."""
    identity = run_identity(
        run_id=session.run_id,
        experiment_id=session.experiment_id,
        case_id=session.case_id,
        method=session.preset.name,
        seed=session.seed,
    )
    return {
        "iteration": session.completed_iteration,
        "execution_epoch": session.execution_epoch,
        "run_identity": identity,
        "run_identity_hash": stable_hash(identity),
        "baseline_snapshot": session.baseline_snapshot.model_dump(mode="json"),
        "scientific_config_hash": scientific_config_hash(session.config),
        "execution_envelope_hash": execution_envelope(
            session.config
        ).content_hash,
        "official_input_identity": official_input_identity(session.paths),
        "experience_knowledge_cutoff": (
            session.experience_store.knowledge_cutoff()
            if (
                session.config.features.experience_graph
                and session.experience_store is not None
            ) else None),
        "last_completed_iteration": session.completed_iteration,
        "in_progress_iteration": in_progress_iteration,
        "current_snapshot": session.current_snapshot.model_dump(mode="json"),
        "best_snapshot": session.best_snapshot.model_dump(mode="json"),
        "best_total": session.best_total,
        "no_progress": session.no_progress,
        "iteration_start_snapshot": (
            session.iteration_start_snapshot.model_dump(mode="json")
            if session.iteration_start_snapshot else None
        ),
        "iteration_start_total_drv": session.iteration_start_total_drv,
        "iteration_frontier_violation_ids": sorted(
            session.iteration_frontier_violation_ids
        ),
        "processed_frontier_violation_ids": sorted(
            session.processed_frontier_violation_ids
        ),
        "deferred_frontier_violation_ids": sorted(
            session.deferred_frontier_violation_ids
        ),
        "skipped_frontier_violation_ids": sorted(
            session.skipped_frontier_violation_ids
        ),
        "skipped_frontier_by_rule": dict(
            sorted(session.skipped_frontier_by_rule.items())
        ),
        "window_index": session.window_index,
        "window_results": [
            item.model_dump(mode="json") for item in session.window_results
        ],
        "iteration_sandbox_jobs_used": session.iteration_sandbox_jobs_used,
        "previous_regions": [
            item.model_dump(mode="json") for item in session.previous_regions
        ],
        "repair_attempt_memory": (
            session.planner.repair_attempt_memory.artifact()
            if session.planner is not None else {}
        ),
    }


def _current_repair_kernel_audit(
    session: _Session, *, audit_complete: bool,
) -> dict:
    """Snapshot truthful static configuration and current kernel counters."""
    existing = dict(
        session.manifest.get("repair_kernel_integration") or {}
    )
    if (
        session.planner is not None
        and session.planner.integration_audit is not None
    ):
        existing.update(
            session.planner.integration_audit.model_dump(mode="json")
        )
    existing.update({
        "config": session.config.repair_kernel_integration.model_dump(
            mode="json"
        ),
        "integration_code_digest": integration_code_digest(),
        "formal_runtime_digest": formal_integration_runtime_digest(
            session.config.project_root
        ),
        "supported_rule_ids": list(
            session.config.repair_kernel_integration.supported_rule_ids
        ),
        "audit_complete": audit_complete,
    })
    existing.update({
        "legacy_candidate_intent_calls": existing.get(
            "legacy_candidate_intent_call_count", 0
        ),
        "legacy_lowerer_calls": existing.get(
            "legacy_lowerer_call_count", 0
        ),
        "legacy_repair_programmer_calls": existing.get(
            "legacy_repair_programmer_call_count", 0
        ),
        "condensed_candidate_count": existing.get(
            "condensed_candidate_count", 0
        ),
    })
    return existing


class ResearchRuntime:
    """DAC26 project-runtime implementation. E0 remains an explicit separate path."""

    def __init__(
        self, cfg: AppConfig, *, llm_client: LLMClient | None = None,
        backend=None, adapter=None, test_mode: bool = False,
    ):
        self.base_cfg = cfg
        self.project_root = cfg.project_root.resolve()
        self.benchmark_root = (
            self.project_root / "benchmarks" / "EvoDRC" /
            "DAC26_DRC_Benchmark"
        )
        self._llm_client = llm_client
        self._backend = backend
        self._adapter = adapter
        self.test_mode = test_mode
        self.session: _Session | None = None

    def _case_paths(self, case_id: str) -> dict[str, Path]:
        base = self.benchmark_root / "testcase" / "asap7" / "block"
        paths = {
            "script": base / "layout_script" / f"{case_id}.py",
            "gds": base / "gds" / f"{case_id}.gds",
            "drc": base / "drc_report" / f"{case_id}.drc.json",
            "connectivity": base / "connectivity" / f"{case_id}.json",
            "rule_deck": (
                self.benchmark_root / "testcase" / "asap7" / "asap7.lydrc"
            ),
            "sanity_helper": self.benchmark_root / "evaluator" / "sanity_check.py",
            "connectivity_helper": (
                self.benchmark_root / "evaluator" / "check_connectivity.py"
            ),
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"case artifacts missing: {missing}")
        return paths

    @staticmethod
    def _validate_run_id(run_id: str) -> None:
        allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-"
        if not run_id or any(char not in allowed for char in run_id):
            raise ValueError("invalid run_id")

    def _new_session(
        self, *, case_id: str, run_id: str, experiment_id: str,
        seed: int, method: str, preset: MethodPreset, config: AppConfig,
    ) -> _Session:
        self._validate_run_id(run_id)
        paths = self._case_paths(case_id)
        run_dir = self.project_root / "runs" / run_id
        if run_dir.exists():
            raise FileExistsError(f"run already exists: {run_dir}")
        run_dir.mkdir(parents=True)
        for directory in [
            "baseline", "iterations", "experience", "score", "metrics",
            "logs", "checkpoints", "continuation",
        ]:
            (run_dir / directory).mkdir(parents=True, exist_ok=True)
        initialize_continuation_files(run_dir)
        return _Session(
            run_id=run_id, case_id=case_id, experiment_id=experiment_id,
            seed=seed, method=method, preset=preset, config=config,
            test_mode=self.test_mode,
            paths=paths, run_dir=run_dir, store=ArtifactStore(run_dir),
            checkpoints=CheckpointStore(
                run_dir / "checkpoints" / "workflow.sqlite"
            ),
            progress=RunProgressLogger(
                run_dir / "logs" / "runtime_events.jsonl"
            ),
        )

    def _resume_session(
        self, *, case_id: str, run_id: str, experiment_id: str,
        seed: int, method: str, preset: MethodPreset, config: AppConfig,
        manifest: dict, resume_state: dict,
        execution_epoch: str | None = None,
    ) -> _Session:
        self._validate_run_id(run_id)
        run_dir = self.project_root / "runs" / run_id
        if not run_dir.is_dir():
            raise FileNotFoundError(f"run does not exist: {run_dir}")
        session = _Session(
            run_id=run_id, case_id=case_id, experiment_id=experiment_id,
            seed=seed, method=method, preset=preset, config=config,
            test_mode=self.test_mode,
            paths=self._case_paths(case_id), run_dir=run_dir,
            store=ArtifactStore(run_dir),
            checkpoints=CheckpointStore(
                run_dir / "checkpoints" / "workflow.sqlite"
            ),
            progress=RunProgressLogger(
                run_dir / "logs" / "runtime_events.jsonl"
            ),
            manifest=manifest, resume_state=resume_state,
            execution_epoch=(
                execution_epoch or "epoch_" + uuid.uuid4().hex
            ),
            active_run_seconds_before=float(
                manifest.get("active_run_seconds", 0.0)
            ),
        )
        session.baseline_snapshot = DesignSnapshotRef.model_validate(
            manifest["baseline_snapshot"]
        )
        session.current_snapshot = DesignSnapshotRef.model_validate(
            resume_state.get("current_snapshot")
            or manifest.get("current_snapshot")
            or manifest["baseline_snapshot"]
        )
        session.best_snapshot = DesignSnapshotRef.model_validate(
            resume_state.get("best_snapshot")
            or manifest.get("best_snapshot")
            or manifest["baseline_snapshot"]
        )
        session.completed_iteration = int(resume_state.get("iteration", 0))
        session.no_progress = int(resume_state.get("no_progress", 0))
        session.best_total = int(resume_state.get(
            "best_total",
            drc_statistics(Path(session.best_snapshot.drc_ref.path)).total_drv,
        ))
        session.previous_regions = [
            RegionState.model_validate(item)
            for item in resume_state.get("previous_regions", [])
        ]
        boundary = (resume_state.get("_checkpoint") or {}).get("boundary")
        if boundary in {"window_commit", "window_complete"}:
            session.iteration_start_snapshot = DesignSnapshotRef.model_validate(
                resume_state["iteration_start_snapshot"]
            )
            session.iteration_start_total_drv = int(
                resume_state["iteration_start_total_drv"]
            )
            session.iteration_frontier_violation_ids = set(
                resume_state.get("iteration_frontier_violation_ids", [])
            )
            session.processed_frontier_violation_ids = set(
                resume_state.get("processed_frontier_violation_ids", [])
            )
            session.deferred_frontier_violation_ids = set(
                resume_state.get("deferred_frontier_violation_ids", [])
            )
            session.skipped_frontier_violation_ids = set(
                resume_state.get("skipped_frontier_violation_ids", [])
            )
            session.skipped_frontier_by_rule = dict(
                resume_state.get("skipped_frontier_by_rule", {})
            )
            session.window_index = int(resume_state.get("window_index", 0))
            session.window_results = [
                WindowResult.model_validate(item)
                for item in resume_state.get("window_results", [])
            ]
            session.iteration_sandbox_jobs_used = int(
                resume_state.get("iteration_sandbox_jobs_used", 0)
            )
        return session

    def _baseline(self, session: _Session, evidence_level: str) -> dict:
        case_id = session.case_id
        script_ref = session.store.copy(
            session.paths["script"], f"baseline/{case_id}.py",
            producer="initialize_case", media_type="text/x-python",
        )
        gds_ref = session.store.copy(
            session.paths["gds"], f"baseline/{case_id}.gds",
            producer="initialize_case", media_type="application/gds",
        )
        drc_ref = session.store.copy(
            session.paths["drc"], f"baseline/{case_id}.drc.json",
            producer="initialize_case", media_type="application/json",
        )
        connectivity_ref = session.store.copy(
            session.paths["connectivity"], f"baseline/{case_id}.connectivity.json",
            producer="initialize_case", media_type="application/json",
        )
        snapshot = DesignSnapshotRef(
            snapshot_id="snapshot_" + stable_hash([
                session.run_id, "baseline", script_ref.sha256,
            ])[:20],
            script_ref=script_ref, gds_ref=gds_ref, drc_ref=drc_ref,
            connectivity_ref=connectivity_ref,
            provenance_run_id=session.run_id,
            provenance_relation_source="OFFICIAL_BASELINE_ROOT",
            score=(1, 0, -drc_statistics(session.paths["drc"]).total_drv),
        )
        session.baseline_snapshot = snapshot
        session.current_snapshot = snapshot
        session.best_snapshot = snapshot
        stats = drc_statistics(session.paths["drc"])
        session.best_total = stats.total_drv
        case_config = self.project_root / "configs" / "cases" / f"{case_id}.yaml"
        if case_config.is_file():
            expected = yaml.safe_load(case_config.read_text(encoding="utf-8")) or {}
            if expected.get("expected_baseline_drv") is not None:
                if int(expected["expected_baseline_drv"]) != stats.total_drv:
                    raise ValueError(
                        f"baseline DRV mismatch: expected "
                        f"{expected['expected_baseline_drv']}, got {stats.total_drv}"
                    )
            if expected.get("expected_violated_rules") is not None:
                if int(expected["expected_violated_rules"]) != stats.rules_violated:
                    raise ValueError(
                        f"baseline rule count mismatch: expected "
                        f"{expected['expected_violated_rules']}, "
                        f"got {stats.rules_violated}"
                    )
        baseline_metrics = IterationMetrics(
            iteration=0,
            transaction_status=IterationTransactionStatus.BASELINE,
            committed_total_drv=stats.total_drv,
            attempted_total_drv=stats.total_drv,
            connectivity_preserved=True,
            drc_by_rule=stats.drc_by_rule,
            drc_by_marker_type=stats.drc_by_marker_type,
            drc_by_rule_and_type=stats.drc_by_rule_and_type,
            artifact_paths={
                "committed_drc": drc_ref.path,
                "committed_script": script_ref.path,
                "committed_gds": gds_ref.path,
            },
        )
        IterationMetricsStore(
            session.run_dir / "metrics" / "iteration_metrics.jsonl"
        ).append(baseline_metrics)
        session.store.write_json(
            "iterations/iter_0000/metrics/iteration_metrics.json",
            baseline_metrics, producer="initialize_case",
            schema_name="IterationMetrics",
        )
        for ref, suffix, media in [
            (script_ref, ".py", "text/x-python"),
            (gds_ref, ".gds", "application/gds"),
            (drc_ref, ".drc.json", "application/json"),
        ]:
            session.store.copy(
                Path(ref.path), f"iterations/iter_0000/committed/{case_id}{suffix}",
                producer="initialize_case", media_type=media,
            )
        resolved_path = session.run_dir / "resolved_config.yaml"
        resolved_path.write_text(yaml.safe_dump(
            session.config.model_dump(mode="json"), sort_keys=True,
        ), encoding="utf-8")
        config_ref = ArtifactRef.from_path(
            resolved_path, producer="initialize_case",
            media_type="application/yaml", schema_name="AppConfig",
        )
        evaluator = self._adapter or DAC26ReportAdapter(
            self.benchmark_root,
            runtime_roots=[session.config.backend.stage_root],
        )
        evaluator_hash = evaluator.evaluator_hash
        hashes = {
            # The checkpoint identity deliberately excludes only the mutable
            # execution envelope.  resolved_config.yaml remains immutable and
            # retains its full content hash in the manifest.
            "config_hash": scientific_config_hash(session.config),
            "baseline_hash": script_ref.sha256,
            "evaluator_hash": evaluator_hash,
            "code_commit": executable_digest(self.project_root),
        }
        session.hashes = hashes
        return {
            "snapshot": snapshot, "stats": stats, "config_ref": config_ref,
            "evaluator": evaluator, "evidence_level": evidence_level,
        }

    def _run_noop(
        self, *, case_id: str, run_id: str, experiment_id: str,
        seed: int, method: str, preset: MethodPreset, config: AppConfig,
    ) -> RunResult:
        session = self._new_session(
            case_id=case_id, run_id=run_id, experiment_id=experiment_id,
            seed=seed, method=method, preset=preset, config=config,
        )
        base = self._baseline(session, "E0")
        session.manifest = {
            "run_id": run_id, "experiment_id": experiment_id,
            "case_id": case_id, "seed": seed, "method": preset.name,
            "evidence_level": "E0", "status": "COMPLETED_NO_OP",
            "source_commit": _source_revision(self.project_root),
            "executable_digest": session.hashes["code_commit"],
            "benchmark_commit": _git_commit(self.benchmark_root),
            "evaluator_hash": session.hashes["evaluator_hash"],
            "resolved_config_hash": config.content_hash,
            "model": {"provider": "NONE", "name": "NO_OP", "effort": None},
            "actual_llm_call_count": 0,
            "baseline_snapshot": base["snapshot"].model_dump(mode="json"),
            "best_snapshot": base["snapshot"].model_dump(mode="json"),
            "start_time": utc_now().isoformat(),
            "end_time": utc_now().isoformat(),
        }
        session.store.write_json(
            "manifest.json", session.manifest,
            producer="finalize_run", schema_name="RunManifest",
        )
        session.checkpoints.connection.close()
        session.store.write_sha256sums()
        return RunResult(
            run_id=run_id, run_dir=session.run_dir,
            status="COMPLETED_NO_OP", iteration=0,
            region_count=0, subgraph_count=0,
            selected_candidate_ids=[],
        )

    def _finalize_aborted(
        self, session: _Session, *, status: str,
        error_code: str, error_message: str,
        failure: BaseException | None = None,
    ) -> None:
        summary = (
            session.llm_logger.summary()
            if session.llm_logger is not None else {}
        )
        session.final_status = status
        interrupted_nodes = sorted(
            session.active_nodes
            | ({session.interrupted_node} if session.interrupted_node else set())
        )
        safe_checkpoint = session.checkpoints.latest_safe(session.run_id)
        metadata = failure_metadata(failure or RuntimeError(error_message))
        interrupted_stage = (
            session.interrupted_node
            or (interrupted_nodes[0] if interrupted_nodes else "WORKFLOW")
        )
        update = {
            "status": status,
            "end_time": utc_now().isoformat(),
            "active_run_seconds": (
                session.active_run_seconds_before
                + max(0.0, time.monotonic() - session.run_started_monotonic)
            ),
            "iterations_completed": session.completed_iteration,
            "actual_llm_call_count": int(
                summary.get("recorded_calls", 0)
            ),
            "successful_llm_call_count": int(
                summary.get("successful_calls", 0)
            ),
            "failed_llm_call_count": int(
                summary.get("failed_calls", 0)
            ),
            "token_usage": {
                key: int(summary.get(key, 0))
                for key in [
                    "input_tokens", "output_tokens",
                    "reasoning_tokens", "cache_read_tokens",
                ]
            },
            "best_snapshot": (
                session.best_snapshot.model_dump(mode="json")
                if session.best_snapshot is not None else None
            ),
            "current_snapshot": (
                session.current_snapshot.model_dump(mode="json")
                if session.current_snapshot is not None else None
            ),
            "last_safe_checkpoint": (
                {
                    key: value for key, value in safe_checkpoint.items()
                    if key != "state"
                } if safe_checkpoint else None
            ),
        }
        if status == "INTERRUPTED":
            update.update({
                "interrupted_at": utc_now().isoformat(),
                "interrupted_stage": interrupted_stage,
                "interruption": {
                "error_code": error_code,
                "message": error_message[:500],
                "active_nodes": interrupted_nodes,
                "incomplete_requests_may_have_provider_usage": True,
                },
            })
        else:
            update.update(metadata)
            update["failure_message"] = error_message[:500]
        try:
            update["repair_kernel_integration"] = (
                _current_repair_kernel_audit(
                    session, audit_complete=False,
                )
            )
        except Exception:
            # Audit persistence must never hide the original failure.
            pass
        if session.resume_state is not None:
            history = list(session.manifest.get("resume_history") or [])
            if history:
                history[-1].update({
                    "ended_at": utc_now().isoformat(), "status": status,
                })
            update["resume_history"] = history
        session.manifest.update(update)
        try:
            session.store.write_json(
                "manifest.json", session.manifest,
                producer="abort_run", schema_name="RunManifest",
            )
        except Exception:
            pass
        if session.progress is not None:
            event_name = (
                "run_interrupted" if status == "INTERRUPTED"
                else "run_aborted_infrastructure"
                if metadata.get("failure_domain") == "INFRASTRUCTURE"
                else "run_stopped"
            )
            session.progress.emit(
                event_name, "Research runtime stopped before finalize",
                level="WARNING" if status == "INTERRUPTED" else "ERROR",
                run_id=session.run_id, status=status,
                active_nodes=interrupted_nodes,
                recorded_llm_calls=summary.get("recorded_calls", 0),
                failure_domain=metadata.get("failure_domain"),
                failure_code=metadata.get("failure_code"),
                interrupted_stage=interrupted_stage,
            )
        if session.sandbox_service is not None:
            try:
                session.sandbox_service.close()
            except Exception:
                pass
        if isinstance(session.experience_store, ExperienceStore):
            try:
                session.experience_store.close()
            except Exception:
                pass
        try:
            session.checkpoints.save(
                session.run_id, status.lower(), session.hashes,
                {
                    "iteration": session.completed_iteration,
                    "status": status,
                    "active_nodes": interrupted_nodes,
                    "current_snapshot": (
                        session.current_snapshot.model_dump(mode="json")
                        if session.current_snapshot else None
                    ),
                    "best_snapshot": (
                        session.best_snapshot.model_dump(mode="json")
                        if session.best_snapshot else None
                    ),
                    **metadata,
                },
            )
        except Exception:
            pass
        try:
            session.checkpoints.connection.close()
        except Exception:
            pass
        try:
            session.store.write_sha256sums()
        except Exception:
            pass

    def _execute_session(self, session: _Session) -> RunResult:
        self.session = session
        workflow = build_workflow(self._nodes(session))
        is_resume = session.resume_state is not None
        session.progress.emit(
            "resume_started" if is_resume else "run_started",
            (
                f"Resuming {session.preset.name} Research Runtime"
                if is_resume else
                f"{session.preset.name} Research Runtime started"
            ),
            run_id=session.run_id, case_id=session.case_id,
            method=session.method,
            iteration=session.completed_iteration,
            max_iterations=session.config.workflow.max_iterations,
        )
        previous_sigint = signal.getsignal(signal.SIGINT)
        previous_sigterm = signal.getsignal(signal.SIGTERM)
        active_task: dict[str, asyncio.Task[Any] | None] = {"task": None}
        received_signal: dict[str, str | None] = {"name": None}

        def terminate_handler(signum, frame):
            session.interrupted_node = next(
                iter(sorted(session.active_nodes)), "WORKFLOW"
            )
            received_signal["name"] = f"SIGNAL_{signal.Signals(signum).name}"
            task = active_task["task"]
            if task is None:
                raise KeyboardInterrupt(received_signal["name"])
            task.cancel()

        async def invoke_workflow() -> None:
            active_task["task"] = asyncio.current_task()
            try:
                await workflow.ainvoke(
                    {
                        "run_id": session.run_id,
                        "experiment_id": session.experiment_id,
                        "case_id": session.case_id,
                        "backend_type": session.config.backend.type,
                        "iteration": session.completed_iteration,
                        "subgraph_results": [],
                        "verification_results": [], "errors": [],
                    },
                    config={
                        "recursion_limit": workflow_recursion_limit(
                            max_iterations=(
                                session.config.workflow.max_iterations
                            ),
                            max_windows_per_iteration=(
                                session.config.workflow.max_windows_per_iteration
                            ),
                        ),
                        "max_concurrency": (
                            session.config.workflow.max_parallel_subgraphs
                        ),
                    },
                )
            finally:
                active_task["task"] = None

        signal.signal(signal.SIGINT, terminate_handler)
        signal.signal(signal.SIGTERM, terminate_handler)
        try:
            asyncio.run(invoke_workflow())
            session.store.write_sha256sums()
        except asyncio.CancelledError as exc:
            signal_name = received_signal["name"] or "ASYNCIO_CANCELLED"
            self._finalize_aborted(
                session, status="INTERRUPTED",
                error_code=signal_name,
                error_message=f"runtime interrupted by {signal_name}",
            )
            raise KeyboardInterrupt(signal_name) from exc
        except KeyboardInterrupt as exc:
            signal_name = (
                str(exc) if str(exc).startswith("SIGNAL_") else None
            )
            self._finalize_aborted(
                session, status="INTERRUPTED",
                error_code=signal_name or "KEYBOARD_INTERRUPT",
                error_message=(
                    f"runtime interrupted by {signal_name}"
                    if signal_name else "user interrupted the runtime"
                ),
            )
            raise
        except IntegrityFailure as exc:
            self._finalize_aborted(
                session, status="BLOCKED_INTEGRITY",
                error_code=exc.failure_code,
                error_message=str(exc), failure=exc,
            )
            raise
        except InfrastructureFailure as exc:
            budget_stop = exc.failure_code == FailureCode.BUDGET_EXHAUSTED
            self._finalize_aborted(
                session, status="BUDGET_EXHAUSTED" if budget_stop else "FAILED",
                error_code=exc.failure_code.value,
                error_message=str(exc), failure=exc,
            )
            if not budget_stop:
                raise
        except Exception as exc:
            self._finalize_aborted(
                session, status="FAILED",
                error_code=type(exc).__name__,
                error_message=str(exc), failure=exc,
            )
            raise
        finally:
            signal.signal(signal.SIGINT, previous_sigint)
            signal.signal(signal.SIGTERM, previous_sigterm)
        return RunResult(
            run_id=session.run_id, run_dir=session.run_dir,
            status=session.final_status,
            iteration=session.completed_iteration,
            region_count=len(session.regions),
            subgraph_count=len(session.subgraphs),
            selected_candidate_ids=sorted(session.selected_candidate_ids),
        )

    def run(
        self, case_id: str, run_id: str, *, experiment_id: str = "development",
        seed: int = 0, method: str = "B6",
    ) -> RunResult:
        preset, config = resolve_method(
            method, self.base_cfg, test_mode=self.test_mode,
        )
        validate_method_identity(
            preset.name, config, run_id=run_id,
            experiment_id=experiment_id,
        )
        if (
            preset.runtime == "research"
            and config.repair_kernel_integration.enabled
            and config.repair_kernel_integration.require_gate
        ):
            verify_formal_integration_gate(
                self.project_root,
                config=config,
                contract_path=(
                    self.project_root
                    / config.repair_kernel_integration.gate_contract
                    if not config.repair_kernel_integration.gate_contract.is_absolute()
                    else config.repair_kernel_integration.gate_contract
                ),
                run_id=run_id, case_id=case_id, method=preset.name,
                experiment_id=experiment_id,
            )
        if preset.runtime == "noop":
            return self._run_noop(
                case_id=case_id, run_id=run_id,
                experiment_id=experiment_id, seed=seed,
                method=method, preset=preset, config=config,
            )
        session = self._new_session(
            case_id=case_id, run_id=run_id, experiment_id=experiment_id,
            seed=seed, method=method, preset=preset, config=config,
        )
        # The persistent flock covers the entire writer lifetime.  If the
        # process is SIGKILLed the kernel releases it, allowing a later resume
        # to distinguish a stale RUNNING manifest from an active owner.
        with continuation_lock(session.run_dir):
            return self._execute_session(session)

    def resume(
        self, run_id: str, *, extend_max_iterations: int | None = None,
        extend_max_http_attempts_per_run: int | None = None,
        extend_max_eda_evaluations_per_run: int | None = None,
        extend_whole_run_budget_seconds: float | None = None,
        continuation_nonce_value: str | None = None,
        requester_authorization: str = "EXPLICIT_USER_CLI",
        verify_only: bool = False,
    ) -> RunResult | ContinuationVerification:
        run_dir = self.project_root / "runs" / run_id
        self._validate_run_id(run_id)
        manifest = json.loads(
            (run_dir / "manifest.json").read_text(encoding="utf-8")
        )
        status = str(manifest.get("status", ""))
        continuation_metadata = manifest.get("continuation") or {}
        strict_continuation = (
            continuation_metadata.get("protocol")
            == "p5-iteration-extension-v1"
        )
        if not strict_continuation:
            if (
                extend_max_iterations is not None
                or extend_max_http_attempts_per_run is not None
                or extend_max_eda_evaluations_per_run is not None
                or extend_whole_run_budget_seconds is not None
                or verify_only
            ):
                raise ValueError(
                    "P5_CONTINUATION_METADATA_REQUIRED: old runs cannot be "
                    "silently upgraded"
                )
            if status.startswith("COMPLETED"):
                raise ValueError("completed runs cannot be resumed")
            if status == "FAILED" and manifest.get("failure_domain") != (
                "INFRASTRUCTURE"
            ):
                raise ValueError(
                    "only infrastructure-failed or interrupted runs can resume"
                )
            if status not in {"FAILED", "INTERRUPTED"}:
                raise ValueError(f"run status is not resumable: {status}")
            return self._resume_legacy(run_id, manifest)

        with continuation_lock(run_dir):
            # Re-read under the owner lock so a concurrent resume cannot race
            # an extension decision made from stale status.
            manifest = json.loads(
                (run_dir / "manifest.json").read_text(encoding="utf-8")
            )
            status = str(manifest.get("status", ""))
            return self._resume_p5(
                run_id=run_id,
                run_dir=run_dir,
                manifest=manifest,
                status=status,
                extend_max_iterations=extend_max_iterations,
                extend_max_http_attempts_per_run=(
                    extend_max_http_attempts_per_run
                ),
                extend_max_eda_evaluations_per_run=(
                    extend_max_eda_evaluations_per_run
                ),
                extend_whole_run_budget_seconds=(
                    extend_whole_run_budget_seconds
                ),
                continuation_nonce_value=continuation_nonce_value,
                requester_authorization=requester_authorization,
                verify_only=verify_only,
            )

    def _verify_resume_gate(
        self, *, preset: MethodPreset, config: AppConfig,
        manifest: dict, run_id: str,
    ) -> None:
        if (
            preset.runtime == "research"
            and config.repair_kernel_integration.enabled
            and config.repair_kernel_integration.require_gate
        ):
            verify_formal_integration_gate(
                self.project_root,
                config=config,
                contract_path=(
                    self.project_root
                    / config.repair_kernel_integration.gate_contract
                    if not config.repair_kernel_integration.gate_contract.is_absolute()
                    else config.repair_kernel_integration.gate_contract
                ),
                run_id=run_id, case_id=str(manifest["case_id"]),
                method=preset.name,
                experiment_id=str(manifest["experiment_id"]),
            )

    def _resume_image_digest(self, config: AppConfig) -> str:
        if self._backend is not None:
            return (
                self._backend.image_digest()
                if hasattr(self._backend, "image_digest")
                else "TEST_BACKEND"
            )
        try:
            return subprocess.check_output(
                [
                    "docker", "image", "inspect", "--format={{.Id}}",
                    config.backend.image,
                ],
                text=True, stderr=subprocess.STDOUT,
            ).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError("RESUME_DOCKER_IMAGE_INSPECTION_FAILED") from exc

    def _resume_legacy(self, run_id: str, manifest: dict) -> RunResult:
        """Preserve pre-P5 interrupted-run behavior; never extend it."""

        run_dir = self.project_root / "runs" / run_id
        method = str(manifest["method"])
        preset, config = resolve_method(
            method, self.base_cfg, test_mode=self.test_mode,
        )
        validate_method_identity(
            preset.name, config, run_id=run_id,
            experiment_id=str(manifest["experiment_id"]),
        )
        self._verify_resume_gate(
            preset=preset, config=config, manifest=manifest, run_id=run_id,
        )
        hashes = {
            "config_hash": config.content_hash,
            "baseline_hash": manifest["baseline_snapshot"][
                "script_ref"
            ]["sha256"],
            "evaluator_hash": manifest["evaluator_hash"],
            "code_commit": _source_revision(self.project_root),
        }
        checkpoint_store = CheckpointStore(
            run_dir / "checkpoints" / "workflow.sqlite",
            readonly=True,
        )
        try:
            resume_state = checkpoint_store.resume(run_id, hashes)
        finally:
            checkpoint_store.connection.close()
        history = list(manifest.get("resume_history") or [])
        prior_failure = {
            key: manifest.get(key) for key in [
                "failure_domain", "failure_code", "failure_stage",
                "retryable", "failure_message",
            ] if key in manifest
        }
        history.append({
            "session": len(history) + 1,
            "started_at": utc_now().isoformat(),
            "from_checkpoint": resume_state["_checkpoint"],
            "prior_status": status,
            "prior_failure": prior_failure or None,
            "status": "RUNNING",
        })
        for key in [
            "failure_domain", "failure_code", "failure_stage", "retryable",
            "failure_message", "http_status_code", "retry_after_seconds",
            "original_exception_type", "interrupted_at", "interrupted_stage",
            "interruption",
        ]:
            manifest.pop(key, None)
        manifest["resume_history"] = history
        manifest["status"] = "RUNNING"
        session = self._resume_session(
            case_id=str(manifest["case_id"]), run_id=run_id,
            experiment_id=str(manifest["experiment_id"]),
            seed=int(manifest["seed"]), method=method,
            preset=preset, config=config, manifest=manifest,
            resume_state=resume_state,
        )
        session.hashes = hashes
        return self._execute_session(session)

    def _resume_p5(
        self, *, run_id: str, run_dir: Path, manifest: dict,
        status: str, extend_max_iterations: int | None,
        extend_max_http_attempts_per_run: int | None,
        extend_max_eda_evaluations_per_run: int | None,
        extend_whole_run_budget_seconds: float | None,
        continuation_nonce_value: str | None,
        requester_authorization: str, verify_only: bool,
    ) -> RunResult | ContinuationVerification:
        from drc_agent.config.loader import AppConfig

        resolved_path = run_dir / "resolved_config.yaml"
        resolved_payload = yaml.safe_load(
            resolved_path.read_text(encoding="utf-8")
        )
        stored_base = AppConfig.model_validate(resolved_payload)
        method = str(manifest["method"])
        preset, original_config = resolve_method(
            method, stored_base, test_mode=self.test_mode,
        )
        _, supplied_config = resolve_method(
            method, self.base_cfg, test_mode=self.test_mode,
        )
        validate_method_identity(
            preset.name, original_config, run_id=run_id,
            experiment_id=str(manifest["experiment_id"]),
        )
        if supplied_config.content_hash != original_config.content_hash:
            raise ValueError("RESUME_SUPPLIED_CONFIG_MISMATCH")

        scientific_hash = scientific_config_hash(original_config)
        executable_identity = executable_digest(self.project_root)
        initial_envelope = execution_envelope(original_config)
        metadata = manifest.get("continuation") or {}
        immutable_run_identity = run_identity(
            run_id=run_id,
            experiment_id=str(manifest["experiment_id"]),
            case_id=str(manifest["case_id"]),
            method=method,
            seed=int(manifest["seed"]),
        )
        if (
            manifest.get("run_id") != run_id
            or manifest.get("resolved_config_hash")
            != original_config.content_hash
            or metadata.get("initial_resolved_config_hash")
            != original_config.content_hash
            or manifest.get("scientific_config_hash") != scientific_hash
            or metadata.get("scientific_config_hash") != scientific_hash
            or manifest.get("executable_digest") != executable_identity
            or metadata.get("initial_execution_envelope")
            != initial_envelope.model_dump(mode="json")
            or metadata.get("initial_execution_envelope_hash")
            != initial_envelope.content_hash
            or stable_hash(original_config.llm.model_dump(mode="json"))
            != manifest.get("llm_config_hash")
            or metadata.get("run_identity") != immutable_run_identity
            or metadata.get("run_identity_hash")
            != stable_hash(immutable_run_identity)
        ):
            raise ValueError("RESUME_SCIENTIFIC_IDENTITY_MISMATCH")
        if (
            official_input_identity(self._case_paths(str(manifest["case_id"])))
            != manifest.get("official_input_identity")
        ):
            raise ValueError("RESUME_OFFICIAL_INPUT_IDENTITY_MISMATCH")

        evaluator = self._adapter or DAC26ReportAdapter(
            self.benchmark_root,
            runtime_roots=[original_config.backend.stage_root],
        )
        if evaluator.evaluator_hash != manifest.get("evaluator_hash"):
            raise ValueError("RESUME_EVALUATOR_IDENTITY_MISMATCH")
        if self._resume_image_digest(original_config) != manifest.get(
            "docker_image_digest"
        ):
            raise ValueError("RESUME_DOCKER_IMAGE_DIGEST_MISMATCH")
        hashes = {
            "config_hash": scientific_hash,
            "baseline_hash": manifest["baseline_snapshot"][
                "script_ref"
            ]["sha256"],
            "evaluator_hash": evaluator.evaluator_hash,
            "code_commit": executable_identity,
        }
        checkpoint_store = CheckpointStore(
            run_dir / "checkpoints" / "workflow.sqlite"
        )
        try:
            resume_state = checkpoint_store.resume(run_id, hashes)
        finally:
            checkpoint_store.connection.close()

        completed = int(resume_state.get("iteration", -1))
        manifest_completed = manifest.get("iterations_completed")
        # A process can disappear while the initial iteration is still in
        # progress, before finalization has ever published this summary
        # field. The hash-verified safe checkpoint is authoritative for a
        # RUNNING/INTERRUPTED recovery; finalized and FAILED manifests must
        # continue to carry an exact explicit count.
        if manifest_completed is None and status in {"RUNNING", "INTERRUPTED"}:
            manifest_completed = completed
        if completed != int(
            manifest_completed if manifest_completed is not None else -1
        ):
            raise ValueError("RESUME_ITERATION_ACCOUNTING_MISMATCH")
        if resume_state.get("scientific_config_hash") != scientific_hash:
            raise ValueError("CHECKPOINT_SCIENTIFIC_CONFIG_MISMATCH")
        if (
            resume_state.get("official_input_identity")
            != manifest.get("official_input_identity")
        ):
            raise ValueError("CHECKPOINT_INPUT_IDENTITY_MISMATCH")
        if (
            resume_state.get("run_identity") != immutable_run_identity
            or resume_state.get("run_identity_hash")
            != stable_hash(immutable_run_identity)
        ):
            raise ValueError("CHECKPOINT_RUN_IDENTITY_MISMATCH")
        baseline = DesignSnapshotRef.model_validate(
            resume_state.get("baseline_snapshot")
        )
        current = DesignSnapshotRef.model_validate(
            resume_state.get("current_snapshot")
        )
        best = DesignSnapshotRef.model_validate(
            resume_state.get("best_snapshot")
        )
        if (
            baseline.model_dump(mode="json") != manifest["baseline_snapshot"]
            or current.model_dump(mode="json")
            != manifest.get("current_snapshot")
            or best.model_dump(mode="json") != manifest.get("best_snapshot")
        ):
            raise ValueError("RESUME_SNAPSHOT_CHECKPOINT_MISMATCH")
        for snapshot in (baseline, current, best):
            verify_snapshot(snapshot, run_dir, run_id=run_id)
        if original_config.repair_kernel_integration.enabled:
            # Formal B6 continuation must prove that the immutable artifacts
            # are still semantically reconstructible before any authorization
            # is appended or provider/backend preflight can occur.  Non-formal
            # profiles do not satisfy (and did not promise) this adapter's
            # frozen-rule-deck contract; their snapshot/lineage artifact chain
            # has already been verified above.
            semantic_adapter = FormalCurrentSemanticAdapter(
                self.project_root,
                original_config,
                require_frozen_rule_deck=not self.test_mode,
            )
            semantic_snapshot_ids = set()
            for snapshot in (current, best):
                if snapshot.snapshot_id in semantic_snapshot_ids:
                    continue
                semantic_adapter.build_snapshot_context(
                    current=snapshot,
                    baseline=baseline,
                    case_id=str(manifest["case_id"]),
                    rule_deck_path=self._case_paths(
                        str(manifest["case_id"])
                    )["rule_deck"],
                    run_id=run_id,
                )
                semantic_snapshot_ids.add(snapshot.snapshot_id)

        records = read_continuation_records(
            run_dir,
            run_id=run_id,
            initial_envelope=initial_envelope,
            scientific_hash=scientific_hash,
            initial_resolved_hash=original_config.content_hash,
            expected_executable_digest=executable_identity,
        )
        current_envelope = authorized_envelope(initial_envelope, records)
        claimed_segment = int(metadata.get("segment_index", -1))
        if claimed_segment > len(records) or claimed_segment < 0:
            raise ValueError("CONTINUATION_MANIFEST_INDEX_MISMATCH")
        if claimed_segment == len(records):
            expected_record_hash = records[-1].record_hash if records else None
            if (
                metadata.get("latest_record_hash") != expected_record_hash
                or metadata.get("authorized_execution_envelope")
                != current_envelope.model_dump(mode="json")
                or metadata.get("authorized_execution_envelope_hash")
                != current_envelope.content_hash
                or manifest.get("execution_envelope")
                != current_envelope.model_dump(mode="json")
            ):
                raise ValueError("CONTINUATION_MANIFEST_HEAD_MISMATCH")
        elif (
            claimed_segment != len(records) - 1
            or status != "COMPLETED_MAX_ITERATIONS"
        ):
            # A journal may be exactly one publication ahead only if the
            # authorization append completed before resume initialization.
            raise ValueError("CONTINUATION_MANIFEST_UNEXPECTEDLY_STALE")
        allowed_envelope_hashes = {
            initial_envelope.content_hash,
            *(record.new_envelope_hash for record in records),
        }
        if resume_state.get("execution_envelope_hash") not in (
            allowed_envelope_hashes
        ):
            raise ValueError("CHECKPOINT_EXECUTION_ENVELOPE_MISMATCH")
        cutoff = verify_experience_cutoff(
            run_dir,
            resume_state.get("experience_knowledge_cutoff"),
            enabled=original_config.features.experience_graph,
            checkpoint_boundary=(
                resume_state.get("_checkpoint") or {}
            ).get("boundary"),
            checkpoint_state=resume_state,
        )
        verify_repair_attempt_memory(resume_state)
        verify_frontier_state(resume_state)
        checkpoint_hashes = continuation_state_hashes(resume_state)
        usage = cumulative_usage(
            run_dir,
            case_id=str(manifest["case_id"]),
            manifest=manifest,
            config=original_config,
            envelope=current_envelope,
        )
        identity = {
            **immutable_run_identity,
            "baseline_script_sha256": hashes["baseline_hash"],
            "official_input_identity_hash": stable_hash(
                manifest["official_input_identity"]
            ),
            "evaluator_hash": hashes["evaluator_hash"],
            "docker_image": manifest.get("docker_image"),
            "docker_image_digest": manifest.get("docker_image_digest"),
            "model_hash": manifest.get("llm_config_hash"),
        }
        if any(record.identity != identity for record in records):
            raise ValueError("CONTINUATION_RECORD_IDENTITY_MISMATCH")

        requested_nonce = None
        existing_record = None
        proposed_envelope = None
        envelope_requested = any(value is not None for value in (
            extend_max_iterations,
            extend_max_http_attempts_per_run,
            extend_max_eda_evaluations_per_run,
            extend_whole_run_budget_seconds,
        ))
        budget_amendment_requested = any(value is not None for value in (
            extend_max_http_attempts_per_run,
            extend_max_eda_evaluations_per_run,
            extend_whole_run_budget_seconds,
        ))
        if envelope_requested:
            target_iterations = (
                extend_max_iterations
                if extend_max_iterations is not None
                else current_envelope.max_iterations
            )
            if not 1 <= target_iterations <= MAX_CONTINUATION_ITERATIONS:
                raise ValueError(
                    "extend-max-iterations must be between 1 and "
                    f"{MAX_CONTINUATION_ITERATIONS}"
                )
            target_payload = current_envelope.model_dump(mode="json")
            requested_limits = {
                "max_http_attempts_per_run": (
                    extend_max_http_attempts_per_run
                ),
                "max_eda_evaluations_per_run": (
                    extend_max_eda_evaluations_per_run
                ),
                "whole_run_budget_seconds": (
                    extend_whole_run_budget_seconds
                ),
            }
            target_payload["max_iterations"] = target_iterations
            for field, value in requested_limits.items():
                if value is not None:
                    target_payload[field] = value
            requested_envelope = ExecutionEnvelope.model_validate(
                target_payload
            )
            requested_nonce = continuation_nonce(
                continuation_nonce_value, target_iterations,
                envelope=(
                    requested_envelope
                    if budget_amendment_requested else None
                ),
            )
            existing_record = next(
                (item for item in records if item.nonce == requested_nonce),
                None,
            )
            if existing_record is not None:
                if (
                    existing_record.new_envelope != requested_envelope
                    or existing_record.identity != identity
                ):
                    raise ValueError("CONTINUATION_NONCE_CONFLICT")
                proposed_envelope = existing_record.new_envelope
            else:
                for field in (
                    "max_http_attempts_per_run",
                    "max_eda_evaluations_per_run",
                    "whole_run_budget_seconds",
                ):
                    old = getattr(current_envelope, field)
                    new = getattr(requested_envelope, field)
                    if new != old and (
                        old is None or new is None or new <= old
                    ):
                        raise ValueError(
                            "budget amendments must strictly increase an "
                            "existing finite cumulative ceiling"
                        )
                iteration_increased = (
                    requested_envelope.max_iterations
                    > current_envelope.max_iterations
                )
                if requested_envelope.max_iterations < (
                    current_envelope.max_iterations
                ):
                    raise ValueError("continuation iteration limit cannot decrease")
                if status == "COMPLETED_MAX_ITERATIONS":
                    if not iteration_increased:
                        raise ValueError(
                            "completed-max continuation must increase the "
                            "iteration ceiling"
                        )
                    if completed != current_envelope.max_iterations:
                        raise ValueError(
                            "CONTINUATION_COMPLETION_BOUNDARY_MISMATCH"
                        )
                elif status == "BUDGET_EXHAUSTED":
                    if iteration_increased or not budget_amendment_requested:
                        raise ValueError(
                            "budget-exhausted resume requires only an explicit "
                            "bounded budget increase"
                        )
                    if completed >= current_envelope.max_iterations:
                        raise ValueError(
                            "budget-exhausted resume has no remaining iteration "
                            "under the current iteration ceiling"
                        )
                else:
                    raise ValueError(
                        "new envelope authorization requires "
                        "COMPLETED_MAX_ITERATIONS or BUDGET_EXHAUSTED"
                    )
                if requested_envelope == current_envelope:
                    raise ValueError(
                        "continuation target is already authorized; reuse its "
                        "original nonce for an idempotent request"
                    )
                proposed_envelope = requested_envelope

        effective_envelope = proposed_envelope or current_envelope
        pending_authorization = (
            status == "COMPLETED_MAX_ITERATIONS"
            and completed < effective_envelope.max_iterations
        )
        resumable_interruption = (
            status in {"INTERRUPTED", "RUNNING"}
            or status == "FAILED"
            and manifest.get("failure_domain") == "INFRASTRUCTURE"
        )
        already_applied = bool(
            existing_record is not None
            and not pending_authorization
        )
        if status == "COMPLETED_DRC_CLEAN" and not already_applied:
            raise ValueError(
                "completed runs cannot be resumed: naturally DRC-clean runs "
                "do not need continuation"
            )
        budget_amendment_pending = bool(
            status == "BUDGET_EXHAUSTED"
            and proposed_envelope is not None
            and existing_record is None
            and proposed_envelope != current_envelope
        )
        if status == "BUDGET_EXHAUSTED" and not (
            budget_amendment_pending or already_applied
        ):
            raise ValueError("budget exhaustion requires an explicit amendment")
        if not (
            pending_authorization
            or budget_amendment_pending
            or resumable_interruption
            or already_applied
            or (
                status == "COMPLETED_MAX_ITERATIONS"
                and envelope_requested
                and proposed_envelope is not None
                and proposed_envelope.max_iterations > completed
            )
        ):
            if status.startswith("COMPLETED"):
                raise ValueError("completed runs cannot be resumed")
            raise ValueError(f"run status is not resumable: {status}")

        if verify_only:
            return continuation_verification_result(
                run_id=run_id,
                prior_status=status,
                completed_iterations=completed,
                checkpoint=resume_state,
                scientific_hash=scientific_hash,
                executable_identity=executable_identity,
                current_envelope=current_envelope,
                proposed_envelope=proposed_envelope,
                nonce=requested_nonce,
                records=records,
                usage=usage,
                cutoff=cutoff,
                hashes=checkpoint_hashes,
                status="VERIFIED_RESUMABLE",
                verify_only=True,
            )
        if already_applied:
            return continuation_verification_result(
                run_id=run_id,
                prior_status=status,
                completed_iterations=completed,
                checkpoint=resume_state,
                scientific_hash=scientific_hash,
                executable_identity=executable_identity,
                current_envelope=current_envelope,
                proposed_envelope=proposed_envelope,
                nonce=requested_nonce,
                records=records,
                usage=usage,
                cutoff=cutoff,
                hashes=checkpoint_hashes,
                status="CONTINUATION_ALREADY_APPLIED",
                verify_only=False,
            )

        # Validate the immutable, originally qualified profile.  The explicit
        # record below, not a modified gate/config file, authorizes more rounds.
        self._verify_resume_gate(
            preset=preset, config=original_config,
            manifest=manifest, run_id=run_id,
        )
        execution_epoch = None
        if envelope_requested and existing_record is None:
            record = make_continuation_record(
                run_id=run_id,
                nonce=requested_nonce,
                requester_authorization=requester_authorization,
                prior_status=status,
                completed_iterations=completed,
                scientific_hash=scientific_hash,
                initial_resolved_hash=original_config.content_hash,
                executable_identity=executable_identity,
                identity=identity,
                current_envelope=current_envelope,
                target_envelope=proposed_envelope,
                checkpoint=resume_state["_checkpoint"],
                cumulative_usage=usage,
                state_hashes=checkpoint_hashes,
                previous=records[-1] if records else None,
            )
            journal_ref = append_continuation_record(run_dir, record)
            records.append(record)
            current_envelope = record.new_envelope
            execution_epoch = record.new_execution_epoch
            continuation_metadata = dict(manifest["continuation"])
            continuation_metadata.update({
                "segment_index": record.segment_index,
                "latest_record_hash": record.record_hash,
                "authorized_execution_envelope": (
                    current_envelope.model_dump(mode="json")
                ),
                "authorized_execution_envelope_hash": (
                    current_envelope.content_hash
                ),
                "journal_ref": journal_ref.model_dump(mode="json"),
            })
            manifest["continuation"] = continuation_metadata
            manifest["execution_envelope"] = current_envelope.model_dump(
                mode="json"
            )
        elif existing_record is not None:
            current_envelope = authorized_envelope(
                initial_envelope, records
            )
            if existing_record.new_execution_epoch not in set(
                manifest.get("execution_epochs") or []
            ):
                # The authorization was durably appended but execution never
                # reached initialize_case; reuse its reserved epoch once.
                execution_epoch = existing_record.new_execution_epoch

        latest_record = records[-1] if records else None
        continuation_metadata = dict(manifest["continuation"])
        continuation_metadata.update({
            "segment_index": len(records),
            "latest_record_hash": (
                latest_record.record_hash if latest_record else None
            ),
            "authorized_execution_envelope": (
                current_envelope.model_dump(mode="json")
            ),
            "authorized_execution_envelope_hash": (
                current_envelope.content_hash
            ),
        })
        if latest_record is not None:
            continuation_metadata["journal_ref"] = (
                ArtifactRef.from_path(
                    run_dir / "continuation" / "records.jsonl",
                    producer="resume_iteration_extension",
                    media_type="application/x-ndjson",
                    schema_name="ContinuationRecord[]",
                    schema_version="1.0",
                ).model_dump(mode="json")
            )
        manifest["continuation"] = continuation_metadata
        manifest["execution_envelope"] = current_envelope.model_dump(
            mode="json"
        )
        manifest["max_iterations"] = current_envelope.max_iterations

        effective_config = apply_execution_envelope(
            original_config, current_envelope,
        )
        validate_method_identity(
            preset.name, effective_config, run_id=run_id,
            experiment_id=str(manifest["experiment_id"]),
        )
        history = list(manifest.get("resume_history") or [])
        prior_failure = {
            key: manifest.get(key) for key in [
                "failure_domain", "failure_code", "failure_stage",
                "retryable", "failure_message",
            ] if key in manifest
        }
        history.append({
            "session": len(history) + 1,
            "started_at": utc_now().isoformat(),
            "from_checkpoint": resume_state["_checkpoint"],
            "prior_status": status,
            "prior_failure": prior_failure or None,
            "continuation_segment_index": len(records),
            "authorized_max_iterations": current_envelope.max_iterations,
            "status": "RUNNING",
        })
        for key in [
            "failure_domain", "failure_code", "failure_stage", "retryable",
            "failure_message", "http_status_code", "retry_after_seconds",
            "original_exception_type", "interrupted_at", "interrupted_stage",
            "interruption",
        ]:
            manifest.pop(key, None)
        manifest["resume_history"] = history
        manifest["status"] = "RUNNING"
        session = self._resume_session(
            case_id=str(manifest["case_id"]), run_id=run_id,
            experiment_id=str(manifest["experiment_id"]),
            seed=int(manifest["seed"]), method=method,
            preset=preset, config=effective_config, manifest=manifest,
            resume_state=resume_state, execution_epoch=execution_epoch,
        )
        session.hashes = hashes
        return self._execute_session(session)

    def _nodes(self, session: _Session) -> dict:
        async def initialize_case(state):
            resumed_manifest = session.manifest
            initial_evidence_level = (
                "TEST_FAKE_LLM"
                if isinstance(self._llm_client, FakeLLMClient) else "E2"
            )
            if session.resume_state is None:
                base = self._baseline(session, initial_evidence_level)
            else:
                resolved_path = session.run_dir / "resolved_config.yaml"
                base = {
                    "snapshot": session.baseline_snapshot,
                    "config_ref": ArtifactRef.from_path(
                        resolved_path, producer="resume_run",
                        media_type="application/yaml",
                        schema_name="AppConfig",
                    ),
                    "evaluator": self._adapter or DAC26ReportAdapter(
                        self.benchmark_root,
                        runtime_roots=[session.config.backend.stage_root],
                    ),
                    "evidence_level": initial_evidence_level,
                }
            llm_log = session.run_dir / "logs" / "llm_calls.jsonl"
            session.llm_log = llm_log
            logger = LLMAuditLogger(
                llm_log, progress=session.progress,
            )
            session.llm_logger = logger
            if self._llm_client is None:
                llm = OpenAICompatibleClient(session.config.llm, logger)
            else:
                llm = self._llm_client
                if isinstance(llm, OpenAICompatibleClient):
                    llm.logger = logger
                if isinstance(llm, FakeLLMClient) and llm.logger is None:
                    llm.logger = logger
                if isinstance(llm, FakeLLMClient) and not self.test_mode:
                    raise ValueError("FakeLLM requires explicit test_mode")
            backend = self._backend or KLayoutBackend(
                self.project_root, image=session.config.backend.image,
                stage_root=session.config.backend.stage_root,
                artifact_retention_mode=(
                    session.config.backend.artifact_retention_mode
                ),
                container_workspace_mode=(
                    session.config.backend.container_workspace_mode
                ),
                max_physical_attempts_per_run=(
                    session.config.workflow.max_eda_evaluations_per_run
                ),
            )
            image_digest = (
                backend.image_digest() if hasattr(backend, "image_digest")
                else "TEST_BACKEND"
            )
            if (
                session.resume_state is not None
                and image_digest
                != session.manifest.get("docker_image_digest")
            ):
                raise ValueError("RESUME_DOCKER_IMAGE_DIGEST_MISMATCH")
            if session.config.features.experience_graph:
                base_experience_dir = (
                    self.project_root / "knowledge" /
                    "base_experience_graph" / "v1"
                )
                if session.resume_state is None:
                    experience = ExperienceStore.branch_from_base(
                        base_experience_dir, session.run_dir, session.run_id,
                    )
                else:
                    experience = ExperienceStore(
                        session.run_dir / "experience" / "experience.sqlite",
                        base_db=base_experience_dir / "experience.sqlite",
                        base_hash=session.manifest.get("base_experience_hash"),
                        graph_version="v1",
                    )
                base_experience_hash = experience.base_hash
            else:
                experience = NullExperienceStore()
                base_experience_hash = "DISABLED"
            session.experience_store = experience
            session.transaction_executor = TransactionExecutor(
                backend=backend, adapter=base["evaluator"],
                store=session.store,
                acceptance=session.config.acceptance,
                backend_config=session.config.backend,
                benchmark_paths=session.paths,
            )
            session.sandbox_service = RuntimeSandboxService(
                config=session.config,
                backend=backend,
                adapter=base["evaluator"],
                paths=session.paths,
                run_id=session.run_id,
                case_id=session.case_id,
                run_store=session.store,
                experience_store=experience,
                baseline_snapshot=lambda: session.baseline_snapshot,
                current_snapshot=lambda: session.current_snapshot,
                current_violations=lambda: session.violations,
                rule_deck_hash=file_sha256(session.paths["rule_deck"]),
                evaluator_hash=session.hashes["evaluator_hash"],
                execution_epoch=session.execution_epoch,
            )
            session.planner = SubgraphPlanner(
                config=session.config, llm=llm,
                experience_store=experience,
                artifact_store=session.store, run_id=session.run_id,
                coordinator_mode=session.preset.coordinator,
                candidate_sandbox=session.sandbox_service.candidate,
                bundle_sandbox=session.sandbox_service.bundle,
                project_root=self.project_root,
                event_sink=(
                    lambda event, **details: session.progress.emit(
                        event, "Formal repair-kernel event",
                        run_id=session.run_id, **details,
                    )
                ),
            )
            if session.resume_state is not None:
                memory = (
                    session.resume_state.get("repair_attempt_memory") or {}
                )
                session.planner.repair_attempt_memory.load_artifact(memory)
            evidence_level = (
                "TEST_FAKE_LLM"
                if isinstance(llm, FakeLLMClient) else "E2"
            )
            session.manifest = {
                "run_id": session.run_id,
                "experiment_id": session.experiment_id,
                "case_id": session.case_id, "seed": session.seed,
                "method": session.preset.name,
                "experience_graph": (
                    session.config.features.experience_graph
                ),
                "method_identity": {
                    "method": session.preset.name,
                    "experience_graph": (
                        session.config.features.experience_graph
                    ),
                    "run_directory_name": session.run_id,
                    "experiment_id": session.experiment_id,
                    "validated": True,
                },
                "evidence_level": evidence_level,
                "test_mode": self.test_mode,
                "source_commit": _source_revision(self.project_root),
                "executable_digest": session.hashes["code_commit"],
                "p4_v2_source_digest": p4_v2_source_digest(self.project_root),
                "benchmark_commit": _git_commit(self.benchmark_root),
                "evaluator_hash": session.hashes["evaluator_hash"],
                "evaluator_module": "evaluator/process_klayout_reports.py",
                "evaluator_entrypoint": "process_single_file",
                "connectivity_entrypoint": "evaluator/check_connectivity.py",
                "docker_image": session.config.backend.image,
                "docker_image_digest": image_digest,
                "execution_protocol_version": EXECUTION_PROTOCOL_VERSION,
                "execution_epoch": session.execution_epoch,
                "execution_epochs": [session.execution_epoch],
                "resolved_config_hash": session.config.content_hash,
                "scientific_config_hash": scientific_config_hash(
                    session.config
                ),
                "execution_envelope": execution_envelope(
                    session.config
                ).model_dump(mode="json"),
                "continuation": initial_continuation_manifest(
                    session.config,
                    identity=run_identity(
                        run_id=session.run_id,
                        experiment_id=session.experiment_id,
                        case_id=session.case_id,
                        method=session.preset.name,
                        seed=session.seed,
                    ),
                ),
                "official_input_identity": official_input_identity(
                    session.paths
                ),
                "active_run_seconds": 0.0,
                "llm_config_hash": stable_hash(
                    session.config.llm.model_dump(mode="json")
                ),
                "model": {
                    "provider": session.config.llm.provider,
                    "name": session.config.llm.model,
                    "base_url": ("TEST_ONLY" if session.config.llm.provider == "fake" else resolve_base_url(session.config.llm)),
                    "temperature": session.config.llm.temperature,
                    "top_p": session.config.llm.top_p,
                    "effort": session.config.llm.effort,
                    "enable_thinking": session.config.llm.enable_thinking,
                    "thinking_budget": session.config.llm.thinking_budget,
                    "max_output_tokens": session.config.llm.max_output_tokens,
                    "api_key_env": session.config.llm.api_key_env,
                },
                "prompt_version": session.config.llm.prompt_version,
                "skill_version": session.config.llm.skill_version,
                "candidate_schema_version": "2.0",
                "rule_guidance_version": RuleAwareCandidateGuide.VERSION,
                "max_geometry_variants_per_hypothesis": (
                    session.config.agent.max_geometry_variants_per_hypothesis
                ),
                "candidate_evidence": session.config.candidate_evidence.model_dump(
                    mode="json"
                ),
                "candidate_graph": session.config.candidate_graph.model_dump(
                    mode="json"
                ),
                "transaction_batching": session.config.transaction.model_dump(
                    mode="json"
                ),
                "attribution": session.config.attribution.model_dump(mode="json"),
                "candidate_k_including_noop": (
                    session.config.agent.top_k_candidates_including_noop
                ),
                "message_rounds": session.config.agent.message_rounds,
                "max_iterations": session.config.workflow.max_iterations,
                "solver": session.config.coordinator.model_dump(mode="json"),
                "features": session.config.features.model_dump(mode="json"),
                "repair_kernel_integration": {
                    **session.config.repair_kernel_integration.model_dump(mode="json"),
                    "version": "formal-repair-kernel-v1",
                    "integration_code_digest": integration_code_digest(),
                    "formal_runtime_digest": formal_integration_runtime_digest(self.project_root),
                    "phase4r2_contract_sha256": (
                        file_sha256(self.project_root / "contracts" / "phase4r2_closure.json")
                        if (self.project_root / "contracts" / "phase4r2_closure.json").is_file()
                        else "UNAVAILABLE"
                    ),
                    "phase4r2_policy_hash": (
                        json.loads((self.project_root / "contracts" / "phase4r2_d_block1_calibration.json").read_text(encoding="utf-8")).get("selected_policy_hash")
                        if (self.project_root / "contracts" / "phase4r2_d_block1_calibration.json").is_file()
                        else None
                    ),
                    "supported_families": list(
                        session.config.repair_kernel_integration.supported_families
                    ),
                    "supported_rule_ids": list(
                        session.config.repair_kernel_integration.supported_rule_ids
                    ),
                    "invocation_count": 0,
                    "supported_invocation_count": 0,
                    "symbolic_llm_plan_count": 0,
                    "legacy_candidate_intent_call_count": 0,
                    "legacy_lowerer_call_count": 0,
                    "legacy_repair_programmer_call_count": 0,
                },
                "base_experience_hash": base_experience_hash,
                "baseline_snapshot": session.baseline_snapshot.model_dump(mode="json"),
                "actual_llm_call_count": 0,
                "token_usage": {
                    "input_tokens": 0, "output_tokens": 0,
                    "reasoning_tokens": 0, "cache_read_tokens": 0,
                },
                "start_time": utc_now().isoformat(), "status": "RUNNING",
                "conformance_status": {
                    key: "NOT_EVALUATED"
                    for key in ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]
                },
            }
            if session.resume_state is not None:
                session.manifest = resumed_manifest
                epochs = list(session.manifest.get("execution_epochs") or [])
                if session.execution_epoch not in epochs:
                    epochs.append(session.execution_epoch)
                session.manifest.update({
                    "status": "RUNNING",
                    "end_time": None,
                    "execution_protocol_version": EXECUTION_PROTOCOL_VERSION,
                    "execution_epoch": session.execution_epoch,
                    "execution_epochs": epochs,
                    "current_snapshot": session.current_snapshot.model_dump(
                        mode="json"
                    ),
                    "best_snapshot": session.best_snapshot.model_dump(
                        mode="json"
                    ),
                })
            manifest_ref = session.store.write_json(
                "manifest.json", session.manifest,
                producer=(
                    "resume_run" if session.resume_state is not None
                    else "initialize_case"
                ), schema_name="RunManifest",
            )
            if session.resume_state is None:
                identity = run_identity(
                    run_id=session.run_id,
                    experiment_id=session.experiment_id,
                    case_id=session.case_id,
                    method=session.preset.name,
                    seed=session.seed,
                )
                session.checkpoints.save(
                    session.run_id, "initialize_case", session.hashes,
                    {
                        "iteration": 0,
                        "run_identity": identity,
                        "run_identity_hash": stable_hash(identity),
                        "baseline_snapshot": (
                            session.baseline_snapshot.model_dump(mode="json")
                        ),
                        "current_snapshot": session.current_snapshot.model_dump(
                            mode="json"
                        ),
                        "best_snapshot": session.best_snapshot.model_dump(
                            mode="json"
                        ),
                        "best_total": session.best_total,
                        "no_progress": 0,
                        "previous_regions": [],
                        "experience_knowledge_cutoff": (
                            experience.knowledge_cutoff()
                            if session.config.features.experience_graph
                            else None
                        ),
                        "repair_attempt_memory": (
                            session.planner.repair_attempt_memory.artifact()
                        ),
                        "official_input_identity": official_input_identity(
                            session.paths
                        ),
                        "scientific_config_hash": scientific_config_hash(
                            session.config
                        ),
                        "execution_envelope_hash": execution_envelope(
                            session.config
                        ).content_hash,
                        "manifest_ref": manifest_ref.model_dump(mode="json"),
                    },
                )

            def write_provider_health(payload: dict) -> ArtifactRef:
                path = session.run_dir / "logs" / "llm_provider_health.json"
                history = []
                if path.is_file():
                    previous = json.loads(path.read_text(encoding="utf-8"))
                    history = list(previous.get("history") or [])
                    history.append({
                        key: previous.get(key) for key in [
                            "provider", "model", "started_at", "ended_at",
                            "status", "retry_count", "failure_code",
                            "retryable",
                        ]
                    })
                return session.store.write_json(
                    "logs/llm_provider_health.json",
                    {**payload, "history": history},
                    producer="provider_preflight",
                    schema_name="LLMProviderHealth",
                )

            if session.config.llm.provider != "fake":
                health_started = utc_now()
                retries_before = int(
                    logger.reliability_summary().get("retry_attempts", 0)
                )
                session.progress.emit(
                    "provider_preflight_started",
                    "LLM provider preflight started",
                    run_id=session.run_id,
                    provider=session.config.llm.provider,
                    model=session.config.llm.model,
                )
                def record_preflight_failure(
                    failure: InfrastructureFailure,
                ) -> None:
                    failure.with_stage("LLM_PREFLIGHT")
                    retries_after = int(
                        logger.reliability_summary().get(
                            "retry_attempts", retries_before,
                        )
                    )
                    write_provider_health({
                        "provider": session.config.llm.provider,
                        "model": session.config.llm.model,
                        "started_at": health_started.isoformat(),
                        "ended_at": utc_now().isoformat(),
                        "status": "FAILED",
                        "retry_count": retries_after - retries_before,
                        "failure_code": failure.failure_code.value,
                        "retryable": failure.retryable,
                    })
                    session.progress.emit(
                        "provider_preflight_failed",
                        "LLM provider preflight failed",
                        level="ERROR", run_id=session.run_id,
                        provider=session.config.llm.provider,
                        model=session.config.llm.model,
                        failure_code=failure.failure_code.value,
                        retryable=failure.retryable,
                    )

                try:
                    if not hasattr(llm, "preflight"):
                        raise InfrastructureFailure(
                            "real-provider client does not implement preflight",
                            failure_code=FailureCode.CONFIGURATION,
                            retryable=False,
                            provider=session.config.llm.provider,
                            model=session.config.llm.model,
                            original_exception_type=type(llm).__name__,
                            failure_stage="LLM_PREFLIGHT",
                        )
                    probe = await llm.preflight(run_id=session.run_id)
                    if not probe.ok:
                        raise ValueError("provider preflight returned ok=false")
                    probe_provider = str(probe.provider)
                    probe_model = str(probe.model)
                except InfrastructureFailure as exc:
                    record_preflight_failure(exc)
                    raise
                except Exception as exc:
                    failure = InfrastructureFailure(
                        "provider preflight returned an invalid protocol response",
                        failure_code=FailureCode.PROVIDER_PROTOCOL_ERROR,
                        retryable=False,
                        provider=session.config.llm.provider,
                        model=session.config.llm.model,
                        original_exception_type=type(exc).__name__,
                        failure_stage="LLM_PREFLIGHT",
                    )
                    record_preflight_failure(failure)
                    raise failure from exc
                retries_after = int(
                    logger.reliability_summary().get(
                        "retry_attempts", retries_before,
                    )
                )
                write_provider_health({
                        "provider": probe_provider,
                        "model": probe_model,
                        "started_at": health_started.isoformat(),
                        "ended_at": utc_now().isoformat(),
                        "status": "PASS", "retryable": None,
                        "failure_code": None,
                        "retry_count": retries_after - retries_before,
                })
                session.progress.emit(
                    "provider_preflight_completed",
                    "LLM provider preflight completed",
                    run_id=session.run_id, provider=probe_provider,
                    model=probe_model, ok=probe.ok,
                )
            return {
                "config_ref": base["config_ref"],
                "manifest_ref": manifest_ref,
                "baseline_snapshot": base["snapshot"],
                "current_snapshot": session.current_snapshot,
                "best_verified_snapshot": session.best_snapshot,
            }

        async def parse_design_state(state):
            snapshot = session.current_snapshot
            iteration = int(state["iteration"])
            session.violations = ViolationParser().parse_dac26_json(
                Path(snapshot.drc_ref.path), case_id=session.case_id,
                report_iteration=iteration,
                rule_catalog=load_rule_catalog(
                    self.project_root / "configs" / "rules" /
                    "asap7_rule_catalog.yaml"
                ),
            )
            source_objects = SourceObjectMapper().build_map(
                Path(snapshot.script_ref.path)
            )
            session.objects, connectivity_mapping = map_connectivity_components(
                source_objects, session.paths["connectivity"],
            )
            predicate_registry = RulePredicateRegistry(
                session.paths["rule_deck"],
                require_frozen_hash=not self.test_mode,
            )
            supported = [
                item for item in session.violations
                if item.rule_id in predicate_registry.supported_rule_ids
            ]
            supported_layers = {
                layer
                for item in supported
                for layer in predicate_registry.get(
                    item.rule_id
                ).predicate.involved_layers
            }
            session.physical_geometries = (
                SourceHierarchyPhysicalBuilder().build(
                    Path(snapshot.script_ref.path), session.objects,
                    layer_filter=supported_layers,
                ) if supported_layers else []
            )
            witness_builder = RuleWitnessBuilder(predicate_registry)
            witnesses = [
                witness_builder.build(
                    item, session.physical_geometries, session.objects,
                )
                for item in supported
            ]
            session.rule_witnesses = {
                item.violation_id: item.model_dump(mode="json")
                for item in witnesses
            }
            session.rule_knowledge_packs = {
                rule_id: predicate_registry.get(rule_id).model_dump(mode="json")
                for rule_id in sorted({item.rule_id for item in supported})
            }
            session.rule_predicates = {
                rule_id: value["predicate"]
                for rule_id, value in session.rule_knowledge_packs.items()
            }
            topology = LocalTopologyBuilder(
                dbu_per_um=session.config.region.dbu_per_um,
            ).build(session.violations, session.objects)
            session.design = DesignState(
                case_id=session.case_id,
                legal_layers={item.layer for item in session.objects},
                objects={
                    item.object_id: item.model_dump(mode="json")
                    for item in session.objects
                },
                violations={
                    item.violation_id: item.model_dump(mode="json")
                    for item in session.violations
                },
                local_topology=topology.model_dump(mode="json"),
                physical_geometries={
                    item.geometry_id: item.model_dump(mode="json")
                    for item in session.physical_geometries
                },
                rule_predicates=session.rule_predicates,
                rule_witnesses=session.rule_witnesses,
                rule_knowledge_packs=session.rule_knowledge_packs,
                manufacturing_grid_dbu=session.config.backend.manufacturing_grid_dbu,
                dbu_per_um=session.config.region.dbu_per_um,
                source_hashes={
                    item.source_anchor_id: item.source_span.source_hash
                    for item in session.objects
                    if item.source_anchor_id and item.source_span
                },
            )
            prefix = f"iterations/iter_{iteration:04d}/state"
            session.store.write_json(
                f"{prefix}/physical_geometries.json",
                [item.model_dump(mode="json") for item in session.physical_geometries],
                producer="parse_design_state",
                schema_name="FlattenedPhysicalGeometry[]", schema_version="1.0",
            )
            session.store.write_json(
                f"{prefix}/rule_predicates.json", session.rule_predicates,
                producer="parse_design_state",
                schema_name="RulePredicateIRMap", schema_version="3.0",
            )
            session.store.write_json(
                f"{prefix}/rule_witnesses.json", session.rule_witnesses,
                producer="parse_design_state",
                schema_name="RuleWitnessMap", schema_version="1.0",
            )
            session.store.write_json(
                f"{prefix}/rule_knowledge_packs.json",
                session.rule_knowledge_packs, producer="parse_design_state",
                schema_name="RuleKnowledgePackMap", schema_version="1.0",
            )
            session.store.write_json(
                f"{prefix}/local_topology.json", topology,
                producer="parse_design_state",
                schema_name="LocalRoutingTopology", schema_version="1.0",
            )
            session.store.write_json(
                f"{prefix}/connectivity_mapping.json",
                connectivity_mapping, producer="parse_design_state",
                schema_name="ConnectivityMappingReport", schema_version="2.0",
            )
            session.store.write_json(
                f"{prefix}/physical_grounding_coverage.json",
                physical_grounding_coverage(
                    session.objects, session.physical_geometries, witnesses,
                ),
                producer="parse_design_state",
                schema_name="PhysicalGroundingCoverage", schema_version="1.0",
            )
            session.store.write_json(
                f"{prefix}/mapping_coverage.json",
                mapping_coverage(session.objects),
                producer="parse_design_state", schema_name="MappingCoverage",
                schema_version="2.0",
            )
            design_ref = session.store.write_json(
                f"{prefix}/design_state.json", session.design,
                producer="parse_design_state", schema_name="DesignState",
            )
            violations_ref = session.store.write_json(
                f"{prefix}/violations.json",
                [item.model_dump(mode="json") for item in session.violations],
                producer="parse_design_state", schema_name="ViolationRecord[]",
            )
            return {
                "design_state_ref": design_ref,
                "violations_ref": violations_ref,
            }

        async def build_regions(state):
            catalog = load_rule_catalog(
                self.project_root / "configs" / "rules" /
                "asap7_rule_catalog.yaml"
            )
            result = RegionBuilder(session.config.region).build(
                session.violations, session.objects, catalog,
                previous=session.previous_regions,
            )
            session.regions = result.regions
            iteration = int(state["iteration"])
            regions_ref = session.store.write_json(
                f"iterations/iter_{iteration:04d}/state/regions.json",
                [item.model_dump(mode="json") for item in session.regions],
                producer="build_regions", schema_name="RegionState[]",
            )
            return {"regions_ref": regions_ref}

        async def build_agent_graph(state):
            session.potential_access_summaries = session.potential_access_builder.build_all(
                snapshot_id=session.current_snapshot.snapshot_id,
                regions=session.regions,
                design=session.design,
            )
            session.raw_graph = AgentGraphBuilder(
                geometry_edges=session.config.features.geometry_edges,
                proximity_only_score_cap=session.config.agent_graph.proximity_only_score_cap,
                shared_net_edges=session.config.features.shared_net_edges,
                resource_edges=session.config.features.resource_edges,
                timing_edges=False,
            ).build(
                session.regions,
                DesignContext(
                    iteration=int(state["iteration"]), timing_enabled=False,
                ),
                potential_access_summaries=session.potential_access_summaries,
            )
            iteration = int(state["iteration"])
            session.store.write_json(
                f"iterations/iter_{iteration:04d}/state/potential_physical_access.json",
                [item.model_dump(mode="json") for item in session.potential_access_summaries.values()],
                producer="build_agent_graph",
                schema_name="PotentialPhysicalAccessSummary[]",
            )
            return {}

        async def prune_agent_graph(state):
            session.graph = AgentGraphPruner().prune(
                session.raw_graph, session.config.agent_graph,
            )
            report = graph_observability_report(
                session.objects, session.regions,
            )
            session.graph.observability.update({
                "net_evidence_level": report.net_evidence_level,
                "connectivity_component_object_count": (
                    report.connectivity_component_object_count
                ),
                "resource_evidence_level": report.resource_evidence_level,
                "proxy_resource_region_count": (
                    report.proxy_resource_region_count
                ),
                "resource_capacity_known": report.resource_capacity_known,
            })
            iteration = int(state["iteration"])
            session.store.write_json(
                f"iterations/iter_{iteration:04d}/state/agent_graph.json",
                session.graph, producer="prune_agent_graph",
                schema_name="AgentGraph",
            )
            session.store.write_json(
                f"iterations/iter_{iteration:04d}/state/"
                "graph_observability.json",
                report, producer="prune_agent_graph",
                schema_name="GraphObservabilityReport",
            )
            return {}

        async def start_iteration(state):
            iteration = int(state["iteration"]) + 1
            checkpoint = (
                (session.resume_state or {}).get("_checkpoint") or {}
            )
            resume_current_iteration = (
                checkpoint.get("boundary")
                in {"window_commit", "window_complete"}
                and int(
                    (session.resume_state or {}).get(
                        "in_progress_iteration", -1
                    )
                ) == iteration
            )
            if not resume_current_iteration:
                session.iteration_start_snapshot = session.current_snapshot
                session.iteration_start_total_drv = drc_statistics(
                    Path(session.current_snapshot.drc_ref.path)
                ).total_drv
                session.iteration_frontier_violation_ids = {
                    item.violation_id for item in session.violations
                }
                requested=set(session.config.workflow.engineering_target_violation_ids)
                if requested:
                    if not requested<=session.iteration_frontier_violation_ids:
                        raise ValueError("ENGINEERING_TARGET_NOT_IN_CURRENT_BASELINE")
                    session.iteration_frontier_violation_ids.intersection_update(requested)
                session.processed_frontier_violation_ids = set()
                session.deferred_frontier_violation_ids = set()
                session.skipped_frontier_violation_ids = set()
                session.skipped_frontier_by_rule = {}
                session.window_index = 0
                session.window_results = []
                session.iteration_sandbox_jobs_used = 0
            else:
                session.progress.emit(
                    "iteration_window_resume",
                    "Resuming current iteration from a window boundary",
                    run_id=session.run_id, iteration=iteration,
                    boundary=checkpoint["boundary"],
                    window_index=session.window_index,
                    current_snapshot_id=session.current_snapshot.snapshot_id,
                )
            session.active_window = None
            session.all_noop = False
            session.iteration_exit_reason = "FRONTIER_EXHAUSTED"
            session.store.write_json(
                f"iterations/iter_{iteration:04d}/iteration_start.json",
                {
                    "iteration": iteration,
                    "snapshot_id": session.current_snapshot.snapshot_id,
                    "total_drv": session.iteration_start_total_drv,
                    "iteration_frontier_violation_ids": sorted(
                        session.iteration_frontier_violation_ids
                    ),
                    "resumed_from_window_boundary": resume_current_iteration,
                    "next_window_index": session.window_index,
                },
                producer="start_iteration", schema_name="IterationStart",
            )
            return {"active_window": None, "window_result": None}

        async def select_active_subgraphs(state):
            iteration = int(state["iteration"]) + 1
            remaining = (session.config.candidate_evidence.sandbox_max_jobs_per_iteration
                - session.iteration_sandbox_jobs_used)
            if remaining < 3:
                # Keep pending frontier untouched. An iteration may finish
                # without paying for plans that cannot reach required checks.
                session.iteration_exit_reason = "DEFERRED_BUDGET"
                session.active_window = None
                session.subgraphs = []
                session.plans = {}
                session.progress.emit("window_deferred_budget", "No planning without verification capacity",
                    run_id=session.run_id,iteration=iteration,window_index=session.window_index,
                    remaining_sandbox_budget=remaining,required_minimum=3)
                return {"active_subgraphs":[],"active_window":None}
            if (
                session.window_index
                >= session.config.workflow.max_windows_per_iteration
            ):
                session.iteration_exit_reason = (
                    "MAX_WINDOWS_PER_ITERATION"
                )
                session.active_window = None
                session.subgraphs = []
                session.plans = {}
                session.degraded_candidates = {}
                session.progress.emit(
                    "iteration_window_budget_exhausted",
                    "Current iteration reached its rolling-window budget",
                    run_id=session.run_id, iteration=iteration,
                    window_index=session.window_index,
                    max_windows_per_iteration=(
                        session.config.workflow.max_windows_per_iteration
                    ),
                )
                return {"active_subgraphs": [], "active_window": None}
            current_ids = {item.violation_id for item in session.violations}
            pending = (
                session.iteration_frontier_violation_ids
                - session.processed_frontier_violation_ids
            ) & current_ids
            integration = session.config.repair_kernel_integration
            supported_rule_ids = set(integration.supported_rule_ids)
            pending, skipped, skipped_by_rule = (
                partition_supported_pending_frontier(
                    violations=session.violations,
                    pending=pending,
                    integration_enabled=integration.enabled,
                    supported_rule_ids=supported_rule_ids,
                )
            )
            session.skipped_frontier_violation_ids.update(skipped)
            session.skipped_frontier_by_rule.update(skipped_by_rule)
            if not pending:
                session.active_window = None
                session.subgraphs = []
                session.plans = {}
                session.degraded_candidates = {}
                return {"active_subgraphs": [], "active_window": None}
            all_subgraphs = AgentGraphPruner.active_subgraphs(
                session.graph,
                session.config.agent_graph.max_active_subgraph_regions,
            )
            if session.config.workflow.engineering_target_violation_ids:
                from drc_agent.workflow.active_window import engineering_scope_views
                all_subgraphs=engineering_scope_views(session.graph,all_subgraphs,
                    set(session.config.workflow.engineering_target_violation_ids))
            window = select_active_window(
                graph=session.graph, subgraphs=all_subgraphs,
                iteration=iteration, window_index=session.window_index,
                base_snapshot_id=session.current_snapshot.snapshot_id,
                iteration_frontier_violation_ids=(
                    pending
                ),
                processed_frontier_violation_ids=(
                    session.processed_frontier_violation_ids
                ),
                current_violation_ids=current_ids,
                max_parallel_subgraphs=(
                    session.config.workflow.max_parallel_subgraphs
                ),
                max_active_regions=(
                    session.config.agent_graph.max_active_subgraph_regions
                ),
                top_k_candidates_including_noop=(
                    session.config.agent.top_k_candidates_including_noop
                ),
                max_non_noop_actions_per_batch=(
                    session.config.transaction.max_non_noop_actions_per_batch
                ),
                deferred_frontier_violation_ids=(
                    session.deferred_frontier_violation_ids
                ),
            )
            session.active_window = window
            session.plans = {}
            session.degraded_candidates = {}
            if window is None:
                session.subgraphs = []
                return {"active_subgraphs": [], "active_window": None}
            from drc_agent.backends.validation_budget import reserve_window_from_environment
            reserve_window_from_environment(run_id=session.run_id,window_key=window.window_id)
            session.aggregate_candidates = []
            session.aggregate_bundle = None
            session.aggregate_candidate_graph = None
            session.patch_plan = None
            session.transaction_result = None
            session.window_admitted_frontier_violation_ids = set()
            session.window_llm_started_frontier_violation_ids = set()
            session.window_plan_validated_frontier_violation_ids = set()
            session.window_kernel_attempted_frontier_violation_ids = set()
            session.window_physical_evaluated_frontier_violation_ids = set()
            session.window_attempted_frontier_violation_ids = set()
            session.window_unsupported_frontier_violation_ids = set()
            session.window_llm_invocation_count = 0
            session.window_coordination = None
            by_id = {item.subgraph_id: item for item in all_subgraphs}
            session.subgraphs = [by_id[item] for item in window.subgraph_ids]
            prefix = (
                f"iterations/iter_{iteration:04d}/windows/"
                f"window_{window.window_index:04d}"
            )
            session.store.write_json(
                f"{prefix}/window.json", window,
                producer="select_active_window", schema_name="ActiveWindow",
            )
            refs = []
            for subgraph in session.subgraphs:
                ref = session.store.write_json(
                    f"{prefix}/planning/subgraphs/{subgraph.subgraph_id}/"
                    "subgraph.json",
                    subgraph, producer="select_active_window",
                    schema_name="AgentSubgraph",
                )
                refs.append(AgentSubgraphRef(
                    subgraph_id=subgraph.subgraph_id, artifact_ref=ref,
                ))
            session.progress.emit(
                "active_window_selected", "Rolling active window selected",
                run_id=session.run_id, iteration=iteration,
                window_id=window.window_id,
                window_index=window.window_index,
                subgraph_ids=window.subgraph_ids,
                region_ids=window.region_ids,
                decision_region_ids=window.decision_region_ids,
                helper_region_ids=window.helper_region_ids,
                context_only_region_ids=window.context_only_region_ids,
            )
            return {"active_subgraphs": refs, "active_window": window}

        async def plan_subgraph(state):
            ref = state["subgraph_ref"]
            if isinstance(ref, dict):
                ref = AgentSubgraphRef.model_validate(ref)
            subgraph = AgentSubgraph.model_validate_json(
                Path(ref.artifact_ref.path).read_text(encoding="utf-8")
            )
            window = session.active_window
            if window is None:
                raise RuntimeError("SUBGRAPH_PLANNING_WITHOUT_ACTIVE_WINDOW")
            subgraph_regions = set(subgraph.region_ids)
            decision_region_ids = sorted(
                subgraph_regions & set(window.decision_region_ids)
            )
            helper_region_ids = sorted(
                subgraph_regions & set(window.helper_region_ids)
            )
            context_only_region_ids = sorted(
                subgraph_regions & set(window.context_only_region_ids)
            )
            claimed_frontier_violation_ids = sorted({
                violation_id
                for region_id in subgraph.region_ids
                for violation_id in session.graph.regions[region_id].violation_ids
                if violation_id in set(
                    window.claimed_frontier_violation_ids
                    or window.frontier_violation_ids
                )
            })
            kernel_audit = getattr(session.planner, "integration_audit", None)
            audit_events = (
                kernel_audit.events if kernel_audit is not None else []
            )
            audit_event_start = len(audit_events)
            planning_region_ids = set(decision_region_ids + helper_region_ids)

            def failed_call_audit() -> tuple[set[str], int]:
                events = audit_events[audit_event_start:]
                requests = [
                    item for item in events if item.get("event") in {
                        "repair_kernel_plan_requested",
                        "repair_kernel_plan_revision_requested",
                        "repair_kernel_final_plan_requested",
                        "repair_kernel_execution_revision_requested",
                    } and item.get("region_id") in planning_region_ids
                ]
                target_entries = [
                    item for item in events
                    if item.get("event")
                    == "repair_kernel_targets_entered_planning"
                    and item.get("region_id") in planning_region_ids
                ]
                attempted = {
                    str(target) for item in target_entries
                    for target in item.get("target_violation_ids", [])
                } & set(claimed_frontier_violation_ids)
                return attempted, len(requests)
            session.progress.emit(
                "subgraph_planning_started",
                "Planning active subgraph",
                run_id=session.run_id,
                iteration=int(state["iteration"]) + 1,
                subgraph_id=subgraph.subgraph_id,
                region_count=len(subgraph.region_ids),
                decision_region_count=len(decision_region_ids),
                helper_region_count=len(helper_region_ids),
                context_only_region_count=len(context_only_region_ids),
                hierarchical=subgraph.hierarchical,
                expected_primary_llm_calls_upper_bound=(
                    2 * (
                        len(decision_region_ids) + len(helper_region_ids)
                    )
                ),
                candidate_intents_are_llm_semantic=(
                    not session.config.repair_kernel_integration.enabled
                ),
                repair_kernel_integration=(
                    session.config.repair_kernel_integration.enabled
                ),
                affordance_hints_are_non_exhaustive=True,
                max_llm_concurrency=session.config.llm.max_concurrent_requests,
                llm_logical_timeout_seconds=(
                    session.config.llm.timeout_seconds
                ),
                llm_queue_timeout_seconds=(
                    session.config.llm.queue_timeout_seconds
                ),
                llm_http_request_timeout_seconds=(
                    session.config.llm.request_timeout_seconds
                ),
            )
            catalog = load_rule_catalog(
                self.project_root / "configs" / "rules" /
                "asap7_rule_catalog.yaml"
            )
            sandbox_trials_before = int(
                session.sandbox_service.metrics.get("trials", 0)
            )
            try:
                plan = await session.planner.plan(
                    iteration=int(state["iteration"]) + 1,
                    subgraph=subgraph, graph=session.graph,
                    violations=session.violations, objects=session.objects,
                    design=session.design, catalog=catalog,
                    script=Path(session.current_snapshot.script_ref.path),
                    current_snapshot_id=session.current_snapshot.snapshot_id,
                    current_snapshot=session.current_snapshot,
                    baseline_snapshot=session.baseline_snapshot,
                    rule_deck_path=session.paths["rule_deck"],
                    case_id=session.case_id,
                    evaluator_hash=session.hashes["evaluator_hash"],
                    execution_epoch=session.execution_epoch,
                    window_id=session.active_window.window_id,
                    window_index=session.active_window.window_index,
                    artifact_prefix=(
                        f"iterations/iter_{int(state['iteration']) + 1:04d}/"
                        f"windows/window_{session.active_window.window_index:04d}"
                    ),
                    defer_isolated_sandbox=False,
                    defer_final_joint_sandbox=True,
                    sandbox_job_limit=max(
                        session.config.candidate_evidence.sandbox_max_jobs_per_iteration
                        - session.iteration_sandbox_jobs_used,
                        0,
                    ),
                    decision_region_ids=decision_region_ids,
                    helper_region_ids=helper_region_ids,
                    context_only_region_ids=context_only_region_ids,
                    claimed_frontier_violation_ids=claimed_frontier_violation_ids,
                    helper_assignments=[
                        item for item in window.helper_assignments
                        if item.helper_region_id in subgraph_regions
                    ],
                )
            except asyncio.CancelledError:
                raise
            except (InfrastructureFailure, IntegrityFailure):
                raise
            except Exception as exc:
                attempted, llm_calls = failed_call_audit()
                session.window_attempted_frontier_violation_ids.update(
                    attempted
                )
                session.window_llm_invocation_count += llm_calls
                noops = [
                    make_noop_candidate(
                        session.graph.regions[region_id].model_copy(update={
                            "iteration": int(state["iteration"]) + 1,
                        }),
                        subgraph.subgraph_id,
                    )
                    for region_id in sorted(subgraph.region_ids)
                ]
                session.degraded_candidates[subgraph.subgraph_id] = noops
                iteration = int(state["iteration"]) + 1
                failure_ref = session.store.write_json(
                    f"iterations/iter_{iteration:04d}/planning/subgraphs/"
                    f"{subgraph.subgraph_id}/degraded_failure.json",
                    {
                        "status": "DEGRADED_NO_OP",
                        "failure_domain": "SUBGRAPH_LOCAL",
                        "exception_type": type(exc).__name__,
                        "message": str(exc)[:500],
                        "region_ids": sorted(subgraph.region_ids),
                        "noop_candidate_ids": [
                            item.candidate_id for item in noops
                        ],
                    },
                    producer="plan_subgraph",
                    schema_name="DegradedSubgraphFailure",
                )
                return {"subgraph_results": [SubgraphPlanResult(
                    subgraph_id=subgraph.subgraph_id,
                    bundle_ref=None, status="DEGRADED_NO_OP",
                    errors=[
                        f"{type(exc).__name__}: {str(exc)[:460]}",
                        f"failure_artifact={failure_ref.path}",
                        f"attempted_frontier={sorted(attempted)}",
                        f"llm_invocations={llm_calls}",
                    ],
                )]}
            session.window_admitted_frontier_violation_ids.update(
                plan.admitted_frontier_violation_ids
            )
            session.window_llm_started_frontier_violation_ids.update(
                plan.llm_started_frontier_violation_ids
            )
            session.window_plan_validated_frontier_violation_ids.update(
                plan.plan_validated_frontier_violation_ids
            )
            session.window_kernel_attempted_frontier_violation_ids.update(
                plan.kernel_attempted_frontier_violation_ids
            )
            session.window_physical_evaluated_frontier_violation_ids.update(
                plan.physical_evaluated_frontier_violation_ids
            )
            session.window_attempted_frontier_violation_ids.update(
                plan.attempted_frontier_violation_ids
            )
            session.window_unsupported_frontier_violation_ids.update(
                plan.unsupported_frontier_violation_ids
            )
            session.window_llm_invocation_count += plan.llm_invocation_count
            session.iteration_sandbox_jobs_used += max(
                int(session.sandbox_service.metrics.get("trials", 0))
                - sandbox_trials_before,
                0,
            )
            if session.active_window.window_index == 0:
                legacy_prefix = (
                    f"iterations/iter_{int(state['iteration']) + 1:04d}/"
                    f"planning/subgraphs/{subgraph.subgraph_id}"
                )
                for artifact_ref in plan.artifact_refs.values():
                    artifact_path = Path(artifact_ref.path)
                    if artifact_path.is_file():
                        session.store.copy(
                            artifact_path,
                            f"{legacy_prefix}/{artifact_path.name}",
                            producer="planning.legacy_alias",
                            media_type=artifact_ref.media_type,
                        )
            session.plans[subgraph.subgraph_id] = plan
            return {"subgraph_results": [SubgraphPlanResult(
                subgraph_id=subgraph.subgraph_id,
                bundle_ref=plan.artifact_refs["bundle"], status="PLANNED",
            )]}

        async def aggregate_plans(state):
            iteration = int(state["iteration"]) + 1
            window = session.active_window
            if window is None:
                session.selected_all_noop = True
                return {"selected_batches": []}
            prefix = (
                f"iterations/iter_{iteration:04d}/windows/"
                f"window_{window.window_index:04d}"
            )
            plans = [
                session.plans[item.subgraph_id]
                for item in sorted(
                    session.subgraphs, key=lambda value: value.subgraph_id
                )
                if item.subgraph_id in session.plans
            ]
            candidates = [
                candidate for plan in plans for candidate in plan.candidates
            ] + [
                candidate
                for subgraph_id in sorted(session.degraded_candidates)
                for candidate in session.degraded_candidates[subgraph_id]
            ]
            session.aggregate_candidates = candidates
            session.planning_had_failures = (
                bool(session.degraded_candidates)
                or any(
                    result.failures for plan in plans
                    for result in plan.region_results
                )
            )
            session.explored_non_noop = any(
                not candidate.is_noop for candidate in candidates
            )
            _write_jsonl(
                session.run_dir / prefix / "planning/candidates.jsonl",
                [item.model_dump(mode="json") for item in candidates],
            )
            if not candidates:
                session.selected_all_noop = True
                session.all_noop = not session.planning_had_failures
                return {"selected_batches": []}
            remaining = max(
                session.config.candidate_evidence.sandbox_max_jobs_per_iteration
                - session.iteration_sandbox_jobs_used,
                0,
            )
            sandbox_scope = session.subgraphs[0]

            def run_sandbox(selected, patch_plan):
                common = {
                    "iteration": iteration,
                    "subgraph": sandbox_scope,
                    "source_map": session.objects,
                    "script": Path(session.current_snapshot.script_ref.path),
                    "execution_epoch": session.execution_epoch,
                    "window_id": window.window_id,
                    "window_index": window.window_index,
                    "precompiled_patch_plan": patch_plan,
                    "artifact_prefix": prefix,
                    "job_run_id": (
                        f"{session.run_id}-i{iteration:04d}-"
                        f"w{window.window_index:04d}-sandbox"
                    ),
                }
                if len(selected) == 1:
                    return session.sandbox_service.candidate(
                        candidate=selected[0], **common,
                    )
                return session.sandbox_service.bundle(
                    candidates=selected, **common,
                )

            # Window coordination invokes RuntimeSandboxService, whose SQLite-
            # backed cache (and, for B6, experience store) belongs to the
            # workflow thread. Keep the full coordination callback on that
            # thread instead of handing thread-affine resources to a worker.
            coordination = coordinate_active_window(
                window=window, candidates=candidates,
                source_map=session.objects,
                script=Path(session.current_snapshot.script_ref.path),
                snapshot=session.current_snapshot,
                config=session.config,
                resource_capacities=session.design.resource_capacities,
                sandbox=run_sandbox,
                sandbox_jobs_remaining=remaining,
                rule_deck_sha256=file_sha256(session.paths["rule_deck"]),
                evaluator_sha256=session.hashes["evaluator_hash"],
            )
            session.window_coordination = coordination
            session.aggregate_candidate_graph = coordination.candidate_graph
            session.aggregate_bundle = coordination.bundle
            session.patch_plan = coordination.patch_plan
            session.scheduled_batches = coordination.master_batches
            session.iteration_sandbox_jobs_used += coordination.sandbox_jobs_used
            selected = (
                coordination.bundle.selected_candidate_ids
                if coordination.bundle else []
            )
            session.selected_candidate_ids = sorted(selected)
            chosen = {
                item.candidate_id: item for item in candidates
            }
            session.selected_all_noop = not any(
                candidate_id in chosen and not chosen[candidate_id].is_noop
                for candidate_id in selected
            )
            session.all_noop = (
                coordination.status == "NO_OP"
                and not session.explored_non_noop
                and not session.planning_had_failures
            )
            session.store.write_json(
                f"{prefix}/planning/candidate_graph.json",
                coordination.candidate_graph,
                producer="coordinate_active_window",
                schema_name="CandidateGraph",
            )
            session.store.write_json(
                f"{prefix}/planning/window_coordination.json",
                {
                    "status": coordination.status,
                    "selected_candidate_ids": sorted(selected),
                    "rounds": coordination.rounds,
                    "no_good_count": coordination.no_good_count,
                    "sandbox_jobs_used": coordination.sandbox_jobs_used,
                    "evidence_reused": coordination.evidence_reused,
                    "failure_codes": coordination.failure_codes or [],
                    "selection_cardinality_caps_tried": (
                        coordination.selection_cardinality_caps_tried or []
                    ),
                    "selected_non_noop_counts": (
                        coordination.selected_non_noop_counts or []
                    ),
                    "backoff_count": coordination.backoff_count,
                    "master_batch_count": len(coordination.master_batches),
                    "patch_plan_sha256": (
                        stable_hash(coordination.patch_plan.model_dump(mode="json"))
                        if coordination.patch_plan else None
                    ),
                },
                producer="coordinate_active_window",
                schema_name="WindowCoordinationResult",
            )
            bundle_payload = {
                "subgraph_bundles": [
                    plan.bundle.model_dump(mode="json") for plan in plans
                ],
                "aggregate_bundle": (
                    coordination.bundle.model_dump(mode="json")
                    if coordination.bundle else None
                ),
            }
            session.store.write_json(
                f"{prefix}/planning/bundles.json",
                bundle_payload,
                producer="coordinate_active_window",
                schema_name="AggregateBundles",
            )
            # Preserve the first-window aggregate artifact contract for P0-L and
            # resume readers. Window-scoped paths above remain authoritative.
            if window.window_index == 0:
                legacy_prefix = f"iterations/iter_{iteration:04d}/planning"
                _write_jsonl(
                    session.run_dir / legacy_prefix / "candidates.jsonl",
                    [item.model_dump(mode="json") for item in candidates],
                )
                _write_jsonl(
                    session.run_dir / legacy_prefix / "candidate_intents.jsonl",
                    [
                        intent.model_dump(mode="json")
                        for plan in plans
                        for result in plan.region_results
                        for intent in result.intents
                    ],
                )
                session.store.write_json(
                    f"{legacy_prefix}/bundles.json",
                    bundle_payload,
                    producer="planning.legacy_alias",
                    schema_name="AggregateBundles",
                )
            return {"selected_batches": coordination.master_batches}

        async def prepare_transaction(state):
            # coordinate_active_window compiled and exact-sandboxed this plan.
            if (
                session.aggregate_bundle is None
                or session.patch_plan is None
                or len(session.scheduled_batches) != 1
            ):
                session.patch_plan = None
                return {"current_transaction": None}
            return {}

        async def execute_transaction(state):
            if session.patch_plan is None:
                session.transaction_result = None
                return {"current_transaction": None}
            iteration = int(state["iteration"]) + 1
            master_trace = ExecutionTraceContext(
                formal_run_id=session.run_id,
                execution_epoch=session.execution_epoch,
                formal_iteration=iteration,
                window_id=session.active_window.window_id,
                window_index=session.active_window.window_index,
                subgraph_id=(
                    session.active_window.subgraph_ids[0]
                    if len(session.active_window.subgraph_ids) == 1
                    else session.active_window.window_id
                ),
                proposal_id=session.aggregate_bundle.bundle_id,
                parent_snapshot_id=session.current_snapshot.snapshot_id,
                purpose=ExecutionPurpose.MASTER,
                invocation_id="invocation_" + stable_hash({
                    "epoch": session.execution_epoch,
                    "iteration": iteration,
                    "window": session.active_window.window_id,
                    "bundle": session.aggregate_bundle.bundle_id,
                    "parent": session.current_snapshot.snapshot_id,
                    "purpose": "MASTER",
                })[:20],
            )
            session.transaction_result = session.transaction_executor.execute(
                run_id=session.run_id, case_id=session.case_id,
                iteration=iteration,
                baseline_snapshot=session.baseline_snapshot,
                current_snapshot=session.current_snapshot,
                patch_plan=session.patch_plan,
                source_map=session.objects,
                bundle_ids=[session.aggregate_bundle.bundle_id],
                artifact_prefix=(
                    f"iterations/iter_{iteration:04d}/windows/"
                    f"window_{session.active_window.window_index:04d}"
                ),
                job_run_id=(
                    f"{session.run_id}-i{iteration:04d}-"
                    f"w{session.active_window.window_index:04d}"
                ),
                legacy_artifact_prefix=(
                    f"iterations/iter_{iteration:04d}"
                    if session.active_window.window_index == 0 else None
                ),
                trace_context=master_trace,
                physical_effect_fingerprint=stable_hash([
                    build_physical_effect_fingerprint(item).sha256
                    for item in sorted(
                        session.aggregate_candidates,
                        key=lambda value: value.candidate_id,
                    )
                    if item.candidate_id in set(
                        session.aggregate_bundle.selected_candidate_ids
                    )
                    and not item.is_noop
                ]),
            )
            if session.planner is not None and session.planner.integration_audit is not None:
                selected_ids = set(session.aggregate_bundle.selected_candidate_ids)
                kernel_selected = [
                    item for item in session.aggregate_candidates
                    if item.candidate_id in selected_ids
                    and any(
                        edit.provenance.generator == "formal-repair-kernel-v1"
                        for edit in item.edits
                    )
                ]
                if kernel_selected:
                    audit = session.planner.integration_audit
                    audit.master_transaction_kernel_candidate_count += 1
                    candidate_ids = [
                        item.candidate_id for item in kernel_selected
                    ]
                    session.progress.emit(
                        "repair_kernel_candidate_transaction",
                        "Kernel candidate entered formal master transaction",
                        run_id=session.run_id,
                        iteration=iteration,
                        candidate_ids=candidate_ids,
                    )
                    if session.transaction_result.accepted:
                        audit.master_kernel_commit_count += 1
                        session.progress.emit(
                            "repair_kernel_master_commit",
                            "Kernel candidate committed by formal master transaction",
                            run_id=session.run_id,
                            iteration=iteration,
                            candidate_ids=candidate_ids,
                        )
            outcome = session.transaction_result.verification.outcome.value
            session.fatal_failure = outcome in {
                "LAYOUT_GENERATION_FAILURE", "GDS_SANITY_FAILURE",
            } and any(
                code in {
                    "DRC_EXECUTION_FAILED",
                    "INVALID_LYRPT_XML_AFTER_RETRY",
                }
                for code in session.transaction_result.verification.failure_codes
            ) or any(
                code.startswith(("EVALUATOR_FAILURE_", "VERIFICATION_EVIDENCE_INVALID_"))
                for code in session.transaction_result.verification.failure_codes
            )
            return {
                "current_transaction": session.transaction_result.transaction,
            }

        async def verify_result(state):
            if session.transaction_result is None:
                return {"verification_results": []}
            return {
                "verification_results": [
                    session.transaction_result.verification
                ],
            }

        async def commit_or_rollback(state):
            iteration = int(state["iteration"]) + 1
            window = session.active_window
            if window is None:
                return {"window_result": None}
            coordination = session.window_coordination
            removed = new = 0
            removed_ids: set[str] = set()
            connectivity = None
            if session.transaction_result is not None:
                result = session.transaction_result
                removed = result.verification.removed_original_count
                new = result.verification.new_violation_count
                connectivity = result.verification.connectivity_preserved
                removed_ids.update(
                    result.verification.removed_original_violation_ids
                )
                if result.accepted:
                    # A physically accepted child is not master-visible until
                    # its immediate parent receipt can deterministically
                    # rebuild the current semantic context.  Failure leaves
                    # the previous safe snapshot published and reaches the
                    # typed runtime stop boundary.
                    semantic_adapter = FormalCurrentSemanticAdapter(
                        self.project_root,
                        session.config,
                        require_frozen_rule_deck=not self.test_mode,
                    )
                    if session.config.repair_kernel_integration.enabled:
                        semantic_adapter.build_snapshot_context(
                            current=result.committed_snapshot,
                            baseline=session.baseline_snapshot,
                            case_id=session.case_id,
                            rule_deck_path=session.paths["rule_deck"],
                            run_id=session.run_id,
                        )
                    else:
                        semantic_adapter.validate_snapshot_provenance(
                            current=result.committed_snapshot,
                            baseline=session.baseline_snapshot,
                            run_id=session.run_id,
                        )
                    session.current_snapshot = result.committed_snapshot
                    status = "COMMIT"
                    final_snapshot = result.committed_snapshot
                    committed_total = drc_statistics(
                        Path(final_snapshot.drc_ref.path)
                    ).total_drv
                    if committed_total < session.best_total:
                        session.best_total = committed_total
                        session.best_snapshot = final_snapshot
                else:
                    status = "ROLLBACK"
                    final_snapshot = session.current_snapshot
            else:
                status = (
                    "NO_OP" if coordination is not None
                    and coordination.status == "NO_OP"
                    else "NO_SAFE_BUNDLE"
                )
                final_snapshot = session.current_snapshot
            claimed_frontier = set(
                window.claimed_frontier_violation_ids
                or window.frontier_violation_ids
            )
            attempted_frontier = (
                session.window_attempted_frontier_violation_ids
                & claimed_frontier
            )
            unsupported_frontier = (
                session.window_unsupported_frontier_violation_ids
                & claimed_frontier
            )
            deferred_frontier = (
                claimed_frontier - attempted_frontier - unsupported_frontier
            )
            admitted_frontier = (
                session.window_admitted_frontier_violation_ids
                & claimed_frontier
            )
            llm_started_frontier = (
                session.window_llm_started_frontier_violation_ids
                & claimed_frontier
            )
            plan_validated_frontier = (
                session.window_plan_validated_frontier_violation_ids
                & claimed_frontier
            )
            kernel_attempted_frontier = (
                session.window_kernel_attempted_frontier_violation_ids
                & claimed_frontier
            )
            physical_evaluated_frontier = (
                session.window_physical_evaluated_frontier_violation_ids
                & claimed_frontier
            )
            removed_by_commit = (
                removed_ids & session.iteration_frontier_violation_ids
                if status == "COMMIT" else set()
            )
            state_domain = claimed_frontier | removed_by_commit
            frontier_states = {
                violation_id: (
                    FrontierState.REMOVED_BY_COMMIT
                    if violation_id in removed_by_commit else
                    FrontierState.UNSUPPORTED_WITH_EVIDENCE
                    if violation_id in unsupported_frontier else
                    FrontierState.ATTEMPTED_THIS_ITERATION
                    if violation_id in attempted_frontier else
                    FrontierState.PENDING_NOT_YET_ADMITTED
                )
                for violation_id in sorted(state_domain)
            }
            state_counts = {
                state.value: sum(value == state for value in frontier_states.values())
                for state in FrontierState
            }
            result = WindowResult(
                window_id=window.window_id,
                iteration=iteration,
                window_index=window.window_index,
                base_snapshot_id=window.base_snapshot_id,
                final_snapshot_id=final_snapshot.snapshot_id,
                status=status,
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
                unsupported_frontier_violation_ids=sorted(
                    unsupported_frontier
                ),
                removed_by_commit_violation_ids=sorted(removed_by_commit),
                frontier_state_counts=state_counts,
                frontier_states=frontier_states,
                decision_region_ids=window.decision_region_ids,
                helper_region_ids=window.helper_region_ids,
                context_only_region_ids=window.context_only_region_ids,
                total_region_count=len(window.region_ids),
                llm_invocation_count=session.window_llm_invocation_count,
                selected_candidate_ids=sorted(
                    session.aggregate_bundle.selected_candidate_ids
                    if session.aggregate_bundle else []
                ),
                removed_original=removed,
                new_violations=new,
                connectivity_preserved=connectivity,
                joint_sandbox_rounds=(
                    coordination.rounds if coordination else 0
                ),
                no_good_count=(
                    coordination.no_good_count if coordination else 0
                ),
            )
            session.processed_frontier_violation_ids.update(
                attempted_frontier | unsupported_frontier
            )
            session.deferred_frontier_violation_ids.update(deferred_frontier)
            session.deferred_frontier_violation_ids.difference_update(
                attempted_frontier
            )
            session.window_results.append(result)
            prefix = (
                f"iterations/iter_{iteration:04d}/windows/"
                f"window_{window.window_index:04d}"
            )
            session.store.write_json(
                f"{prefix}/window_result.json", result,
                producer="commit_or_rollback_window",
                schema_name="WindowResult",
            )
            session.window_index += 1
            if status == "COMMIT":
                if session.config.features.experience_graph:
                    # A next-window checkpoint must never skip an unpublished
                    # committed window. Startup validates and publishes this
                    # immutable journal before any B6 retrieval.
                    from drc_agent.experience.window import publish_runtime_window
                    publish_runtime_window(session, iteration, stage_only=True)
                session.checkpoints.save(
                    session.run_id, "window_commit", session.hashes,
                    _window_checkpoint_state(
                        session, in_progress_iteration=iteration,
                    ),
                )
            session.progress.emit(
                "active_window_finished", "Rolling active window finished",
                run_id=session.run_id, iteration=iteration,
                window_id=window.window_id, window_index=window.window_index,
                status=status, final_snapshot_id=final_snapshot.snapshot_id,
                total_regions=len(window.region_ids),
                decision_regions=len(window.decision_region_ids),
                helper_regions=len(window.helper_region_ids),
                context_only_regions=len(window.context_only_region_ids),
                llm_invocation_count=session.window_llm_invocation_count,
                claimed_target_count=len(claimed_frontier),
                attempted_target_count=len(attempted_frontier),
                deferred_target_count=len(deferred_frontier),
                unsupported_target_count=len(unsupported_frontier),
            )
            return {"window_result": result}

        async def update_experience(state):
            if not session.config.features.experience_graph:
                return {}
            from drc_agent.experience.window import publish_runtime_window
            iteration = int(state["iteration"]) + 1
            window_prefix = (
                f"iterations/iter_{iteration:04d}/windows/"
                f"window_{session.active_window.window_index:04d}"
            )
            publication = publish_runtime_window(session, iteration)
            session.store.write_json(
                f"{window_prefix}/planning/experience_admissions.json",
                publication, producer="update_experience",
                schema_name="WindowExperiencePublication",
            )
            return {}

        async def update_agent_graph(state):
            iteration = int(state["iteration"]) + 1
            from drc_agent.backends.validation_budget import reserve_window_from_environment
            if session.active_window is not None:
                reserve_window_from_environment(run_id=session.run_id,
                    window_key=session.active_window.window_id,release=True)
            window = session.active_window
            result = session.window_results[-1] if session.window_results else None
            if window is None or result is None or result.status != "COMMIT":
                if window is not None and result is not None:
                    session.checkpoints.save(
                        session.run_id, "window_complete", session.hashes,
                        _window_checkpoint_state(
                            session, in_progress_iteration=iteration,
                        ),
                    )
                return {}
            invalidated = (
                session.sandbox_service.cache.invalidate_all()
                if session.sandbox_service is not None else 0
            )
            old_region_count = len(session.regions)
            old_edge_count = len(getattr(session.graph, "edges", []))
            old_snapshot_id = window.base_snapshot_id
            refresh_updates = {}
            refresh_state = {**state, "iteration": iteration}
            for refresh_node in (
                parse_design_state, build_regions,
                build_agent_graph, prune_agent_graph,
            ):
                refresh_updates.update(await refresh_node({
                    **refresh_state, **refresh_updates,
                }))
            selected_ids = set(result.selected_candidate_ids)
            footprints = [
                candidate.edit_footprint_dbu.model_dump(mode="json")
                for candidate in session.aggregate_candidates
                if candidate.candidate_id in selected_ids
                and not candidate.is_noop
            ]
            prefix = (
                f"iterations/iter_{iteration:04d}/windows/"
                f"window_{window.window_index:04d}/refresh"
            )
            refresh_ref = session.store.write_json(
                f"{prefix}/agent_graph_refresh.json",
                {
                    "contract": "POST_WINDOW_GRAPH_STATE_REFRESH",
                    "mode": "FULL_REBUILD_FALLBACK",
                    "old_snapshot_id": old_snapshot_id,
                    "new_snapshot_id": session.current_snapshot.snapshot_id,
                    "changed_area": footprints,
                    "old_region_count": old_region_count,
                    "new_region_count": len(session.regions),
                    "old_edge_count": old_edge_count,
                    "new_edge_count": len(getattr(session.graph, "edges", [])),
                    "region_lineage_ids": sorted(
                        region.lineage_id for region in session.regions
                    ),
                    "stale_sandbox_entries_invalidated": invalidated,
                    "next_window_context_snapshot_id": (
                        session.current_snapshot.snapshot_id
                    ),
                },
                producer="refresh_after_window",
                schema_name="AgentGraphRefreshReport",
            )
            session.checkpoints.save(
                session.run_id, "window_complete", session.hashes,
                _window_checkpoint_state(
                    session, in_progress_iteration=iteration,
                ),
            )
            return {
                "design_state_ref": refresh_updates.get("design_state_ref"),
                "violations_ref": refresh_updates.get("violations_ref"),
                "regions_ref": refresh_updates.get("regions_ref"),
                "agent_graph_ref": refresh_ref,
            }

        async def finish_iteration(state):
            iteration = int(state["iteration"]) + 1
            stats = drc_statistics(Path(session.current_snapshot.drc_ref.path))
            made_progress = stats.total_drv < session.iteration_start_total_drv
            if session.iteration_exit_reason != "DEFERRED_BUDGET":
                session.no_progress = 0 if made_progress else session.no_progress + 1
            session.completed_iteration = iteration
            statuses = [item.status for item in session.window_results]
            session.all_noop = (
                bool(statuses)
                and all(item == "NO_OP" for item in statuses)
                and not session.planning_had_failures
                and not session.explored_non_noop
            )
            if "COMMIT" in statuses:
                transaction_status = IterationTransactionStatus.COMMIT
            elif "ROLLBACK" in statuses:
                transaction_status = IterationTransactionStatus.ROLLBACK
            elif session.all_noop:
                transaction_status = IterationTransactionStatus.NO_OP
            else:
                transaction_status = IterationTransactionStatus.NO_PROGRESS
            iteration_prefix = f"iterations/iter_{iteration:04d}"
            final_window = (
                session.window_results[-1] if session.window_results else None
            )
            last_committed_window = next(
                (
                    item for item in reversed(session.window_results)
                    if item.status == "COMMIT"
                ),
                None,
            )
            final_master_attempt_pointer_ref = None
            if last_committed_window is not None:
                if (
                    last_committed_window.final_snapshot_id
                    != session.current_snapshot.snapshot_id
                ):
                    raise RuntimeError(
                        "ITERATION_FINAL_COMMITTED_SNAPSHOT_MISMATCH"
                    )
                latest_attempt_path = (
                    session.run_dir / iteration_prefix / "windows"
                    / f"window_{last_committed_window.window_index:04d}"
                    / "transaction" / "latest_attempt.json"
                )
                if not latest_attempt_path.is_file():
                    raise RuntimeError(
                        "ITERATION_FINAL_MASTER_ATTEMPT_POINTER_MISSING"
                    )
                final_master_attempt_pointer_ref = ArtifactRef.from_path(
                    latest_attempt_path,
                    producer="finish_iteration",
                    media_type="application/json",
                    schema_name="LatestAttemptPointer",
                )
            iteration_final = IterationFinalEvidencePointer(
                iteration=iteration,
                window_count=len(session.window_results),
                final_window_id=(
                    final_window.window_id if final_window else None
                ),
                final_window_index=(
                    final_window.window_index if final_window else None
                ),
                final_window_status=(
                    final_window.status if final_window else None
                ),
                final_snapshot=session.current_snapshot,
                final_total_drv=stats.total_drv,
                last_committed_window_id=(
                    last_committed_window.window_id
                    if last_committed_window else None
                ),
                last_committed_window_index=(
                    last_committed_window.window_index
                    if last_committed_window else None
                ),
                final_master_attempt_pointer_ref=(
                    final_master_attempt_pointer_ref
                ),
            )
            iteration_final_ref = session.store.write_json(
                f"{iteration_prefix}/iteration_final.json",
                iteration_final,
                producer="finish_iteration",
                schema_name="IterationFinalEvidencePointer",
            )
            ledger_path_value = os.environ.get("DRC_VALIDATION_LEDGER")
            production_metrics = collect_production_execution_metrics(
                session.run_dir,
                run_id=session.run_id,
                iteration=iteration,
                ledger_path=(
                    Path(ledger_path_value) if ledger_path_value else None
                ),
            )
            production_metrics_ref = session.store.write_json(
                f"{iteration_prefix}/metrics/production_execution_metrics.json",
                production_metrics,
                producer="finish_iteration",
                schema_name="ProductionExecutionMetrics",
            )
            candidate_metrics = production_metrics["candidate_workload"]
            attempt_metrics = production_metrics["physical_attempts"]
            tool_metrics = production_metrics["tool_execution"]
            graph_metrics = production_metrics["agent_graph_snapshot"]
            graph_observability = (
                getattr(session.graph, "observability", {}) or {}
            )
            usage = llm_usage_for_iteration(session.llm_log, iteration)
            metrics = IterationMetrics(
                iteration=iteration,
                metrics_semantics_version="2.0",
                transaction_status=transaction_status,
                committed_total_drv=stats.total_drv,
                attempted_total_drv=stats.total_drv,
                removed_original_drv=max(
                    session.iteration_start_total_drv - stats.total_drv, 0
                ),
                new_drv=(
                    sum(int(item.new_violations) for item in session.window_results)
                    if all(
                        item.new_violations is not None
                        for item in session.window_results
                    ) else None
                ),
                connectivity_preserved=(
                    all(
                        item.connectivity_preserved is True
                        for item in session.window_results
                    )
                    if all(
                        item.connectivity_preserved is not None
                        for item in session.window_results
                    ) else None
                ),
                drc_by_rule=stats.drc_by_rule,
                drc_by_marker_type=stats.drc_by_marker_type,
                drc_by_rule_and_type=stats.drc_by_rule_and_type,
                region_count=len(session.regions),
                window_count=len(session.window_results),
                active_subgraph_count=int(
                    candidate_metrics["active_subgraph_workload_count"]
                ),
                candidate_count=int(candidate_metrics["candidate_count"]),
                non_noop_candidate_count=int(
                    candidate_metrics["non_noop_candidate_count"]
                ),
                selected_non_noop_count=int(
                    candidate_metrics["selected_non_noop_count"]
                ),
                tool_runtime_seconds=float(
                    tool_metrics["aggregate_tool_busy_seconds"]
                ),
                raw_agent_edge_count=len(
                    getattr(session.raw_graph, "edges", [])
                ),
                pruned_agent_edge_count=len(
                    getattr(session.graph, "edges", [])
                ),
                hard_component_count=int(
                    graph_observability.get("hard_component_count") or 0
                ),
                largest_hard_component_size=int(
                    graph_observability.get(
                        "largest_hard_component_size"
                    ) or 0
                ),
                planning_view_count=int(
                    graph_metrics.get("planning_view_count") or 0
                ),
                boundary_dependency_count=int(
                    graph_metrics.get("boundary_dependency_count") or 0
                ),
                candidate_graph_edge_count=int(
                    candidate_metrics["candidate_graph_edge_count"]
                ),
                candidate_graph_completeness_milli=int(
                    candidate_metrics[
                        "candidate_graph_completeness_milli"
                    ]
                ),
                unresolved_candidate_pair_count=int(
                    candidate_metrics[
                        "unresolved_candidate_pair_count"
                    ]
                ),
                sandbox_trial_count=int(attempt_metrics["registered"]),
                sandbox_clean_progress_count=int(
                    attempt_metrics["accepted"]
                ),
                sandbox_cache_hit_count=int(
                    (session.sandbox_service.metrics or {}).get(
                        "cache_hits", 0
                    )
                    if session.sandbox_service is not None else 0
                ),
                selected_sandbox_verified_count=int(
                    candidate_metrics["selected_sandbox_verified_count"]
                ),
                selected_proven_count=int(
                    candidate_metrics["selected_proven_count"]
                ),
                selected_heuristic_count=int(
                    candidate_metrics["selected_heuristic_count"]
                ),
                transaction_batch_count=sum(
                    item.status in {"COMMIT", "ROLLBACK"}
                    for item in session.window_results
                ),
                attribution_quality="ARTIFACT_BACKED_PRODUCTION_V1",
                artifact_paths={
                    "committed_drc": session.current_snapshot.drc_ref.path,
                    "committed_script": session.current_snapshot.script_ref.path,
                    "iteration_final": iteration_final_ref.path,
                    "production_execution_metrics": (
                        production_metrics_ref.path
                    ),
                },
                **usage,
            )
            persist_iteration_metrics(
                metrics,
                store=IterationMetricsStore(
                    session.run_dir / "metrics" / "iteration_metrics.jsonl"
                ),
                artifact_writer=session.store.write_json,
                artifact_path=(
                    f"iterations/iter_{iteration:04d}/metrics/"
                    "iteration_metrics.json"
                ),
            )
            summary_ref = session.store.write_json(
                f"iterations/iter_{iteration:04d}/iteration_summary.json",
                {
                    "iteration": iteration,
                    "start_snapshot_id": (
                        session.iteration_start_snapshot.snapshot_id
                    ),
                    "final_snapshot_id": session.current_snapshot.snapshot_id,
                    "start_total_drv": session.iteration_start_total_drv,
                    "final_total_drv": stats.total_drv,
                    "progress": made_progress,
                    "no_progress": session.no_progress,
                    "frontier_violation_ids": sorted(
                        session.iteration_frontier_violation_ids
                    ),
                    "processed_frontier_violation_ids": sorted(
                        session.processed_frontier_violation_ids
                    ),
                    "attempted_frontier_violation_ids": sorted(
                        session.processed_frontier_violation_ids
                    ),
                    "deferred_frontier_violation_ids": sorted(
                        session.deferred_frontier_violation_ids
                    ),
                    "skipped_frontier_violation_ids": sorted(
                        session.skipped_frontier_violation_ids
                    ),
                    "skipped_frontier_violation_count": len(
                        session.skipped_frontier_violation_ids
                    ),
                    "skipped_frontier_by_rule": {
                        rule_id: sum(
                            item == rule_id
                            for item in session.skipped_frontier_by_rule.values()
                        )
                        for rule_id in sorted(set(
                            session.skipped_frontier_by_rule.values()
                        ))
                    },
                    "iteration_exit_reason": session.iteration_exit_reason,
                    "iteration_final_ref": iteration_final_ref.model_dump(
                        mode="json"
                    ),
                    "production_execution_metrics_ref": (
                        production_metrics_ref.model_dump(mode="json")
                    ),
                    "window_count": len(session.window_results),
                    "windows": [
                        item.model_dump(mode="json")
                        for item in session.window_results
                    ],
                },
                producer="finish_iteration",
                schema_name="IterationSummary",
            )
            session.checkpoints.save(
                session.run_id, "iteration_complete", session.hashes,
                {
                    **_window_checkpoint_state(
                        session, in_progress_iteration=iteration,
                    ),
                    "iteration": iteration,
                    "baseline_snapshot": (
                        session.baseline_snapshot.model_dump(mode="json")
                    ),
                    "current_snapshot": session.current_snapshot.model_dump(mode="json"),
                    "best_snapshot": session.best_snapshot.model_dump(mode="json"),
                    "best_total": session.best_total,
                    "no_progress": session.no_progress,
                    "previous_regions": [
                        item.model_dump(mode="json")
                        for item in session.previous_regions
                    ],
                    "repair_attempt_memory": (
                        session.planner.repair_attempt_memory.artifact()
                    ),
                    "iteration_summary_ref": summary_ref.model_dump(mode="json"),
                    "iteration_final_ref": iteration_final_ref.model_dump(
                        mode="json"
                    ),
                    "production_execution_metrics_ref": (
                        production_metrics_ref.model_dump(mode="json")
                    ),
                    "official_input_identity": official_input_identity(
                        session.paths
                    ),
                    "scientific_config_hash": scientific_config_hash(
                        session.config
                    ),
                    "execution_envelope_hash": execution_envelope(
                        session.config
                    ).content_hash,
                },
            )
            return {"iteration": iteration}

        async def check_stopping_condition(state):
            residual = drc_statistics(
                Path(session.current_snapshot.drc_ref.path)
            ).total_drv
            status = stopping_condition(
                residual_count=residual, connectivity_preserved=True,
                iteration=int(state["iteration"]),
                max_iterations=session.config.workflow.max_iterations,
                no_progress=session.no_progress,
                max_no_progress=(
                    session.config.workflow.max_no_progress_iterations
                ),
                all_noop=session.all_noop,
                budget_exhausted=(
                    session.config.workflow.whole_run_budget_seconds is not None
                    and session.active_run_seconds_before
                    + time.monotonic() - session.run_started_monotonic
                    >= session.config.workflow.whole_run_budget_seconds
                ),
                integrity_failure=session.fatal_failure,
            )
            return {"stopping_status": status}

        async def finalize_run(state):
            if session.sandbox_service is not None:
                session.sandbox_service.close()
            if isinstance(session.experience_store, ExperienceStore):
                session.experience_store.export(
                    session.run_dir / "experience" / "jsonl"
                )
                session.experience_store.close()
            all_usage = {
                "llm_calls": 0, "input_tokens": 0,
                "output_tokens": 0, "reasoning_tokens": 0,
                "cache_read_tokens": 0,
            }
            for iteration in range(1, session.completed_iteration + 1):
                usage = llm_usage_for_iteration(session.llm_log, iteration)
                for key in all_usage:
                    all_usage[key] += usage[key]
            official_score_ref = None
            if session.best_snapshot and session.best_snapshot.drc_ref:
                evaluator = self._adapter or DAC26ReportAdapter(
                    self.benchmark_root,
                    runtime_roots=[session.config.backend.stage_root],
                )
                score = evaluator.score_repair(
                    session.baseline_snapshot.drc_ref,
                    session.best_snapshot.drc_ref,
                )
                official_score_ref = session.store.write_json(
                    "score/official.json", score,
                    producer="finalize_run",
                    schema_name="OfficialRepairScore",
                )
            status_value = state["stopping_status"]
            if hasattr(status_value, "value"):
                status_value = status_value.value
            session.final_status = f"COMPLETED_{status_value}"
            initial_total = drc_statistics(
                Path(session.baseline_snapshot.drc_ref.path)
            ).total_drv
            scope_pass = bool(session.graph is not None) and all(
                (
                    item.scope_kind != "FLAT"
                    or len(item.region_ids)
                    <= session.config.agent_graph.max_active_subgraph_regions
                ) and (
                    item.scope_kind != "PLANNING_VIEW"
                    or len(item.region_ids)
                    <= session.config.agent_graph.max_llm_view_regions
                )
                for item in session.subgraphs
            )
            candidate_audit = (
                session.aggregate_candidate_graph.audit
                if session.aggregate_candidate_graph is not None else None
            )
            run_conformance = {
                "C0": "SEE_PROJECT_CONFORMANCE_REPORT",
                "C1": "PASS",
                "C2": "SEE_PROJECT_CONFORMANCE_REPORT",
                "C3": "PASS" if scope_pass else "FAIL",
                "C4": (
                    "PASS" if candidate_audit is not None
                    and candidate_audit.completeness_milli == 1000
                    else "NOT_EVALUATED"
                ),
                "C5": "SEE_PROJECT_CONFORMANCE_REPORT",
                "C6": "PASS" if session.best_total < initial_total else "FAIL",
            }
            conformance_ref = session.store.write_json(
                "metrics/conformance_status.json", run_conformance,
                producer="finalize_run", schema_name="ConformanceStatus",
                schema_version="2.0",
            )
            run_metrics_ref = session.store.write_json(
                "metrics/run_metrics.json", {
                    "initial_drv": initial_total,
                    "best_drv": session.best_total,
                    "iterations_completed": session.completed_iteration,
                    **all_usage,
                }, producer="finalize_run", schema_name="RunMetrics",
                schema_version="2.0",
            )
            ledger_path_value = os.environ.get("DRC_VALIDATION_LEDGER")
            production_execution_metrics = (
                collect_production_execution_metrics(
                    session.run_dir,
                    run_id=session.run_id,
                    ledger_path=(
                        Path(ledger_path_value)
                        if ledger_path_value else None
                    ),
                )
            )
            production_execution_metrics_ref = session.store.write_json(
                "metrics/production_execution_metrics.json",
                production_execution_metrics,
                producer="finalize_run",
                schema_name="ProductionExecutionMetrics",
            )
            session.manifest.update({
                "end_time": utc_now().isoformat(),
                "active_run_seconds": (
                    session.active_run_seconds_before
                    + max(
                        0.0,
                        time.monotonic() - session.run_started_monotonic,
                    )
                ),
                "status": session.final_status,
                "stopping_status": status_value,
                "iterations_completed": session.completed_iteration,
                "actual_llm_call_count": all_usage["llm_calls"],
                "token_usage": {
                    key: value for key, value in all_usage.items()
                    if key != "llm_calls"
                },
                "best_snapshot": session.best_snapshot.model_dump(mode="json"),
                "current_snapshot": session.current_snapshot.model_dump(mode="json"),
                "official_score_ref": (
                    official_score_ref.model_dump(mode="json")
                    if official_score_ref else None
                ),
                "metrics_ref": run_metrics_ref.model_dump(mode="json"),
                "conformance_status": run_conformance,
                "conformance_ref": conformance_ref.model_dump(mode="json"),
                "project_conformance_report": (
                    "docs/reports/v2_conformance_2026-08-20.json"
                ),
                "sandbox_metrics": (
                    dict(session.sandbox_service.metrics)
                    if session.sandbox_service is not None else {}
                ),
                "sandbox_metrics_contract": {
                    "schema_version": "1.0",
                    "scope": (
                        "WINDOW_COORDINATION_RUNTIME_SANDBOX_SERVICE_ONLY"
                    ),
                    "excludes": [
                        "ROOT", "DEBT", "CONDENSATION", "MASTER",
                    ],
                },
                "production_execution_metrics_ref": (
                    production_execution_metrics_ref.model_dump(mode="json")
                ),
                "llm_reliability": (
                    session.llm_logger.reliability_summary()
                    if session.llm_logger is not None else {}
                ),
            })
            session.manifest["repair_kernel_integration"] = (
                _current_repair_kernel_audit(
                    session, audit_complete=True,
                )
            )
            if session.resume_state is not None:
                history = list(session.manifest.get("resume_history") or [])
                if history:
                    history[-1].update({
                        "ended_at": utc_now().isoformat(),
                        "status": session.final_status,
                    })
                session.manifest["resume_history"] = history
                session.progress.emit(
                    "resume_completed", "Resumed workflow completed",
                    run_id=session.run_id,
                    iteration=session.completed_iteration,
                    status=session.final_status,
                )
            manifest_ref = session.store.write_json(
                "manifest.json", session.manifest,
                producer="finalize_run", schema_name="RunManifest",
                schema_version="2.0",
            )
            session.checkpoints.save(
                session.run_id, "finalize_run", session.hashes,
                {
                    "iteration": session.completed_iteration,
                    "status": session.final_status,
                    "manifest_ref": manifest_ref.model_dump(mode="json"),
                },
            )
            session.checkpoints.connection.close()
            return {"manifest_ref": manifest_ref}

        raw_nodes = {
            "initialize_case": initialize_case,
            "parse_design_state": parse_design_state,
            "build_regions": build_regions,
            "build_agent_graph": build_agent_graph,
            "prune_agent_graph": prune_agent_graph,
            "start_iteration": start_iteration,
            "select_active_window": select_active_subgraphs,
            "plan_subgraph": plan_subgraph,
            "coordinate_active_window": aggregate_plans,
            "prepare_transaction": prepare_transaction,
            "execute_transaction": execute_transaction,
            "verify_result": verify_result,
            "commit_or_rollback_window": commit_or_rollback,
            "update_experience": update_experience,
            "refresh_after_window": update_agent_graph,
            "finish_iteration": finish_iteration,
            "check_stopping_condition": check_stopping_condition,
            "finalize_run": finalize_run,
        }

        def instrument(name, node):
            async def instrumented(state):
                started = time.monotonic()
                session.active_nodes.add(name)
                session.progress.emit(
                    "workflow_node_started", f"Starting {name}",
                    run_id=session.run_id, node=name,
                    iteration=state.get("iteration"),
                )
                try:
                    result = await node(state)
                except asyncio.CancelledError:
                    session.interrupted_node = name
                    session.progress.emit(
                        "workflow_node_cancelled", f"Cancelled {name}",
                        level="WARNING", run_id=session.run_id,
                        node=name, iteration=state.get("iteration"),
                        duration_seconds=round(time.monotonic() - started, 3),
                    )
                    raise
                except BaseException as exc:
                    if isinstance(exc, KeyboardInterrupt):
                        session.interrupted_node = name
                    session.progress.emit(
                        "workflow_node_failed", f"Failed {name}",
                        level="ERROR", run_id=session.run_id,
                        node=name, iteration=state.get("iteration"),
                        duration_seconds=round(time.monotonic() - started, 3),
                        error_code=type(exc).__name__,
                    )
                    raise
                else:
                    session.progress.emit(
                        "workflow_node_completed", f"Completed {name}",
                        run_id=session.run_id, node=name,
                        iteration=state.get("iteration"),
                        duration_seconds=round(time.monotonic() - started, 3),
                    )
                    return result
                finally:
                    session.active_nodes.discard(name)

            return instrumented

        return {
            name: instrument(name, node)
            for name, node in raw_nodes.items()
        }

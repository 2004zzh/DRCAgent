from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field

from .common import ArtifactRef, Box, StrictModel


class BackendCapability(StrEnum):
    PREPARE_JOB = "prepare_job"
    GENERATE_LAYOUT = "generate_layout"
    RUN_DRC = "run_drc"
    RUN_CONNECTIVITY = "run_connectivity"
    RUN_LVS = "run_lvs"
    RUN_TIMING = "run_timing"
    APPLY_REPAIR = "apply_repair"
    COLLECT_RESULTS = "collect_results"
    CLEANUP = "cleanup"


class EvidenceValidity(StrEnum):
    """Whether a verification value is backed by this physical attempt."""

    NOT_EVALUATED = "NOT_EVALUATED"
    VALID = "VALID"
    ERROR = "ERROR"


class ExecutionPurpose(StrEnum):
    UNKNOWN = "UNKNOWN"
    BASELINE = "BASELINE"
    ROOT = "ROOT"
    DEBT = "DEBT"
    CONDENSATION = "CONDENSATION"
    CANDIDATE = "CANDIDATE"
    JOINT = "JOINT"
    MASTER = "MASTER"


class AttemptLifecycle(StrEnum):
    PREPARING = "PREPARING"
    READY = "READY"
    RUNNING = "RUNNING"
    LAYOUT_GENERATED = "LAYOUT_GENERATED"
    DRC_VERIFIED = "DRC_VERIFIED"
    PUBLISHING = "PUBLISHING"
    PUBLISHED = "PUBLISHED"
    PREPARE_FAILED = "PREPARE_FAILED"
    TERMINAL = "TERMINAL"
    RETAINED = "RETAINED"
    CLEANED = "CLEANED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"


class AttemptTerminalStatus(StrEnum):
    EVALUATED = "EVALUATED"
    EXECUTION_FAILURE = "EXECUTION_FAILURE"
    CANCELLED = "CANCELLED"


class P32ProductionCanaryChecks(StrictModel):
    """Exact safety polarity required from the P3.2 production canary."""

    all_plan_bindings_pass: Literal[True]
    all_plans_real_llm: Literal[True]
    candidate_graph_reached: Literal[True]
    cp_sat_selected_nonnoop: Literal[True]
    historical_patch_injection: Literal[False]
    http_attempt_cap_respected: Literal[True]
    master_acceptance_unchanged: Literal[True]
    normal_production_trajectory_used: Literal[True]
    positive_master_commit: Literal[True]
    serialized_master_path: Literal[True]
    two_distinct_regions_planned: Literal[True]


class ExecutionTraceContext(StrictModel):
    """Stable logical trace context; never used as workspace ownership."""

    protocol_version: str = "p3.2-execution-v1"
    formal_run_id: str
    execution_epoch: str = "epoch_initial"
    formal_iteration: int = Field(default=0, ge=0)
    window_id: str | None = None
    window_index: int | None = Field(default=None, ge=0)
    subgraph_id: str | None = None
    region_id: str | None = None
    plan_id: str | None = None
    plan_revision: int = Field(default=0, ge=0)
    target_violation_id: str | None = None
    root_proposal_id: str | None = None
    proposal_id: str | None = None
    parent_snapshot_id: str | None = None
    depth: int = Field(default=0, ge=0)
    purpose: ExecutionPurpose = ExecutionPurpose.UNKNOWN
    invocation_id: str | None = None


class ExpectedArtifact(StrictModel):
    path: Path
    media_type: str = "application/octet-stream"
    required: bool = True


class CommandSpec(StrictModel):
    tool_name: str
    action: str
    executable: str
    argv: list[str]
    cwd: Path
    env_sources: list[Path] = Field(default_factory=list)
    extra_env: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: int
    expected_artifacts: list[ExpectedArtifact] = Field(default_factory=list)
    allowed_exit_codes: set[int] = Field(default_factory=lambda: {0})
    network_policy: Literal["NONE", "BENCHMARK_DEFAULT", "REQUIRED"] = "NONE"
    attempt_id: str | None = None
    owner_token_hash: str | None = None
    container_name: str | None = None
    cidfile: Path | None = None


class ToolResult(StrictModel):
    tool_name: str
    action: str
    status: Literal[
        "SUCCESS", "FAILED", "TIMEOUT", "UNSUPPORTED", "INVALID_INPUT",
        "MISSING_ARTIFACT", "CANCELLED",
    ]
    return_code: int | None = None
    started_at: datetime
    ended_at: datetime
    runtime_seconds: float
    stdout_ref: ArtifactRef | None = None
    stderr_ref: ArtifactRef | None = None
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    metrics: dict[str, int | float | str | bool | None] = Field(default_factory=dict)
    command_fingerprint: str
    error_code: str | None = None
    error_message: str | None = None
    attempt_id: str | None = None
    owner_token_hash: str | None = None
    container_name: str | None = None


class PrepareJobRequest(StrictModel):
    run_id: str
    case_id: str
    iteration: int
    baseline_script: ArtifactRef
    baseline_gds: ArtifactRef
    baseline_drc: ArtifactRef
    rule_deck: ArtifactRef
    connectivity: ArtifactRef | None = None
    trace_context: ExecutionTraceContext | None = None
    logical_evidence_key: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$",
    )
    retry_of_attempt_id: str | None = None
    evaluator_hash: str | None = None
    verification_policy_hash: str | None = None
    canonical_patch_effect_sha256: str | None = None
    execution_protocol_version: str = "p3.2-execution-v1"
    requested_attempt_id: str | None = Field(
        default=None, pattern=r"^attempt_[0-9a-f]{32}$",
    )


class PreparedJob(StrictModel):
    job_id: str
    host_path: Path
    container_path: str = "/job"
    case_id: str
    iteration: int
    manifest_ref: ArtifactRef
    run_id: str | None = None
    attempt_id: str | None = None
    owner_token: str | None = None
    owner_token_hash: str | None = None
    logical_evidence_key: str | None = None
    execution_epoch: str = "epoch_legacy"
    purpose: ExecutionPurpose = ExecutionPurpose.UNKNOWN
    lifecycle_state: AttemptLifecycle = AttemptLifecycle.READY
    trace_context: ExecutionTraceContext | None = None


class DRCRequest(StrictModel):
    job: PreparedJob
    gds_relpath: str
    report_relpath: str
    rule_deck_relpath: str
    timeout_seconds: int


class BackendProbeResult(StrictModel):
    backend: str
    available: bool
    executable_path: str | None = None
    version: str | None = None
    license_ok: bool | None = None
    capabilities: set[BackendCapability] = Field(default_factory=set)
    evidence_refs: list[ArtifactRef] = Field(default_factory=list)


class ConnectivityResult(StrictModel):
    checker: str
    checker_hash: str
    preserved: bool
    broken_net_count: int = 0
    broken_seed_ids: list[str] = Field(default_factory=list)
    top_cell_preserved: bool
    dbu_preserved: bool
    raw_result_ref: ArtifactRef


class GDSSanityResult(StrictModel):
    readable: bool
    top_cell_name: str
    dbu: float
    cell_count: int
    shape_count_by_layer: dict[str, int]
    bbox_dbu: Box
    forbidden_layers: list[str] = Field(default_factory=list)
    passed: bool


class TimingRequest(StrictModel):
    netlist: ArtifactRef
    liberty_files: list[ArtifactRef]
    sdc: ArtifactRef
    spef: ArtifactRef
    top_module: str
    corners: list[str]
    analysis: Literal["MAX", "MIN", "BOTH"]
    baseline_metrics_ref: ArtifactRef | None = None


class TimingResult(StrictModel):
    setup_wns: float | None = None
    setup_tns: float | None = None
    hold_wns: float | None = None
    hold_tns: float | None = None
    setup_violating_endpoints: int | None = None
    hold_violating_endpoints: int | None = None
    affected_path_deltas: list[dict] = Field(default_factory=list)
    reports: list[ArtifactRef] = Field(default_factory=list)
    fidelity: Literal["SIGNOFF", "ACADEMIC_STA", "PROXY"]

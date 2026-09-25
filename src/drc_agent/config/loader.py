from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator

from drc_agent.schemas.common import StrictModel, stable_hash


class FeaturesConfig(StrictModel):
    experience_graph: bool = True
    dynamic_graph: bool = True
    graph_rewire: bool = True
    joint_planning: bool = True
    message_passing: bool = True
    geometry_edges: bool = True
    shared_net_edges: bool = True
    resource_edges: bool = True
    timing_edges: bool = False
    timing_verification: bool = False


class WorkflowConfig(StrictModel):
    max_iterations: int = Field(default=5, ge=1)
    max_windows_per_iteration: int = Field(default=32, ge=1)
    max_no_progress_iterations: int = Field(default=2, ge=1)
    max_parallel_subgraphs: int = Field(default=4, ge=1)
    transaction_mode: Literal["serialized"] = "serialized"
    checkpoint_backend: Literal["sqlite"] = "sqlite"
    fail_closed: bool = True
    whole_run_budget_seconds: float | None = Field(default=None, gt=0.0)
    max_eda_evaluations_per_run: int | None = Field(default=None, ge=1)
    engineering_canary: bool = False
    engineering_target_violation_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_engineering_scope(self):
        if self.engineering_target_violation_ids and not self.engineering_canary:
            raise ValueError("target scope restriction requires explicit engineering_canary")
        if self.engineering_canary and self.max_windows_per_iteration>3:
            raise ValueError("engineering canary is limited to three windows per iteration")
        if self.engineering_canary and self.max_iterations!=1:
            raise ValueError("engineering canary uses exactly one bounded iteration")
        return self


class RegionConfig(StrictModel):
    seed_window_um: float = 2.0
    dbu_per_um: int = 4000
    adaptive_halo: bool = True
    halo_min_dbu: int = 72
    halo_max_dbu: int = 4000
    affinity_max_gap_dbu: int = 400
    max_region_markers: int = 12
    max_region_objects: int = 128
    max_region_span_um: float = 4.0
    max_context_tokens: int = 6000


class AgentGraphConfig(StrictModel):
    max_soft_neighbors_per_relation: int = 4
    soft_edge_threshold: float = 0.35
    strong_edge_threshold: float = 0.65
    proximity_only_score_cap: float = 0.34
    max_active_subgraph_regions: int = 12
    max_llm_view_regions: int = 12
    preserve_maximum_spanning_forest: bool = True


class AgentConfig(StrictModel):
    top_k_candidates_including_noop: int = Field(default=4, ge=1)
    max_geometry_variants_per_hypothesis: int = Field(default=3, ge=1, le=8)
    message_rounds: int = Field(default=2, ge=1)
    max_message_rounds: int = Field(default=3, ge=1)
    max_candidate_distance_dbu: int = Field(default=512, ge=1)
    allowed_distances_dbu: list[int] = Field(default_factory=lambda: [8, 16, 24, 32, 48, 64])


class RepairKernelIntegrationConfig(StrictModel):
    """Formal-runtime routing for the verified repair kernel.

    Disabled by default so historical/legacy development profiles retain their
    existing behavior.  Formal B5/B6 profiles enable ``formal_v1`` explicitly
    and require the immutable integration gate before starting.
    """

    enabled: bool = False
    mode: Literal[
        "disabled", "formal_v1", "test_fake_llm", "development_integration"
    ] = "disabled"
    require_gate: bool = False
    gate_contract: Path = Path("contracts/formal_integration_gate.json")
    legacy_fallback_for_supported: bool = False
    legacy_fallback_for_unsupported: bool = False
    max_plans_per_region: int = Field(default=3, ge=1, le=4)
    max_trajectory_depth: int = Field(default=4, ge=1, le=4)
    max_kernel_live_calls_per_region: int = Field(default=32, ge=0)
    reuse_verified_kernel_evidence: bool = True
    require_trajectory_condensation: bool = True
    supported_families: list[str] = Field(default_factory=lambda: [
        "SPACING", "ENCLOSURE", "VIA_METAL_CONTEXTUAL",
    ])
    # Optional formal-run allowlist.  An empty list preserves the historical
    # behaviour (all reviewed rules in the configured families).  Staged
    # coverage profiles use this to expose only rules with current live,
    # strict-clean qualification evidence.
    supported_rule_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_routing(self) -> "RepairKernelIntegrationConfig":
        if not self.enabled and self.mode != "disabled":
            raise ValueError("repair kernel mode requires enabled=true")
        if self.enabled and self.mode == "disabled":
            raise ValueError("enabled repair kernel requires a non-disabled mode")
        if self.legacy_fallback_for_supported:
            raise ValueError(
                "legacy fallback for supported repair families is forbidden"
            )
        return self


class RepairProgrammingConfig(StrictModel):
    sound_simple_fast_path: bool = True
    enabled: bool = True
    max_revisions: int = Field(default=2, ge=0, le=2)
    max_regions_per_subgraph: int = Field(default=2, ge=0)
    max_calls_per_subgraph: int = Field(default=6, ge=0)
    require_rule_witness: bool = True
    escalate_contextual: bool = True
    escalate_unsupported: bool = True
    escalate_after_no_progress: bool = True


class CandidateEvidenceConfig(StrictModel):
    sandbox_enabled: bool = True
    sandbox_top_m_per_view: int = Field(default=8, ge=0)
    sandbox_max_jobs_per_iteration: int = Field(default=32, ge=0)
    sandbox_connectivity_for_risky_actions: bool = True
    joint_sandbox_enabled: bool = True
    joint_sandbox_max_bundle_size: int = Field(default=5, ge=1)
    joint_sandbox_max_rounds: int = Field(default=5, ge=0)
    cache_key_version: str = "sandbox-v2"


class CandidateGraphConfig(StrictModel):
    pairwise_preview_version: str = "gc-preview-v2"
    interaction_margin_dbu: int = Field(default=72, ge=0)
    fail_on_incomplete_audit: bool = True


class TransactionConfig(StrictModel):
    max_non_noop_actions_per_batch: int = Field(default=5, ge=1)
    max_modified_objects_per_batch: int = Field(default=16, ge=1)
    replan_after_each_commit: bool = True


class AttributionConfig(StrictModel):
    enabled: bool = True
    delta_debug_enabled: bool = False
    max_replays_per_failure: int = Field(default=4, ge=0)


class CoordinatorConfig(StrictModel):
    solver: Literal["cp_sat"] = "cp_sat"
    time_limit_seconds: float = 5.0
    oversized_time_limit_seconds: float = 30.0
    num_search_workers: int = 1
    random_seed: int = 0
    log_search_progress: bool = False
    allow_unverified_exploration: bool = True
    max_unverified_non_noop_per_subgraph: int = Field(default=1, ge=0)


class AcceptanceConfig(StrictModel):
    require_connectivity: bool = True
    require_drc_progress: bool = True
    reject_new_violations: bool = True
    keep_best_verified: bool = True


class RetrievalConfig(StrictModel):
    seed_k: int = 8
    rrf_k0: int = 60
    min_verified_success: int = 2
    min_contrastive_failure: int = 2
    min_alternative_action: int = 1
    min_joint_episode_if_available: int = 1
    expansion_hops: int = 1
    max_episodes: int = 24
    max_evidence_nodes: int = 60
    max_context_tokens: int = 8000
    embedding_enabled: bool = False


class LLMConfig(StrictModel):
    enabled: bool = False
    provider: Literal["openrouter", "siliconflow", "fake"] = "openrouter"
    model: str = ""
    base_url: str | None = None
    api_key_env: str | None = None
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    timeout_seconds: float = Field(default=120.0, gt=0.0)
    queue_timeout_seconds: float = Field(default=600.0, gt=0.0)
    request_timeout_seconds: float = Field(default=120.0, gt=0.0)
    max_retries: int = Field(default=2, ge=0, le=8)
    retry_backoff_base_seconds: float = Field(default=1.0, ge=0.0)
    retry_backoff_max_seconds: float = Field(default=8.0, ge=0.0)
    retry_jitter_fraction: float = Field(default=0.2, ge=0.0, le=1.0)
    circuit_breaker_threshold: int = Field(default=3, ge=1)
    circuit_breaker_cooldown_seconds: float = Field(default=60.0, ge=0.0)
    max_concurrent_requests: int = Field(default=4, ge=1)
    max_http_attempts_per_run: int | None = Field(default=None, ge=1)
    seed: int | None = None
    effort: str | None = None
    enable_thinking: bool | None = None
    thinking_budget: int | None = Field(default=None, ge=128, le=32768)
    allow_fallback: bool = False
    max_output_tokens: int = Field(default=3000, ge=128)
    prompt_version: str = "region-agent-v1"
    skill_version: str = "drc-layout-ir-v1"
    evodrc_skill_enabled: bool = True
    evodrc_skill_root: Path = Path(
        "benchmarks/EvoDRC/agent/knowledge/cla"
    )

    @model_validator(mode="after")
    def validate_enabled(self) -> "LLMConfig":
        if not self.enabled:
            return self
        if not self.model.strip():
            raise ValueError("llm.model is required when llm.enabled=true")
        if self.provider != "fake" and not self.api_key_env:
            raise ValueError("llm.api_key_env is required for a real provider")
        if self.allow_fallback:
            raise ValueError("paid research runs must be fail-closed; llm.allow_fallback must be false")
        return self


class BackendConfig(StrictModel):
    type: Literal["klayout_dac26"] = "klayout_dac26"
    image: str = "drc-benchmark-repair:latest"
    stage_root: Path = Path("/tmp/2510878-drc-agent")
    layout_timeout_seconds: int = 300
    drc_timeout_seconds: int = 900
    connectivity_timeout_seconds: int = 300
    keep_failed_workspace: bool = True
    artifact_retention_mode: Literal["full", "compact"] = "full"
    container_workspace_mode: Literal["bind", "copy"] = "bind"
    manufacturing_grid_dbu: int = Field(default=4, ge=1)
    max_concurrent_eda_jobs: int = Field(default=2, ge=1)


class StoragePathConfig(StrictModel):
    role: Literal["host_root", "daemon_root", "stage", "results"]
    path: Path
    required: bool = True


class StorageHealthConfig(StrictModel):
    targets: list[StoragePathConfig]
    pause_free_bytes: int = Field(default=30 * 1024 ** 3, ge=0)
    resume_free_bytes: int = Field(default=50 * 1024 ** 3, gt=0)
    worst_concurrent_batch_bytes: int = Field(default=0, ge=0)
    pause_free_inode_fraction: float = Field(default=0.10, ge=0, lt=1)
    resume_free_inode_fraction: float = Field(default=0.15, gt=0, le=1)
    sample_interval_seconds: int = Field(default=30, ge=30, le=60)

    @model_validator(mode="after")
    def validate_storage_watermarks(self) -> "StorageHealthConfig":
        roles = [target.role for target in self.targets]
        required_roles = {"host_root", "daemon_root", "stage", "results"}
        if set(roles) != required_roles or len(roles) != len(required_roles):
            raise ValueError(
                "storage targets require exactly one host_root, daemon_root, "
                "stage, and results path"
            )
        effective_pause = max(
            self.pause_free_bytes,
            2 * self.worst_concurrent_batch_bytes,
        )
        if self.resume_free_bytes <= effective_pause:
            raise ValueError(
                "storage resume bytes must exceed the effective pause bytes"
            )
        if self.resume_free_inode_fraction <= self.pause_free_inode_fraction:
            raise ValueError(
                "storage resume inode fraction must exceed pause fraction"
            )
        return self


class ResourceControlConfig(StrictModel):
    """Operational cross-process controls, excluded from scientific hashes."""

    protocol_version: Literal["p5-global-resource-v1"] = (
        "p5-global-resource-v1"
    )
    enabled: bool = True
    coordination_root: Path
    global_http_limit: Literal[8] = 8
    global_eda_limit: Literal[4] = 4
    queue_timeout_seconds: float = Field(default=600.0, gt=0)
    poll_interval_seconds: float = Field(default=0.05, gt=0, le=5)
    storage: StorageHealthConfig

    @property
    def operational_hash(self) -> str:
        return stable_hash(self.model_dump(mode="json"))

    def environment(self) -> dict[str, str]:
        """Return explicit child-process settings for a formal launcher."""
        if not self.enabled:
            return {}
        targets = [
            {
                "role": target.role,
                "path": str(target.path.resolve()),
                "required": target.required,
            }
            for target in self.storage.targets
        ]
        return {
            "DRC_GLOBAL_RESOURCE_ROOT": str(
                self.coordination_root.resolve()
            ),
            "DRC_GLOBAL_HTTP_LIMIT": str(self.global_http_limit),
            "DRC_GLOBAL_EDA_LIMIT": str(self.global_eda_limit),
            "DRC_GLOBAL_RESOURCE_QUEUE_TIMEOUT_SECONDS": str(
                self.queue_timeout_seconds
            ),
            "DRC_GLOBAL_RESOURCE_POLL_INTERVAL_SECONDS": str(
                self.poll_interval_seconds
            ),
            "DRC_STORAGE_HEALTH_ROOT": str(
                self.coordination_root.resolve()
            ),
            "DRC_STORAGE_HEALTH_PATHS_JSON": json.dumps(
                targets, sort_keys=True, separators=(",", ":"),
            ),
            "DRC_STORAGE_PAUSE_FREE_BYTES": str(
                self.storage.pause_free_bytes
            ),
            "DRC_STORAGE_RESUME_FREE_BYTES": str(
                self.storage.resume_free_bytes
            ),
            "DRC_STORAGE_WORST_BATCH_BYTES": str(
                self.storage.worst_concurrent_batch_bytes
            ),
            "DRC_STORAGE_PAUSE_FREE_INODE_FRACTION": str(
                self.storage.pause_free_inode_fraction
            ),
            "DRC_STORAGE_RESUME_FREE_INODE_FRACTION": str(
                self.storage.resume_free_inode_fraction
            ),
        }


class AppConfig(StrictModel):
    schema_version: str = "1.0"
    project_root: Path
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    region: RegionConfig = Field(default_factory=RegionConfig)
    agent_graph: AgentGraphConfig = Field(default_factory=AgentGraphConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    coordinator: CoordinatorConfig = Field(default_factory=CoordinatorConfig)
    acceptance: AcceptanceConfig = Field(default_factory=AcceptanceConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    backend: BackendConfig = Field(default_factory=BackendConfig)
    repair_programming: RepairProgrammingConfig = Field(default_factory=RepairProgrammingConfig)
    candidate_evidence: CandidateEvidenceConfig = Field(default_factory=CandidateEvidenceConfig)
    candidate_graph: CandidateGraphConfig = Field(default_factory=CandidateGraphConfig)
    transaction: TransactionConfig = Field(default_factory=TransactionConfig)
    attribution: AttributionConfig = Field(default_factory=AttributionConfig)
    repair_kernel_integration: RepairKernelIntegrationConfig = Field(
        default_factory=RepairKernelIntegrationConfig
    )
    # Operational admission is recorded by the launcher/resource policy.  It
    # is excluded here so merely upgrading the code cannot change historical
    # resolved/scientific hashes or invalidate old resume checkpoints.
    resource_control: ResourceControlConfig | None = Field(
        default=None, exclude=True,
    )

    @model_validator(mode="after")
    def validate_resource_control(self) -> "AppConfig":
        if self.resource_control is None or not self.resource_control.enabled:
            return self
        if self.llm.max_concurrent_requests > 4:
            raise ValueError(
                "P5 run-local HTTP concurrency cannot exceed 4"
            )
        if self.backend.max_concurrent_eda_jobs > 2:
            raise ValueError(
                "P5 run-local EDA concurrency cannot exceed 2"
            )
        return self

    @property
    def content_hash(self) -> str:
        return stable_hash(self.model_dump(mode="json"))


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path, overlays: list[Path] | None = None,
                overrides: dict[str, Any] | None = None) -> AppConfig:
    with path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    for overlay in overlays or []:
        with overlay.open(encoding="utf-8") as stream:
            raw = _deep_merge(raw, yaml.safe_load(stream) or {})
    if overrides:
        raw = _deep_merge(raw, overrides)
    return AppConfig.model_validate(raw)

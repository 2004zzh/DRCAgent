from __future__ import annotations

import difflib
import json
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from pydantic import Field

from drc_agent.backends.execution import (
    EXECUTION_PROTOCOL_VERSION,
    ExecutionControl,
    ExecutionCancelled,
    JobPreparationError,
    append_jsonl,
    attempt_event,
    build_logical_evidence_key,
    new_attempt_id,
)
from drc_agent.backends.validation_budget import claim_from_environment, finish_claim
from drc_agent.backends.dac26 import DAC26CaseMeta, DAC26ReportAdapter
from drc_agent.backends.klayout import KLayoutBackend
from drc_agent.infrastructure.storage_health import (
    record_storage_incident_from_environment,
    storage_pause_from_exception,
)
from drc_agent.evaluation.matching import conservative_violation_delta_evidence
from drc_agent.config.loader import AcceptanceConfig, BackendConfig
from drc_agent.evaluation.policy import evaluate_acceptance
from drc_agent.observability.metrics import drc_statistics
from drc_agent.patching.compiler import apply_patch_plan
from drc_agent.regions.parser import SourceObjectMapper
from drc_agent.schemas.action import (
    PatchPlan, Transaction, TransactionStatus, VerificationOutcome,
    VerificationResult,
)
from drc_agent.reliability import (
    FailureCode, InfrastructureFailure, InfrastructurePause,
)
from drc_agent.schemas.common import (
    ArtifactRef, StrictModel, file_sha256, stable_hash, utc_now,
)
from drc_agent.schemas.state import LayoutObject
from drc_agent.schemas.tools import (
    AttemptLifecycle,
    AttemptTerminalStatus,
    EvidenceValidity,
    ExecutionPurpose,
    ExecutionTraceContext,
    PrepareJobRequest,
    ToolResult,
)
from drc_agent.schemas.workflow import DesignSnapshotRef
from drc_agent.utils.artifacts import ArtifactStore


class _AttemptLedger:
    """Exactly-once terminal accounting for one physical attempt."""

    def __init__(
        self,
        *,
        path: Path,
        attempt_id: str,
        logical_evidence_key: str,
        trace: ExecutionTraceContext,
        patch_plan_sha256: str,
        physical_effect_sha256: str | None,
        retry_of: str | None,
    ):
        self.path = path
        self.attempt_id = attempt_id
        self.logical_evidence_key = logical_evidence_key
        self.trace = trace
        self.current_stage = "ATTEMPT_REGISTERED"
        self.failure_domain = "EXECUTION"
        self.terminal_written = False
        self.validation_reservation = claim_from_environment(
            "eda", purpose=trace.purpose.value, run_id=trace.formal_run_id,
            attempt_id=attempt_id, formal_iteration=trace.formal_iteration,
        )
        _ACTIVE_LEDGER.set(self)
        append_jsonl(self.path, attempt_event(
            "ATTEMPT_REGISTERED",
            attempt_id=attempt_id,
            logical_evidence_key=logical_evidence_key,
            trace_context=trace.model_dump(mode="json"),
            patch_plan_sha256=patch_plan_sha256,
            physical_effect_sha256=physical_effect_sha256,
            retry_of=retry_of,
            resource_queue_wait_seconds=(
                self.validation_reservation.queue_wait_seconds
                if self.validation_reservation is not None else 0.0
            ),
        ))

    def set_stage(self, stage: str, *, failure_domain: str = "EXECUTION") -> None:
        self.current_stage = stage
        self.failure_domain = failure_domain

    def phase(self, event: str, **details: Any) -> None:
        append_jsonl(self.path, attempt_event(
            event,
            attempt_id=self.attempt_id,
            logical_evidence_key=self.logical_evidence_key,
            **details,
        ))

    def terminal(
        self,
        status: AttemptTerminalStatus,
        **details: Any,
    ) -> None:
        if self.terminal_written:
            return
        try:
            append_jsonl(self.path, attempt_event(
                "ATTEMPT_TERMINAL",
                attempt_id=self.attempt_id,
                terminal_status=status.value,
                logical_evidence_key=self.logical_evidence_key,
                **details,
            ))
        finally:
            # An ENOSPC while publishing the terminal record must not retain
            # a global EDA lease.  The failed write remains a typed storage
            # stop at the decorator boundary.
            self.terminal_written = True
            finish_claim(
                self.validation_reservation, status.value, str(self.path),
            )


_ACTIVE_LEDGER: ContextVar[_AttemptLedger | None] = ContextVar(
    "transaction_attempt_ledger", default=None,
)


def _account_physical_attempts(function: Callable) -> Callable:
    """Guarantee a terminal record even when an unexpected stage raises."""

    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        token = _ACTIVE_LEDGER.set(None)
        try:
            result = function(*args, **kwargs)
            ledger = _ACTIVE_LEDGER.get()
            if ledger is not None and not ledger.terminal_written:
                ledger.terminal(
                    AttemptTerminalStatus.EXECUTION_FAILURE,
                    failure_domain="INFRASTRUCTURE",
                    failure_code="TERMINAL_ACCOUNTING_MISSING",
                    failure_stage=ledger.current_stage,
                )
                raise RuntimeError("TERMINAL_ACCOUNTING_MISSING")
            return result
        except BaseException as exc:
            normalized: BaseException = exc
            if not isinstance(exc, InfrastructureFailure):
                storage_pause = storage_pause_from_exception(
                    exc, stage=(
                        _ACTIVE_LEDGER.get().current_stage
                        if _ACTIVE_LEDGER.get() is not None
                        else "TRANSACTION"
                    ),
                    provider="TransactionExecutor",
                    model="host-storage",
                )
                if storage_pause is not None:
                    storage_pause.health_snapshot = (
                        record_storage_incident_from_environment(
                            code=(
                                storage_pause.original_exception_type
                                or "STORAGE_EXHAUSTED"
                            ),
                            stage=storage_pause.failure_stage,
                        ) or {}
                    )
                    normalized = storage_pause
            ledger = _ACTIVE_LEDGER.get()
            if ledger is not None and not ledger.terminal_written:
                cancelled = isinstance(
                    normalized, (ExecutionCancelled, KeyboardInterrupt),
                )
                if isinstance(normalized, InfrastructureFailure):
                    details = {
                        "failure_domain": "INFRASTRUCTURE",
                        "failure_code": normalized.failure_code.value,
                        "failure_stage": normalized.failure_stage,
                        "retryable": normalized.retryable,
                    }
                else:
                    details = {
                        "failure_domain": ledger.failure_domain,
                        "failure_code": (
                            "CANCELLED" if cancelled
                            else type(normalized).__name__
                        ),
                        "failure_stage": ledger.current_stage,
                    }
                ledger.terminal(
                    AttemptTerminalStatus.CANCELLED if cancelled
                    else AttemptTerminalStatus.EXECUTION_FAILURE,
                    **details,
                )
            if normalized is not exc:
                raise normalized from exc
            raise
        finally:
            _ACTIVE_LEDGER.reset(token)

    return wrapped


class TransactionExecutionResult(StrictModel):
    transaction: Transaction
    verification: VerificationResult
    accepted: bool
    decision_reasons: list[str] = Field(default_factory=list)
    attempted_snapshot: DesignSnapshotRef
    committed_snapshot: DesignSnapshotRef
    attempted_total_drv: int | None
    committed_total_drv: int
    tool_runtime_seconds: float
    rollback_reason: str | None = None
    attempt_id: str | None = None
    logical_evidence_key: str | None = None


def _post_patch_object_matches(
    original: LayoutObject | None,
    patched_objects: list[LayoutObject],
    expected_hash: str,
) -> bool:
    if original is None:
        return False
    anchored = [
        item for item in patched_objects
        if item.source_anchor_id == original.source_anchor_id
    ]
    if any(item.geometry_hash == expected_hash for item in anchored):
        return True
    # Generated specialization declarations can shift subsequent source line
    # numbers. Source variables remain stable within the same source cell and
    # layer; require a unique match and retain the exact geometry-hash check.
    if not original.source_variable or not original.source_cell:
        return False
    stable = [
        item for item in patched_objects
        if item.source_variable == original.source_variable
        and item.source_cell == original.source_cell
        and item.layer == original.layer
    ]
    return len(stable) == 1 and stable[0].geometry_hash == expected_hash


class TransactionExecutor:
    """Owns the only mutation path from a combined PatchPlan to a verified snapshot."""

    def __init__(
        self, *, backend: KLayoutBackend, adapter: DAC26ReportAdapter,
        store: ArtifactStore, acceptance: AcceptanceConfig,
        backend_config: BackendConfig, benchmark_paths: dict[str, Path],
    ):
        self.backend = backend
        self.adapter = adapter
        self.store = store
        self.acceptance = acceptance
        self.backend_config = backend_config
        self.paths = benchmark_paths

    @staticmethod
    def _ref(path: Path, producer: str, media_type: str) -> ArtifactRef:
        return ArtifactRef.from_path(
            path, producer=producer, media_type=media_type,
        )

    def _copy_snapshot(
        self, snapshot: DesignSnapshotRef, prefix: str, case_id: str,
    ) -> DesignSnapshotRef:
        script = self.store.copy(
            Path(snapshot.script_ref.path), f"{prefix}/{case_id}.py",
            producer="transaction.snapshot", media_type="text/x-python",
        )
        gds = self.store.copy(
            Path(snapshot.gds_ref.path), f"{prefix}/{case_id}.gds",
            producer="transaction.snapshot", media_type="application/gds",
        ) if snapshot.gds_ref else None
        drc = self.store.copy(
            Path(snapshot.drc_ref.path), f"{prefix}/{case_id}.drc.json",
            producer="transaction.snapshot", media_type="application/json",
        ) if snapshot.drc_ref else None
        lineage = self.store.copy(
            Path(snapshot.lineage_receipt_ref.path), f"{prefix}/{case_id}.lineage.json",
            producer="transaction.snapshot", media_type="application/json",
        ) if snapshot.lineage_receipt_ref else None
        return DesignSnapshotRef(
            snapshot_id=snapshot.snapshot_id, script_ref=script,
            lineage_receipt_ref=lineage,
            gds_ref=gds, drc_ref=drc,
            connectivity_ref=snapshot.connectivity_ref,
            parent_snapshot_id=snapshot.parent_snapshot_id,
            parent_script_ref=snapshot.parent_script_ref,
            provenance_run_id=snapshot.provenance_run_id,
            provenance_relation_source=(
                snapshot.provenance_relation_source
            ),
            score=snapshot.score,
        )

    @_account_physical_attempts
    def execute(
        self, *, run_id: str, case_id: str, iteration: int,
        baseline_snapshot: DesignSnapshotRef,
        current_snapshot: DesignSnapshotRef,
        patch_plan: PatchPlan, source_map: list[LayoutObject],
        bundle_ids: list[str],
        artifact_prefix: str | None = None,
        job_run_id: str | None = None,
        legacy_artifact_prefix: str | None = None,
        trace_context: ExecutionTraceContext | None = None,
        physical_effect_fingerprint: str | None = None,
        retry_of_attempt_id: str | None = None,
        execution_control: ExecutionControl | None = None,
    ) -> TransactionExecutionResult:
        base_prefix = artifact_prefix or f"iterations/iter_{iteration:04d}"
        control = execution_control or ExecutionControl()
        trace = trace_context or ExecutionTraceContext(
            formal_run_id=run_id,
            execution_epoch="epoch_legacy_call",
            formal_iteration=iteration,
            parent_snapshot_id=current_snapshot.snapshot_id,
            purpose=ExecutionPurpose.MASTER,
        )
        patch_plan_sha256 = stable_hash(patch_plan.model_dump(mode="json"))
        verification_policy_hash = stable_hash(
            self.acceptance.model_dump(mode="json")
        )
        backend_image = str(getattr(self.backend, "image", "test-backend"))
        try:
            backend_image_digest = (
                self.backend.image_digest()
                if hasattr(self.backend, "image_digest")
                else stable_hash(backend_image)
            )
        except RuntimeError as exc:
            raise InfrastructureFailure(
                str(exc),
                failure_code=FailureCode.EDA_UNAVAILABLE,
                retryable=False,
                provider="KLayoutBackend",
                model=backend_image,
                original_exception_type=type(exc).__name__,
                failure_stage="EDA_PREFLIGHT",
            ) from exc
        logical_evidence_key = build_logical_evidence_key(
            parent_script_sha256=current_snapshot.script_ref.sha256,
            parent_gds_sha256=(
                current_snapshot.gds_ref.sha256 if current_snapshot.gds_ref else None
            ),
            parent_drc_sha256=(
                current_snapshot.drc_ref.sha256 if current_snapshot.drc_ref else None
            ),
            connectivity_reference_sha256=(
                current_snapshot.connectivity_ref.sha256
                if current_snapshot.connectivity_ref else None
            ),
            rule_deck_sha256=file_sha256(self.paths["rule_deck"]),
            evaluator_hash=getattr(self.adapter, "evaluator_hash", None),
            sanity_helper_sha256=(
                file_sha256(self.paths["sanity_helper"])
                if self.paths.get("sanity_helper") else None
            ),
            connectivity_helper_sha256=(
                file_sha256(self.paths["connectivity_helper"])
                if self.paths.get("connectivity_helper") else None
            ),
            backend_image=backend_image,
            backend_image_digest=backend_image_digest,
            verification_policy_hash=verification_policy_hash,
            patch_plan_sha256=patch_plan_sha256,
            physical_effect_sha256=physical_effect_fingerprint,
            target_violation_id=trace.target_violation_id,
        )
        current_drc = Path(current_snapshot.drc_ref.path)
        before = drc_statistics(current_drc)
        request = PrepareJobRequest(
            run_id=job_run_id or run_id, case_id=case_id, iteration=iteration,
            baseline_script=current_snapshot.script_ref,
            baseline_gds=current_snapshot.gds_ref,
            baseline_drc=current_snapshot.drc_ref,
            rule_deck=self._ref(
                self.paths["rule_deck"], "initialize_case",
                "application/x-klayout-drc",
            ),
            connectivity=self._ref(
                self.paths["connectivity"], "initialize_case",
                "application/json",
            ),
            trace_context=trace,
            logical_evidence_key=logical_evidence_key,
            retry_of_attempt_id=retry_of_attempt_id,
            evaluator_hash=getattr(self.adapter, "evaluator_hash", None),
            verification_policy_hash=verification_policy_hash,
            canonical_patch_effect_sha256=physical_effect_fingerprint,
            execution_protocol_version=EXECUTION_PROTOCOL_VERSION,
            requested_attempt_id=new_attempt_id(),
        )
        attempt_index = self.store.root / "attempt_index.jsonl"
        prepare_number = 0
        while True:
            assert request.requested_attempt_id is not None
            ledger = _AttemptLedger(
                path=attempt_index,
                attempt_id=request.requested_attempt_id,
                logical_evidence_key=logical_evidence_key,
                trace=trace,
                patch_plan_sha256=patch_plan_sha256,
                physical_effect_sha256=physical_effect_fingerprint,
                retry_of=request.retry_of_attempt_id,
            )
            ledger.set_stage("PREPARE_JOB", failure_domain="INFRASTRUCTURE")
            _ACTIVE_LEDGER.set(ledger)
            try:
                job = self.backend.prepare_job(request)
                break
            except JobPreparationError as exc:
                ledger.terminal(
                    AttemptTerminalStatus.EXECUTION_FAILURE,
                    failure_domain="INFRASTRUCTURE",
                    failure_code=exc.failure_code,
                    failure_stage="PREPARE_JOB",
                    workspace=str(exc.workspace),
                    retryable=exc.retryable,
                )
                if exc.retryable and prepare_number == 0:
                    prepare_number += 1
                    request = request.model_copy(update={
                        "requested_attempt_id": new_attempt_id(),
                        "retry_of_attempt_id": exc.attempt_id,
                    })
                    continue
                budget_exhausted = (
                    exc.failure_code == "PHYSICAL_ATTEMPT_BUDGET_EXHAUSTED"
                )
                storage_exhausted = (
                    exc.failure_code == FailureCode.STORAGE_EXHAUSTED.value
                )
                if storage_exhausted:
                    raise InfrastructurePause(
                        str(exc),
                        failure_code=FailureCode.STORAGE_EXHAUSTED,
                        retryable=True,
                        provider="KLayoutBackend",
                        model=backend_image,
                        original_exception_type=(
                            type(exc.__cause__).__name__
                            if exc.__cause__ is not None else type(exc).__name__
                        ),
                        failure_stage="PREPARE_JOB",
                        pause_scope="SHARED_STORAGE",
                    ) from exc
                raise InfrastructureFailure(
                    str(exc),
                    failure_code=(
                        FailureCode.BUDGET_EXHAUSTED
                        if budget_exhausted
                        else FailureCode.WORKSPACE_OWNERSHIP
                    ),
                    retryable=exc.retryable,
                    provider="KLayoutBackend",
                    model=backend_image,
                    original_exception_type=type(exc).__name__,
                    failure_stage=(
                        "EDA_ATTEMPT_BUDGET"
                        if budget_exhausted else "PREPARE_JOB"
                    ),
                ) from exc
            except InfrastructureFailure:
                raise
            except Exception as exc:
                storage_pause = storage_pause_from_exception(
                    exc, stage="PREPARE_JOB", provider="KLayoutBackend",
                    model=backend_image,
                )
                if storage_pause is not None:
                    storage_pause.health_snapshot = (
                        record_storage_incident_from_environment(
                            code=(
                                storage_pause.original_exception_type
                                or "STORAGE_EXHAUSTED"
                            ),
                            stage="PREPARE_JOB",
                        ) or {}
                    )
                    raise storage_pause from exc
                raise InfrastructureFailure(
                    f"WORKSPACE_PREPARATION_FAILED: {exc}",
                    failure_code=FailureCode.WORKSPACE_OWNERSHIP,
                    retryable=False,
                    provider="KLayoutBackend",
                    model=backend_image,
                    original_exception_type=type(exc).__name__,
                    failure_stage="PREPARE_JOB",
                ) from exc
        attempt_id = request.requested_attempt_id
        if isinstance(self.backend, KLayoutBackend):
            if job.attempt_id != attempt_id:
                raise InfrastructureFailure(
                    "WORKSPACE_OWNERSHIP_INVALID: backend changed attempt identity",
                    failure_code=FailureCode.WORKSPACE_OWNERSHIP,
                    retryable=False,
                    provider="KLayoutBackend",
                    model=backend_image,
                    failure_stage="PREPARE_JOB",
                )
        elif not job.attempt_id:
            # Legacy test backends do not yet own physical workspaces.  Keep
            # their adapter behavior while making persisted evidence unique.
            job.attempt_id = attempt_id
            job.logical_evidence_key = logical_evidence_key
            job.execution_epoch = trace.execution_epoch
            job.trace_context = trace
        ledger.phase(
            "WORKSPACE_READY",
            workspace=str(job.host_path),
            retry_of=request.retry_of_attempt_id,
        )
        prefix = f"{base_prefix}/attempts/{attempt_id}"
        patch_ref = self.store.write_json(
            f"{prefix}/transaction/patch_plan.json", patch_plan,
            producer="prepare_transaction", schema_name="PatchPlan",
            schema_version="3.2",
        )
        transaction = Transaction(
            transaction_id="transaction_" + stable_hash([
                run_id, attempt_id, patch_plan_sha256,
            ])[:20],
            run_id=run_id, case_id=case_id, iteration=iteration,
            base_snapshot_id=current_snapshot.snapshot_id,
            bundle_ids=sorted(bundle_ids), patch_plan_ref=patch_ref,
            job_dir=str(job.host_path), status=TransactionStatus.PREPARED,
            started_at=utc_now(),
            execution_protocol_version=EXECUTION_PROTOCOL_VERSION,
            attempt_id=attempt_id,
            logical_evidence_key=logical_evidence_key,
            execution_epoch=trace.execution_epoch,
            purpose=trace.purpose.value,
        )
        attempted_snapshot_id = "snapshot_" + stable_hash([
            transaction.transaction_id, "attempted",
        ])[:20]
        ledger.phase(
            "PATCH_REGISTERED",
            patch_plan_ref=patch_ref.model_dump(mode="json"),
        )
        transaction.job_dir = str(job.host_path)
        transaction.status = TransactionStatus.RUNNING
        patched_script = job.host_path / "layout_script" / f"{case_id}.py"
        ledger.set_stage("APPLY_PATCH", failure_domain="CANDIDATE")
        lineage_receipt_ref = None
        try:
            control.check()
            lineage = apply_patch_plan(
                patch_plan, Path(current_snapshot.script_ref.path),
                patched_script, source_map,
                parent_snapshot_id=current_snapshot.snapshot_id,
                child_snapshot_id=attempted_snapshot_id,
                provenance_run_id=run_id,
            )
            if lineage is not None:
                lineage_receipt_ref = self.store.write_json(
                    f"{prefix}/attempted/{case_id}.lineage.json", lineage,
                    producer="apply_patch_plan", schema_name="CompilerLineageReceipt",
                )
            original_by_id = {item.object_id: item for item in source_map}
            patched_objects = SourceObjectMapper().build_map(patched_script)
            for object_id, expected_hash in patch_plan.expected_object_after_hashes.items():
                original = original_by_id.get(object_id)
                if not _post_patch_object_matches(
                    original, patched_objects, expected_hash,
                ):
                    raise ValueError(
                        f"POST_PATCH_GEOMETRY_HASH_MISMATCH: {object_id}"
                    )
        except Exception as exc:
            transaction.status = TransactionStatus.FAILED
            transaction.finished_at = utc_now()
            transaction.terminal_status = AttemptTerminalStatus.EXECUTION_FAILURE
            transaction.rollback_reason = "PATCH_APPLICATION_FAILED"
            self.store.write_json(
                f"{prefix}/transaction/transaction.json", transaction,
                producer="transaction.patch_failure", schema_name="Transaction",
                schema_version="3.2",
            )
            if isinstance(self.backend, KLayoutBackend):
                self.backend.mark_job_state(
                    job, AttemptLifecycle.TERMINAL,
                    details={
                        "terminal_status": (
                            AttemptTerminalStatus.EXECUTION_FAILURE.value
                        ),
                        "failure_code": "PATCH_APPLICATION_FAILED",
                    },
                )
            raise ValueError(f"PATCH_APPLICATION_FAILED: {exc}") from exc
        before_lines = Path(current_snapshot.script_ref.path).read_text(
            encoding="utf-8",
        ).splitlines(keepends=True)
        after_lines = patched_script.read_text(encoding="utf-8").splitlines(keepends=True)
        diff_path = self.store.root / prefix / "transaction" / "combined.diff"
        diff_path.parent.mkdir(parents=True, exist_ok=True)
        diff_path.write_text("".join(difflib.unified_diff(
            before_lines, after_lines,
            fromfile=f"committed/{case_id}.py",
            tofile=f"attempted/{case_id}.py",
        )), encoding="utf-8")
        self.store.copy(
            patched_script, f"{prefix}/attempted/{case_id}.py",
            producer="apply_patch_plan", media_type="text/x-python",
        )

        tools: list[ToolResult] = []
        failure: tuple[VerificationOutcome, str] | None = None
        failure_stage: str | None = None
        script_validity = EvidenceValidity.NOT_EVALUATED
        layout_validity = EvidenceValidity.NOT_EVALUATED
        drc_validity = EvidenceValidity.NOT_EVALUATED
        sanity_validity = EvidenceValidity.NOT_EVALUATED
        connectivity_validity = EvidenceValidity.NOT_EVALUATED
        backend_control = (
            {"control": control} if isinstance(self.backend, KLayoutBackend) else {}
        )
        ledger.set_stage("VALIDATE_SCRIPT")
        validate = self.backend.validate_script(
            job, self.backend_config.layout_timeout_seconds,
            **backend_control,
        )
        tools.append(validate)
        if validate.status != "SUCCESS":
            script_validity = EvidenceValidity.ERROR
            failure_stage = "VALIDATE_SCRIPT"
            failure = (
                VerificationOutcome.CANCELLED
                if validate.status == "CANCELLED"
                else VerificationOutcome.SYNTAX_FAILURE,
                "CANCELLED" if validate.status == "CANCELLED"
                else "SCRIPT_VALIDATION_FAILED",
            )
        else:
            script_validity = EvidenceValidity.VALID
        if failure is None:
            ledger.set_stage("GENERATE_LAYOUT")
            layout = self.backend.run_layout_generation(
                job, self.backend_config.layout_timeout_seconds,
                **backend_control,
            )
            tools.append(layout)
            if layout.status != "SUCCESS":
                layout_validity = EvidenceValidity.ERROR
                failure_stage = "GENERATE_LAYOUT"
                failure = (
                    VerificationOutcome.CANCELLED
                    if layout.status == "CANCELLED"
                    else VerificationOutcome.LAYOUT_GENERATION_FAILURE,
                    "CANCELLED" if layout.status == "CANCELLED"
                    else "LAYOUT_GENERATION_FAILED",
                )
            else:
                layout_validity = EvidenceValidity.VALID
                ledger.phase("LAYOUT_GENERATED")
        report_path = job.host_path / "report" / f"{case_id}.lyrpt"
        if failure is None:
            ledger.set_stage("RUN_DRC")
            drc_tool = self.backend.run_drc(
                job, self.backend_config.drc_timeout_seconds,
                **backend_control,
            )
            tools.append(drc_tool)
            if drc_tool.status != "SUCCESS":
                drc_validity = EvidenceValidity.ERROR
                failure_stage = "RUN_DRC"
                failure = (
                    VerificationOutcome.CANCELLED
                    if drc_tool.status == "CANCELLED"
                    else VerificationOutcome.DRC_EXECUTION_FAILURE,
                    "CANCELLED" if drc_tool.status == "CANCELLED"
                    else "DRC_EXECUTION_FAILED",
                )
        if failure is None:
            ledger.set_stage("VALIDATE_DRC_REPORT")
            try:
                self.backend.validate_lyrpt(report_path)
            except ValueError:
                retry = self.backend.run_drc_retry(
                    job, self.backend_config.drc_timeout_seconds,
                    **backend_control,
                )
                tools.append(retry)
                retry_path = job.host_path / "report" / f"{case_id}.retry.lyrpt"
                try:
                    if retry.status != "SUCCESS":
                        raise ValueError("retry execution failed")
                    self.backend.validate_lyrpt(retry_path)
                    report_path = retry_path
                except ValueError:
                    drc_validity = EvidenceValidity.ERROR
                    failure_stage = "VALIDATE_DRC_REPORT"
                    failure = (
                        VerificationOutcome.DRC_EXECUTION_FAILURE,
                        "INVALID_LYRPT_XML_AFTER_RETRY",
                    )

        attempted_drc_path = job.host_path / "report" / f"{case_id}.drc.json"
        score = None
        sanity_data: dict = {}
        connectivity_data: dict = {}
        if failure is None:
            ledger.set_stage("CONVERT_DRC_REPORT")
            try:
                lyrpt_ref = self._ref(
                    report_path, "KLayoutBackend.run_drc", "application/xml",
                )
                attempted_drc_ref = self.adapter.convert_lyrpt_to_json(
                    lyrpt_ref,
                    DAC26CaseMeta(
                        case_id=case_id, design_type="block",
                        layout_script=patched_script,
                    ),
                    attempted_drc_path,
                )
                score = self.adapter.score_repair(
                    current_snapshot.drc_ref, attempted_drc_ref,
                )
                drc_validity = EvidenceValidity.VALID
                ledger.phase("DRC_VERIFIED")
            except Exception as exc:
                drc_validity = EvidenceValidity.ERROR
                failure_stage = "CONVERT_DRC_REPORT"
                failure = (
                    VerificationOutcome.DRC_EXECUTION_FAILURE,
                    "EVALUATOR_FAILURE_" + type(exc).__name__,
                )
        if failure is None:
            ledger.set_stage("SANITY")
            sanity = self.backend.run_sanity(
                job,
                helper=self.paths["sanity_helper"],
                connectivity_helper=self.paths["connectivity_helper"],
                original_gds=Path(baseline_snapshot.gds_ref.path),
                original_script=Path(baseline_snapshot.script_ref.path),
                timeout=self.backend_config.connectivity_timeout_seconds,
                **backend_control,
            )
            tools.append(sanity)
            if sanity.status != "SUCCESS":
                sanity_validity = EvidenceValidity.ERROR
                failure_stage = "SANITY"
                failure = (
                    VerificationOutcome.CANCELLED
                    if sanity.status == "CANCELLED"
                    else VerificationOutcome.GDS_SANITY_FAILURE,
                    "CANCELLED" if sanity.status == "CANCELLED"
                    else "SANITY_EXECUTION_FAILED",
                )
            if failure is None:
                try:
                    sanity_path = job.host_path / "report" / "sanity.json"
                    if not sanity_path.is_file():
                        raise FileNotFoundError(sanity_path)
                    sanity_data = json.loads(
                        sanity_path.read_text(encoding="utf-8")
                    )
                    sanity_validity = EvidenceValidity.VALID
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    sanity_validity = EvidenceValidity.ERROR
                    failure_stage = "SANITY_EVIDENCE"
                    failure = (
                        VerificationOutcome.GDS_SANITY_FAILURE,
                        "SANITY_EVIDENCE_INVALID_" + type(exc).__name__,
                    )
            if failure is None:
                ledger.set_stage("CONNECTIVITY")
                connectivity = self.backend.run_connectivity(
                    job, self.paths["connectivity_helper"],
                    self.backend_config.connectivity_timeout_seconds,
                    **backend_control,
                )
                tools.append(connectivity)
                if connectivity.status != "SUCCESS":
                    connectivity_validity = EvidenceValidity.ERROR
                    failure_stage = "CONNECTIVITY"
                    failure = (
                        VerificationOutcome.CANCELLED
                        if connectivity.status == "CANCELLED"
                        else VerificationOutcome.CONNECTIVITY_FAILURE,
                        "CANCELLED" if connectivity.status == "CANCELLED"
                        else "CONNECTIVITY_EXECUTION_FAILED",
                    )
                if failure is None:
                    try:
                        connectivity_path = (
                            job.host_path / "connectivity" / "result.json"
                        )
                        if not connectivity_path.is_file():
                            raise FileNotFoundError(connectivity_path)
                        connectivity_data = json.loads(
                            connectivity_path.read_text(encoding="utf-8")
                        )
                        connectivity_validity = EvidenceValidity.VALID
                    except (OSError, ValueError, json.JSONDecodeError) as exc:
                        connectivity_validity = EvidenceValidity.ERROR
                        failure_stage = "CONNECTIVITY_EVIDENCE"
                        failure = (
                            VerificationOutcome.CONNECTIVITY_FAILURE,
                            "CONNECTIVITY_EVIDENCE_INVALID_" + type(exc).__name__,
                        )

        removed_ids: list[str] = []
        new_ids: list[str] = []
        removed_markers: list[str] = []
        new_markers: list[str] = []
        if drc_validity == EvidenceValidity.VALID and score is not None:
            ledger.set_stage("DRC_DELTA")
            delta = conservative_violation_delta_evidence(
                current_drc,
                attempted_drc_path,
                case_id=case_id,
            )
            removed_ids = delta.removed_violation_ids
            new_ids = delta.new_violation_ids
            removed_markers = delta.removed_marker_fingerprints
            new_markers = delta.new_marker_fingerprints
            if len(removed_ids) != score.removed_violations:
                removed_ids = []
            if len(new_ids) != score.new_violations:
                new_ids = []
        complete_validity = all(
            value == EvidenceValidity.VALID
            for value in (
                script_validity,
                layout_validity,
                drc_validity,
                sanity_validity,
                connectivity_validity,
            )
        )
        terminal_status = (
            AttemptTerminalStatus.CANCELLED
            if failure is not None and failure[0] == VerificationOutcome.CANCELLED
            else AttemptTerminalStatus.EVALUATED
            if complete_validity
            else AttemptTerminalStatus.EXECUTION_FAILURE
        )
        if failure is not None:
            outcome, failure_code = failure
            verification = VerificationResult(
                verification_id="verification_" + stable_hash([
                    transaction.transaction_id, failure_code,
                ])[:20],
                transaction_id=transaction.transaction_id,
                outcome=outcome,
                script_valid=(
                    validate.status == "SUCCESS"
                    if script_validity != EvidenceValidity.NOT_EVALUATED else None
                ),
                gds_sanity_pass=(
                    bool(sanity_data.get("passed", False))
                    if sanity_validity == EvidenceValidity.VALID else None
                ),
                connectivity_preserved=(
                    bool(
                        connectivity_data.get("passed", False)
                        and connectivity_data.get("connectivity_preserved", False)
                    )
                    if connectivity_validity == EvidenceValidity.VALID else None
                ),
                baseline_residual_count=before.total_drv,
                residual_violation_count=(
                    score.final_violations if score is not None else None
                ),
                new_violation_count=(
                    score.new_violations if score is not None else None
                ),
                removed_original_count=(
                    score.removed_violations if score is not None else None
                ),
                removed_original_violation_ids=removed_ids,
                removed_original_marker_fingerprints=removed_markers,
                new_violation_ids=new_ids,
                new_marker_fingerprints=new_markers,
                failure_codes=[failure_code],
                execution_protocol_version=EXECUTION_PROTOCOL_VERSION,
                attempt_id=attempt_id,
                terminal_status=terminal_status,
                script_validity=script_validity,
                layout_validity=layout_validity,
                drc_validity=drc_validity,
                sanity_validity=sanity_validity,
                connectivity_validity=connectivity_validity,
                fresh_evidence_valid=complete_validity,
                failure_domain="EXECUTION",
                failure_stage=failure_stage,
            )
        else:
            script_valid = validate.status == "SUCCESS"
            sanity_pass = bool(sanity_data.get("passed", False))
            connectivity_preserved = bool(
                connectivity_data.get("passed", False) and
                connectivity_data.get("connectivity_preserved", False)
            )
            if not sanity_pass:
                outcome = VerificationOutcome.GDS_SANITY_FAILURE
            elif not connectivity_preserved:
                outcome = VerificationOutcome.CONNECTIVITY_FAILURE
            elif score.new_violations:
                outcome = VerificationOutcome.NEW_DRC_INTRODUCED
            elif score.removed_violations <= 0:
                outcome = VerificationOutcome.DRC_NO_PROGRESS
            else:
                outcome = VerificationOutcome.SUCCESS
            verification = VerificationResult(
                verification_id="verification_" + stable_hash([
                    transaction.transaction_id, score.model_dump(),
                    sanity_data, connectivity_data,
                ])[:20],
                transaction_id=transaction.transaction_id, outcome=outcome,
                script_valid=script_valid, gds_sanity_pass=sanity_pass,
                connectivity_preserved=connectivity_preserved,
                baseline_residual_count=score.original_violations,
                residual_violation_count=score.final_violations,
                new_violation_count=score.new_violations,
                removed_original_count=score.removed_violations,
                removed_original_violation_ids=removed_ids,
                removed_original_marker_fingerprints=removed_markers,
                new_violation_ids=new_ids,
                new_marker_fingerprints=new_markers,
                failure_codes=[] if outcome == VerificationOutcome.SUCCESS else [outcome.value],
                execution_protocol_version=EXECUTION_PROTOCOL_VERSION,
                attempt_id=attempt_id,
                terminal_status=terminal_status,
                script_validity=script_validity,
                layout_validity=layout_validity,
                drc_validity=drc_validity,
                sanity_validity=sanity_validity,
                connectivity_validity=connectivity_validity,
                fresh_evidence_valid=complete_validity,
                failure_domain=None,
                failure_stage=None,
            )
        ledger.set_stage("ACCEPTANCE")
        decision = evaluate_acceptance(verification, self.acceptance)
        transaction.tool_results = [
            item.model_dump(mode="json") for item in tools
        ]
        transaction.finished_at = utc_now()
        transaction.terminal_status = terminal_status
        transaction.status = (
            TransactionStatus.CANCELLED
            if terminal_status == AttemptTerminalStatus.CANCELLED
            else
            TransactionStatus.COMMITTED if decision.accepted
            else TransactionStatus.ROLLED_BACK
        )

        ledger.set_stage("PERSIST_ATTEMPT")
        gds_path = job.host_path / "gds" / f"{case_id}.gds"
        attempted_script_ref = self.store.copy(
            patched_script, f"{prefix}/attempted/{case_id}.py",
            producer="execute_transaction", media_type="text/x-python",
        )
        attempted_gds_ref = self.store.copy(
            gds_path, f"{prefix}/attempted/{case_id}.gds",
            producer="execute_transaction", media_type="application/gds",
        ) if gds_path.is_file() else None
        if report_path.is_file():
            report_suffix = ".retry.lyrpt" if ".retry." in report_path.name else ".lyrpt"
            self.store.copy(
                report_path, f"{prefix}/attempted/{case_id}{report_suffix}",
                producer="execute_transaction", media_type="application/xml",
            )
        attempted_drc_ref = self.store.copy(
            attempted_drc_path, f"{prefix}/attempted/{case_id}.drc.json",
            producer="convert_report", media_type="application/json",
        ) if attempted_drc_path.is_file() else None
        attempted_total = (
            drc_statistics(attempted_drc_path).total_drv
            if drc_validity == EvidenceValidity.VALID
            and attempted_drc_path.is_file() else None
        )
        if lineage is not None:
            lineage = dict(lineage)
            lineage["physical_eda_verified"] = bool(
                verification.fresh_evidence_valid
            )
            receipt_status = (
                "FRESH_VERIFIED"
                if verification.fresh_evidence_valid else
                "SOURCE_REPARSE_VERIFIED_PHYSICAL_EDA_INCOMPLETE"
            )
            lineage["direct_correspondences"] = [
                {**item, "verification_status": receipt_status}
                for item in lineage.get("direct_correspondences", [])
            ]
            lineage_receipt_ref = self.store.write_json(
                f"{prefix}/attempted/{case_id}.lineage.final.json",
                lineage,
                producer="execute_transaction",
                schema_name="CompilerLineageReceipt",
            )
        attempted_snapshot = DesignSnapshotRef(
            snapshot_id=attempted_snapshot_id,
            script_ref=attempted_script_ref, gds_ref=attempted_gds_ref,
            lineage_receipt_ref=lineage_receipt_ref,
            drc_ref=attempted_drc_ref,
            connectivity_ref=None,
            parent_snapshot_id=(
                current_snapshot.snapshot_id if lineage is not None else None
            ),
            parent_script_ref=(
                current_snapshot.script_ref if lineage is not None else None
            ),
            provenance_run_id=(run_id if lineage is not None else None),
            provenance_relation_source=(
                "COMPILER_REPARSE_PARENT_CHILD"
                if lineage is not None else None
            ),
            score=(
                (
                    int(verification.connectivity_preserved is True),
                    -int(verification.new_violation_count),
                    -int(verification.residual_violation_count),
                )
                if verification.new_violation_count is not None
                and verification.residual_violation_count is not None
                and verification.connectivity_preserved is not None
                else None
            ),
        )
        if decision.accepted:
            committed_snapshot = self._copy_snapshot(
                attempted_snapshot, f"{prefix}/committed", case_id,
            )
            # attempted/committed are two immutable artifact locations for
            # one accepted child state, not two provenance nodes. Keeping the
            # ID preserves the receipt's exact parent->child assertion.
            assert attempted_total is not None
            committed_total = attempted_total
        else:
            committed_snapshot = self._copy_snapshot(
                current_snapshot, f"{prefix}/committed", case_id,
            )
            committed_total = before.total_drv
        verification_ref = self.store.write_json(
            f"{prefix}/verification/verification.json", verification,
            producer="verify_result", schema_name="VerificationResult",
            schema_version="3.2",
        )
        rollback_reason = (
            None if decision.accepted else
            ",".join(decision.reasons or verification.failure_codes)
        )
        transaction.verification_result_ref = verification_ref
        transaction.decision_reasons = list(decision.reasons)
        transaction.rollback_reason = rollback_reason
        self.store.write_json(
            f"{prefix}/transaction/transaction.json", transaction,
            producer="commit_or_rollback", schema_name="Transaction",
            schema_version="3.2",
        )
        self.store.write_json(
            f"{base_prefix}/transaction/latest_attempt.json",
            {
                "execution_protocol_version": EXECUTION_PROTOCOL_VERSION,
                "attempt_id": attempt_id,
                "logical_evidence_key": logical_evidence_key,
                "terminal_status": terminal_status.value,
                "transaction_ref": str(
                    self.store.root / prefix / "transaction" / "transaction.json"
                ),
                "verification_ref": verification_ref.model_dump(mode="json"),
            },
            producer="transaction.compatibility_pointer",
            schema_name="LatestAttemptPointer",
            schema_version="3.2",
        )
        # Preserve the historical "latest transaction" filesystem contract
        # for readers that have not yet learned about attempt-scoped evidence.
        # These files are convenience aliases only; the immutable source of
        # truth remains ``<base_prefix>/attempts/<attempt_id>/...`` and the
        # pointer above records exactly which attempt supplied the alias.
        for snapshot_name in ("attempted", "committed"):
            source_dir = self.store.root / prefix / snapshot_name
            if not source_dir.is_dir():
                continue
            for source in sorted(source_dir.iterdir()):
                if source.is_file():
                    self.store.copy(
                        source,
                        f"{base_prefix}/{snapshot_name}/{source.name}",
                        producer="transaction.latest_alias",
                        media_type="application/octet-stream",
                    )
        for relative, media_type in (
            ("transaction/patch_plan.json", "application/json"),
            ("transaction/transaction.json", "application/json"),
            ("transaction/combined.diff", "text/x-diff"),
            ("verification/verification.json", "application/json"),
        ):
            source = self.store.root / prefix / relative
            if source.is_file():
                self.store.copy(
                    source,
                    f"{base_prefix}/{relative}",
                    producer="transaction.latest_alias",
                    media_type=media_type,
                )
        if legacy_artifact_prefix:
            for relative, media_type in (
                ("transaction/patch_plan.json", "application/json"),
                ("transaction/transaction.json", "application/json"),
                ("transaction/combined.diff", "text/x-diff"),
                ("verification/verification.json", "application/json"),
            ):
                source = self.store.root / prefix / relative
                if source.is_file():
                    self.store.copy(
                        source, f"{legacy_artifact_prefix}/{relative}",
                        producer="transaction.legacy_alias",
                        media_type=media_type,
                    )
        if isinstance(self.backend, KLayoutBackend):
            self.backend.mark_job_state(
                job, AttemptLifecycle.TERMINAL,
                details={
                    "terminal_status": terminal_status.value,
                    "verification_outcome": verification.outcome.value,
                    "fresh_evidence_valid": verification.fresh_evidence_valid,
                },
            )
        evidence_dir = (
            self.store.root / prefix / "transaction" / "tool_evidence"
        )
        ledger.set_stage("COLLECT_RESULTS", failure_domain="INFRASTRUCTURE")
        try:
            collected = self.backend.collect_results(job, evidence_dir)
        except Exception as exc:
            raise InfrastructureFailure(
                f"ARTIFACT_PUBLICATION_FAILED: {exc}",
                failure_code=FailureCode.ARTIFACT_PUBLICATION,
                retryable=isinstance(exc, OSError),
                provider="KLayoutBackend",
                model=backend_image,
                original_exception_type=type(exc).__name__,
                failure_stage="COLLECT_RESULTS",
            ) from exc
        if hasattr(self.backend, "remap_tool_results"):
            try:
                tools = self.backend.remap_tool_results(job, tools)
                transaction.tool_results = [
                    item.model_dump(mode="json") for item in tools
                ]
                self.store.write_json(
                    f"{prefix}/transaction/transaction.json", transaction,
                    producer="commit_or_rollback",
                    schema_name="Transaction", schema_version="3.2",
                )
                self.store.copy(
                    self.store.root / prefix / "transaction" / "transaction.json",
                    f"{base_prefix}/transaction/transaction.json",
                    producer="transaction.latest_alias",
                    media_type="application/json",
                )
                if legacy_artifact_prefix:
                    self.store.copy(
                        self.store.root / prefix / "transaction" / "transaction.json",
                        f"{legacy_artifact_prefix}/transaction/transaction.json",
                        producer="transaction.legacy_alias",
                        media_type="application/json",
                    )
            except Exception as exc:
                raise InfrastructureFailure(
                    f"ARTIFACT_REFERENCE_REMAP_FAILED: {exc}",
                    failure_code=FailureCode.ARTIFACT_PUBLICATION,
                    retryable=False,
                    provider="KLayoutBackend",
                    model=backend_image,
                    original_exception_type=type(exc).__name__,
                    failure_stage="COLLECT_RESULTS",
                ) from exc
        keep_workspace = (
            self.backend_config.keep_failed_workspace and not decision.accepted
        )
        ledger.set_stage("CLEANUP", failure_domain="INFRASTRUCTURE")
        try:
            self.backend.cleanup(job, keep_workspace=keep_workspace)
        except Exception as exc:
            raise InfrastructureFailure(
                f"WORKSPACE_CLEANUP_FAILED: {exc}",
                failure_code=FailureCode.WORKSPACE_OWNERSHIP,
                retryable=False,
                provider="KLayoutBackend",
                model=backend_image,
                original_exception_type=type(exc).__name__,
                failure_stage="CLEANUP",
            ) from exc
        infra_tool = next((
            item for item in tools
            if item.error_code in {"EXECUTABLE_NOT_FOUND", "INVALID_INPUT_PATH"}
        ), None)
        if infra_tool is not None:
            raise InfrastructureFailure(
                infra_tool.error_message or infra_tool.error_code or "EDA unavailable",
                failure_code=FailureCode.EDA_UNAVAILABLE,
                retryable=False,
                provider="KLayoutBackend",
                model=backend_image,
                original_exception_type=infra_tool.error_code,
                failure_stage=infra_tool.action,
            )
        ledger.terminal(
            terminal_status,
            verification_outcome=verification.outcome.value,
            fresh_evidence_valid=verification.fresh_evidence_valid,
            drc_validity=verification.drc_validity.value,
            sanity_validity=verification.sanity_validity.value,
            connectivity_validity=verification.connectivity_validity.value,
            accepted=decision.accepted,
            workspace_retained=keep_workspace,
            result_manifest_refs=[
                item.model_dump(mode="json") for item in collected
                if Path(item.path).name == "result_manifest.json"
            ],
        )
        return TransactionExecutionResult(
            transaction=transaction, verification=verification,
            accepted=decision.accepted, decision_reasons=decision.reasons,
            attempted_snapshot=attempted_snapshot,
            committed_snapshot=committed_snapshot,
            attempted_total_drv=attempted_total,
            committed_total_drv=committed_total,
            tool_runtime_seconds=sum(item.runtime_seconds for item in tools),
            rollback_reason=rollback_reason,
            attempt_id=attempt_id,
            logical_evidence_key=logical_evidence_key,
        )

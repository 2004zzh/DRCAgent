from __future__ import annotations

from pathlib import Path
import json

from drc_agent.config.loader import AppConfig
from drc_agent.development.repair_kernel_bench.identity import (
    normalized_marker_fingerprint,
)
from drc_agent.repair_kernel_multistep_repair.current_context import (
    CurrentSemanticContextBuilder,
)
from drc_agent.repair_kernel_multistep_repair.models import CurrentSemanticContext
from drc_agent.reliability import IntegrityFailure
from drc_agent.schemas.common import StrictModel, file_sha256, stable_hash
from drc_agent.schemas.state import RegionState, ViolationRecord

from .models import FormalKernelExecutionContext


class FormalSemanticBinding(StrictModel):
    formal_region_id: str
    semantic_region_id: str
    formal_violation_id: str
    semantic_violation_id: str
    rule_id: str
    binding_mode: str
    binding_fingerprint: str


class FormalCurrentSemanticResult(StrictModel):
    context: CurrentSemanticContext
    bindings: list[FormalSemanticBinding]


def _require_ref(ref, label: str) -> Path:
    if ref is None:
        code = f"FORMAL_SNAPSHOT_{label}_MISSING"
        raise IntegrityFailure(
            code, failure_code=code,
            failure_stage="CURRENT_SNAPSHOT_ARTIFACT",
        )
    path = Path(ref.path).resolve()
    if not path.is_file():
        code = f"FORMAL_SNAPSHOT_{label}_MISSING"
        raise IntegrityFailure(
            code, failure_code=code,
            failure_stage="CURRENT_SNAPSHOT_ARTIFACT",
            details={"path": str(path)},
        )
    if file_sha256(path) != ref.sha256:
        code = f"FORMAL_SNAPSHOT_{label}_SHA_MISMATCH"
        raise IntegrityFailure(
            code, failure_code=code,
            failure_stage="CURRENT_SNAPSHOT_ARTIFACT",
            details={"path": str(path)},
        )
    return path


class FormalCurrentSemanticAdapter:
    """Rebuild current physical truth from formal snapshot artifact refs."""

    def __init__(
        self,
        project_root: Path,
        config: AppConfig,
        *,
        require_frozen_rule_deck: bool = True,
    ):
        self.project_root = project_root.resolve()
        self.config = config
        self.require_frozen_rule_deck = bool(require_frozen_rule_deck)

    @staticmethod
    def _semantic_region(
        context: CurrentSemanticContext, violation: ViolationRecord,
    ) -> RegionState:
        region_id = context.violation_to_region.get(violation.violation_id)
        region = next(
            (item for item in context.regions if item.region_id == region_id),
            None,
        )
        if region is None:
            raise ValueError("FORMAL_CURRENT_REGION_UNAVAILABLE")
        return region

    @staticmethod
    def _bind_violation(
        context: CurrentSemanticContext, formal: ViolationRecord,
    ) -> tuple[ViolationRecord, str]:
        direct = next(
            (item for item in context.violations
             if item.violation_id == formal.violation_id),
            None,
        )
        if direct is not None:
            return direct, "EXACT_CURRENT_ID"
        fingerprint = normalized_marker_fingerprint(formal)
        matches = [
            item for item in context.violations
            if item.rule_id == formal.rule_id
            and normalized_marker_fingerprint(item) == fingerprint
        ]
        if len(matches) != 1:
            raise ValueError("FORMAL_CURRENT_VIOLATION_BINDING_AMBIGUOUS")
        return matches[0], "NORMALIZED_MARKER_FINGERPRINT"

    def build(
        self, execution: FormalKernelExecutionContext,
    ) -> FormalCurrentSemanticResult:
        current = execution.current_snapshot
        baseline = execution.baseline_snapshot
        semantic = self.build_snapshot_context(
            current=current,
            baseline=baseline,
            case_id=execution.case_id,
            rule_deck_path=Path(execution.rule_deck_path),
            run_id=execution.run_id,
        )
        bindings = []
        wanted = set(execution.region.violation_ids)
        for formal in sorted(
            (item for item in execution.violations
             if item.violation_id in wanted),
            key=lambda item: item.violation_id,
        ):
            current_violation, mode = self._bind_violation(semantic, formal)
            semantic_region = self._semantic_region(semantic, current_violation)
            payload = [
                execution.region.region_id,
                semantic_region.region_id,
                formal.violation_id,
                current_violation.violation_id,
                mode,
            ]
            bindings.append(FormalSemanticBinding(
                formal_region_id=execution.region.region_id,
                semantic_region_id=semantic_region.region_id,
                formal_violation_id=formal.violation_id,
                semantic_violation_id=current_violation.violation_id,
                rule_id=formal.rule_id,
                binding_mode=mode,
                binding_fingerprint=stable_hash(payload),
            ))
        if not bindings:
            raise ValueError("FORMAL_CURRENT_REGION_HAS_NO_BOUND_VIOLATION")
        return FormalCurrentSemanticResult(context=semantic, bindings=bindings)

    def build_snapshot_context(
        self,
        *,
        current,
        baseline,
        case_id: str,
        rule_deck_path: Path,
        run_id: str,
    ) -> CurrentSemanticContext:
        """Validate and rebuild one master snapshot before it is published."""

        validated = self.validate_snapshot_provenance(
            current=current,
            baseline=baseline,
            run_id=run_id,
        )
        builder = CurrentSemanticContextBuilder(
            self.project_root,
            self.config,
            rule_deck_path=rule_deck_path,
            require_frozen_rule_deck=self.require_frozen_rule_deck,
        )
        return builder.build(
            snapshot_id=current.snapshot_id,
            case_id=case_id,
            script_path=validated["script"],
            drc_path=validated["drc"],
            connectivity_path=validated["connectivity"],
            root_script_path=validated["parent_script"],
            lineage_receipt=validated["receipt"],
            lineage_parent_snapshot_id=validated["parent_snapshot_id"],
            lineage_run_id=validated["lineage_run_id"],
            require_physical_lineage=validated["receipt"] is not None,
        )

    def validate_snapshot_provenance(
        self,
        *,
        current,
        baseline,
        run_id: str,
    ) -> dict[str, object]:
        """Verify an immutable snapshot edge without requiring formal rules.

        Every runtime can enforce artifact and immediate-parent integrity.
        Only formal-kernel profiles additionally need the rule-deck-bound
        semantic-context reconstruction performed by ``build_snapshot_context``.
        """

        script = _require_ref(current.script_ref, "SCRIPT")
        drc = _require_ref(current.drc_ref, "DRC")
        connectivity_ref = current.connectivity_ref or baseline.connectivity_ref
        connectivity = _require_ref(connectivity_ref, "CONNECTIVITY")
        if file_sha256(script) != current.script_ref.sha256:
            raise IntegrityFailure(
                "FORMAL_CURRENT_SCRIPT_STALE",
                failure_code="FORMAL_CURRENT_SCRIPT_STALE",
                failure_stage="CURRENT_SNAPSHOT_ARTIFACT",
            )
        receipt = None
        parent_script = script
        parent_snapshot_id = None
        lineage_run_id = None
        if current.lineage_receipt_ref is not None:
            if (
                current.parent_snapshot_id is None
                or current.parent_script_ref is None
                or current.provenance_run_id is None
                or current.provenance_relation_source
                != "COMPILER_REPARSE_PARENT_CHILD"
            ):
                raise IntegrityFailure(
                    "SNAPSHOT_IMMEDIATE_PARENT_PROVENANCE_INCOMPLETE",
                    failure_code=(
                        "SNAPSHOT_IMMEDIATE_PARENT_PROVENANCE_INCOMPLETE"
                    ),
                    failure_stage="CURRENT_SNAPSHOT_PROVENANCE",
                    details={"snapshot_id": current.snapshot_id},
                )
            if current.provenance_run_id != run_id:
                raise IntegrityFailure(
                    "SNAPSHOT_CROSS_RUN_PROVENANCE",
                    failure_code="SNAPSHOT_CROSS_RUN_PROVENANCE",
                    failure_stage="CURRENT_SNAPSHOT_PROVENANCE",
                    details={
                        "expected": run_id,
                        "actual": current.provenance_run_id,
                    },
                )
            parent_script = _require_ref(
                current.parent_script_ref, "PARENT_SCRIPT",
            )
            parent_snapshot_id = current.parent_snapshot_id
            lineage_run_id = run_id
            receipt = json.loads(
                _require_ref(
                    current.lineage_receipt_ref, "LINEAGE_RECEIPT",
                ).read_text(encoding="utf-8")
            )
        elif current.snapshot_id != baseline.snapshot_id:
            raise IntegrityFailure(
                "NON_BASELINE_SNAPSHOT_LINEAGE_RECEIPT_MISSING",
                failure_code="NON_BASELINE_SNAPSHOT_LINEAGE_RECEIPT_MISSING",
                failure_stage="CURRENT_SNAPSHOT_PROVENANCE",
                details={"snapshot_id": current.snapshot_id},
            )
        if receipt is not None:
            from drc_agent.patching.compiler import verify_lineage_receipt

            verify_lineage_receipt(
                script,
                receipt,
                parent_script=parent_script,
                expected_parent_snapshot_id=parent_snapshot_id,
                expected_child_snapshot_id=current.snapshot_id,
                expected_run_id=lineage_run_id,
                require_physical_verified=True,
            )
        return {
            "script": script,
            "drc": drc,
            "connectivity": connectivity,
            "parent_script": parent_script,
            "receipt": receipt,
            "parent_snapshot_id": parent_snapshot_id,
            "lineage_run_id": lineage_run_id,
        }

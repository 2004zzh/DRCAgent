from __future__ import annotations

from collections import defaultdict
from enum import StrEnum
from typing import Callable

from pydantic import Field

from drc_agent.config.loader import AttributionConfig, TransactionConfig
from drc_agent.schemas.action import (
    JointRepairBundle,
    RepairCandidate,
    VerificationResult,
)
from drc_agent.schemas.common import StrictModel, stable_hash
from drc_agent.schemas.workflow import DesignSnapshotRef, TransactionBatch


class AttributionQuality(StrEnum):
    DIRECT_CANDIDATE = "DIRECT_CANDIDATE"
    DIRECT_SMALL_BUNDLE = "DIRECT_SMALL_BUNDLE"
    DELTA_DEBUGGED = "DELTA_DEBUGGED"
    BUNDLE_ONLY = "BUNDLE_ONLY"
    UNKNOWN = "UNKNOWN"


class AttributionResult(StrictModel):
    attribution_id: str
    transaction_id: str
    quality: AttributionQuality
    candidate_ids: list[str]
    removed_original_violation_ids: list[str] = Field(default_factory=list)
    new_violation_ids: list[str] = Field(default_factory=list)
    reproduced_candidate_ids: list[str] = Field(default_factory=list)
    replay_count: int = 0
    claims_candidate_causality: bool = False


class TransactionBatchScheduler:
    """Deterministic bounded batches; execution remains serialized."""

    def schedule(
        self,
        *,
        bundles: list[JointRepairBundle],
        candidates: list[RepairCandidate],
        snapshot: DesignSnapshotRef,
        config: TransactionConfig,
    ) -> list[TransactionBatch]:
        by_id = {candidate.candidate_id: candidate for candidate in candidates}
        pending = []
        for bundle in sorted(bundles, key=lambda item: item.bundle_id):
            selected = [
                by_id[candidate_id]
                for candidate_id in bundle.selected_candidate_ids
                if candidate_id in by_id and not by_id[candidate_id].is_noop
            ]
            if not selected:
                continue
            pending.append((bundle, selected))

        batches: list[TransactionBatch] = []
        current_bundles: list[str] = []
        current_candidates: list[str] = []
        current_objects: set[str] = set()

        def flush() -> None:
            nonlocal current_bundles, current_candidates, current_objects
            if not current_bundles:
                return
            identity = [
                snapshot.snapshot_id,
                current_bundles,
                current_candidates,
            ]
            batches.append(TransactionBatch(
                batch_id=f"batch_{stable_hash(identity)[:16]}",
                bundle_ids=list(current_bundles),
                base_snapshot_id=snapshot.snapshot_id,
                candidate_ids=list(current_candidates),
                non_noop_action_count=len(current_candidates),
                modified_object_count=len(current_objects),
                requires_replan_after_commit=config.replan_after_each_commit,
            ))
            current_bundles = []
            current_candidates = []
            current_objects = set()

        for bundle, selected in pending:
            candidate_ids = sorted(item.candidate_id for item in selected)
            objects = set().union(*(
                set(item.affected_object_ids or item.target_object_ids)
                for item in selected
            ))
            if (
                len(candidate_ids) > config.max_non_noop_actions_per_batch
                or len(objects) > config.max_modified_objects_per_batch
            ):
                flush()
                # Oversized bundles are deferred rather than silently split,
                # because splitting can invalidate cooperation semantics.
                continue
            if (
                len(current_candidates) + len(candidate_ids)
                > config.max_non_noop_actions_per_batch
                or len(current_objects | objects)
                > config.max_modified_objects_per_batch
            ):
                flush()
            current_bundles.append(bundle.bundle_id)
            current_candidates.extend(candidate_ids)
            current_objects.update(objects)
        flush()
        return batches


class FailureAttributor:
    def attribute(
        self,
        *,
        transaction_id: str,
        candidates: list[RepairCandidate],
        verification: VerificationResult,
        config: AttributionConfig,
        replay_subset: Callable[[list[str]], bool] | None = None,
    ) -> AttributionResult:
        candidate_ids = sorted(candidate.candidate_id for candidate in candidates)
        if len(candidate_ids) == 1:
            quality = AttributionQuality.DIRECT_CANDIDATE
            causal = True
        elif candidate_ids and self._source_disjoint(candidates):
            quality = AttributionQuality.DIRECT_SMALL_BUNDLE
            causal = False
        else:
            quality = AttributionQuality.BUNDLE_ONLY if candidate_ids else AttributionQuality.UNKNOWN
            causal = False

        replay_count = 0
        reproduced: list[str] = []
        if (
            config.enabled
            and config.delta_debug_enabled
            and replay_subset is not None
            and len(candidate_ids) > 1
        ):
            current = candidate_ids
            while (
                len(current) > 1
                and replay_count < config.max_replays_per_failure
            ):
                midpoint = len(current) // 2
                halves = [current[:midpoint], current[midpoint:]]
                found = None
                for half in halves:
                    if not half:
                        continue
                    replay_count += 1
                    if replay_subset(half):
                        found = half
                        break
                    if replay_count >= config.max_replays_per_failure:
                        break
                if found is None:
                    break
                current = found
            if current and len(current) < len(candidate_ids):
                quality = AttributionQuality.DELTA_DEBUGGED
                reproduced = current
                causal = len(current) == 1

        return AttributionResult(
            attribution_id=f"attribution_{stable_hash([transaction_id, candidate_ids, quality])[:16]}",
            transaction_id=transaction_id,
            quality=quality,
            candidate_ids=candidate_ids,
            removed_original_violation_ids=(
                verification.removed_original_violation_ids if causal else []
            ),
            new_violation_ids=verification.new_violation_ids if causal else [],
            reproduced_candidate_ids=reproduced,
            replay_count=replay_count,
            claims_candidate_causality=causal,
        )

    @staticmethod
    def _source_disjoint(candidates: list[RepairCandidate]) -> bool:
        seen_anchors: set[str] = set()
        seen_objects: set[str] = set()
        for candidate in candidates:
            anchors = {
                anchor for edit in candidate.edits
                for anchor in edit.source_anchor_ids
            }
            objects = set(candidate.affected_object_ids or candidate.target_object_ids)
            if seen_anchors & anchors or seen_objects & objects:
                return False
            seen_anchors.update(anchors)
            seen_objects.update(objects)
        return True


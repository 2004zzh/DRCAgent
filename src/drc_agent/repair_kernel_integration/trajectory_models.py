from __future__ import annotations

from typing import Any

from pydantic import Field

from drc_agent.repair_kernel_multistep.models import (
    RootTask,
    SearchSnapshot,
    TransitionResult,
)
from drc_agent.schemas.action import PatchPlan, RepairCandidate
from drc_agent.schemas.common import StrictModel, stable_hash


class FormalTrajectoryStep(StrictModel):
    depth: int = Field(ge=1)
    parent_snapshot_id: str
    child_snapshot_id: str | None = None
    proposal_id: str
    candidate: RepairCandidate
    patch_plan: PatchPlan
    transition: TransitionResult
    obligation_ids: list[str] = Field(default_factory=list)
    physical_effect_fingerprint: str
    stage: str
    symbolic_step_binding: dict[str, Any] = Field(default_factory=dict)


class FormalVerifiedTrajectory(StrictModel):
    trajectory_id: str
    root_task: RootTask
    root_snapshot: SearchSnapshot
    final_snapshot: SearchSnapshot
    steps: list[FormalTrajectoryStep]
    strict_clean: bool
    sandbox_calls: int = Field(ge=0)
    state_fingerprints: list[str]
    physical_effect_sequence: list[str]
    evidence: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def build(
        cls,
        *,
        task: RootTask,
        root: SearchSnapshot,
        final: SearchSnapshot,
        steps: list[FormalTrajectoryStep],
        sandbox_calls: int,
        evidence: dict[str, Any] | None = None,
    ) -> "FormalVerifiedTrajectory":
        effects = [item.physical_effect_fingerprint for item in steps]
        identity = stable_hash([
            task.root_sample_id,
            root.state_fingerprint,
            final.state_fingerprint,
            effects,
        ])
        return cls(
            trajectory_id="formal_trajectory_" + identity[:20],
            root_task=task,
            root_snapshot=root,
            final_snapshot=final,
            steps=steps,
            strict_clean=bool(
                not final.root_target_present
                and final.root_relative_removed_original > 0
                and final.root_relative_new == 0
            ),
            sandbox_calls=sandbox_calls,
            state_fingerprints=[root.state_fingerprint] + [
                item.transition.child_snapshot.state_fingerprint
                for item in steps
                if item.transition.child_snapshot is not None
            ],
            physical_effect_sequence=effects,
            evidence=evidence or {},
        )


class FormalTrajectoryOutcome(StrictModel):
    status: str
    target_rule_id: str
    target_violation_id: str
    root_proposal_count: int = Field(ge=0)
    sandbox_calls: int = Field(ge=0)
    trajectories: list[FormalVerifiedTrajectory] = Field(default_factory=list)
    events: list[dict[str, Any]] = Field(default_factory=list)
    deepest_failure: str | None = None
    execution_feedback: dict[str, Any] | None = None


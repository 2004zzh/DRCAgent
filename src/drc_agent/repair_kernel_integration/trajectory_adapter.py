from __future__ import annotations

from typing import Any

from pydantic import Field

from drc_agent.schemas.action import RepairCandidate
from drc_agent.schemas.common import StrictModel, stable_hash


class TrajectoryStep(StrictModel):
    depth: int
    candidate_id: str
    physical_effect_fingerprint: str
    patch_plan_sha256: str
    starting_snapshot_sha256: str
    ending_snapshot_sha256: str
    phase2_classification: str
    transition_class: str
    connectivity: bool
    protected_relations: list[str] = Field(default_factory=list)


class CondensedTrajectory(StrictModel):
    trajectory_id: str
    root_snapshot_sha256: str
    final_snapshot_sha256: str
    candidate: RepairCandidate
    steps: list[TrajectoryStep]
    physical_effect_sequence: list[str]
    equivalence_status: str
    equivalence_evidence: dict[str, Any] = Field(default_factory=dict)
    fingerprint: str


def condense_trajectory(
    *,
    root_script_sha256: str,
    final_script_sha256: str,
    final_candidate: RepairCandidate,
    steps: list[TrajectoryStep],
    equivalence_evidence: dict[str, Any] | None = None,
) -> CondensedTrajectory:
    """Represent a verified root-to-final trajectory as one Graph Action.

    This adapter does not merge source edits itself.  The final candidate must
    already be canonical and rooted at the frozen root source hash; callers
    that cannot prove this receive a fail-closed ``ValueError``.
    """
    if not root_script_sha256 or len(root_script_sha256) != 64:
        raise ValueError("TRAJECTORY_ROOT_HASH_INVALID")
    if not final_script_sha256 or len(final_script_sha256) != 64:
        raise ValueError("TRAJECTORY_FINAL_HASH_INVALID")
    if not steps:
        raise ValueError("TRAJECTORY_EMPTY")
    if any(not item.connectivity for item in steps):
        raise ValueError("TRAJECTORY_CONNECTIVITY_FAILURE")
    if any(item.depth != index + 1 for index, item in enumerate(steps)):
        raise ValueError("TRAJECTORY_DEPTH_NOT_CONTIGUOUS")
    if final_candidate.is_noop:
        raise ValueError("TRAJECTORY_FINAL_NOOP")
    payload = {
        "root": root_script_sha256,
        "final": final_script_sha256,
        "candidate": final_candidate.model_dump(mode="json"),
        "steps": [item.model_dump(mode="json") for item in steps],
    }
    return CondensedTrajectory(
        trajectory_id="trajectory_" + stable_hash(payload)[:20],
        root_snapshot_sha256=root_script_sha256,
        final_snapshot_sha256=final_script_sha256,
        candidate=final_candidate,
        steps=steps,
        physical_effect_sequence=[item.physical_effect_fingerprint for item in steps],
        equivalence_status="PROVEN" if (equivalence_evidence or {}).get("proven") else "PENDING",
        equivalence_evidence=equivalence_evidence or {},
        fingerprint=stable_hash(payload),
    )

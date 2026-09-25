from __future__ import annotations

from collections import defaultdict

from drc_agent.development.repair_kernel_bench.identity import (
    normalized_marker_fingerprint,
)
from drc_agent.schemas.common import stable_hash
from drc_agent.schemas.state import ViolationRecord

from .models import CurrentViolationIdentity, DebtMarkerGroup


def _canonical_geometry(violation: ViolationRecord) -> list:
    values = []
    for item in violation.marker_geometry_dbu:
        if hasattr(item, "start"):
            values.append([
                item.start.x, item.start.y, item.end.x, item.end.y,
            ])
        else:
            values.append([item.x, item.y])
    return sorted(values)


def build_debt_marker_groups(
    violations: list[ViolationRecord], snapshot_id: str,
) -> list[DebtMarkerGroup]:
    grouped: dict[tuple[str, str], list[ViolationRecord]] = defaultdict(list)
    for violation in violations:
        grouped[(
            violation.rule_id, normalized_marker_fingerprint(violation),
        )].append(violation)
    result = []
    for (rule_id, fingerprint), members in sorted(grouped.items()):
        members = sorted(members, key=lambda item: (
            item.source_index, item.violation_id,
        ))
        identities = [
            CurrentViolationIdentity(
                identity_id="current_violation_" + stable_hash([
                    snapshot_id, rule_id, fingerprint, index,
                    member.violation_id,
                ])[:20],
                rule_id=rule_id,
                normalized_marker_fingerprint=fingerprint,
                canonical_marker_geometry=_canonical_geometry(member),
                marker_bbox=member.marker_bbox_dbu,
                multiplicity_index=index,
                current_violation_id=member.violation_id,
                snapshot_id=snapshot_id,
            )
            for index, member in enumerate(members)
        ]
        multiset_key = stable_hash([
            rule_id, fingerprint,
            [item.canonical_marker_geometry for item in identities],
            len(identities),
        ])
        result.append(DebtMarkerGroup(
            group_id="debt_group_" + stable_hash([
                snapshot_id, rule_id, fingerprint, multiset_key,
            ])[:20],
            rule_id=rule_id,
            normalized_fingerprint=fingerprint,
            member_violation_ids=[item.current_violation_id for item in identities],
            member_count=len(identities),
            representative_marker=members[0].marker_bbox_dbu,
            marker_multiset_key=multiset_key,
            identities=identities,
        ))
    return result

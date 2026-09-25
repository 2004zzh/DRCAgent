from __future__ import annotations

from collections import defaultdict

from drc_agent.schemas.common import stable_hash
from drc_agent.schemas.experience import (
    ApplicabilityStatus,
    BlueprintAction,
    BlueprintClaim,
    BlueprintWarning,
    EvidencePack,
    ExperienceQuery,
    RepairBlueprint,
    RollbackTrigger,
    VerificationStep,
)


class BlueprintBuilder:
    async def build(
        self, query: ExperienceQuery, evidence: EvidencePack,
    ) -> RepairBlueprint:
        scope_id = query.planning_scope_id or query.subgraph_id
        if not evidence.items and not query.current_coordination_requirements:
            empty = RepairBlueprint.empty(query.subgraph_id)
            return empty.model_copy(update={
                "planning_scope_id": scope_id,
                "planning_scope_type": query.planning_scope_type,
                "view_ids": query.view_ids,
                "missing_evidence": ["no_applicable_experience"],
                "validation_status": "VALID_WITH_MISSING_EVIDENCE",
            })

        successes = [
            item for item in evidence.items
            if (
                item.kind == "episode" and item.outcome == "VERIFIED_SUCCESS"
            ) or (
                item.kind == "trial" and item.outcome == "CLEAN_PROGRESS"
            )
        ]
        failures = [
            item for item in evidence.items
            if item.outcome and (
                "FAILURE" in item.outcome
                or item.outcome in {
                    "REGRESSION", "CONNECTIVITY_FAIL", "EXECUTION_FAIL",
                    "TIMEOUT", "NO_PROGRESS",
                }
            )
        ]
        priors = [item for item in evidence.items if item.kind == "prior"]
        trials = [item for item in evidence.items if item.kind == "trial"]

        action_evidence: dict[str, list] = defaultdict(list)
        for item in [*successes, *priors]:
            for action in item.action_families:
                if action in query.allowed_action_families:
                    action_evidence[action].append(item)

        preferred: list[BlueprintAction] = []
        conditional: list[BlueprintAction] = []
        for action, support in sorted(action_evidence.items()):
            applicable = [
                item for item in support
                if item.applicability_status == ApplicabilityStatus.APPLICABLE
                and item.applicability
            ]
            conditional_support = [
                item for item in support
                if item.applicability_status == ApplicabilityStatus.CONDITIONAL
                and item.applicability
            ]
            chosen = applicable or conditional_support
            if not chosen:
                continue
            evidence_ids = list(dict.fromkeys(
                item.experience_id for item in chosen
            ))[:4]
            predicates = []
            avoid = []
            risks = []
            for item in chosen:
                predicates.extend(item.applicability)
                avoid.extend(item.avoid_conditions)
                risks.extend(item.expected_secondary_risks)
            first = chosen[0]
            item_class = (
                "VERIFIED_EPISODE" if first.kind == "episode"
                else "CANDIDATE_TRIAL" if first.kind == "trial"
                else "UNVERIFIED_PRIOR"
            )
            blueprint_action = BlueprintAction(
                action_family=action,
                applicability=list({item.predicate_id: item for item in predicates}.values()),
                avoid_conditions=list({item.predicate_id: item for item in avoid}.values()),
                expected_secondary_risks=sorted(set(risks)),
                evidence_ids=evidence_ids,
                evidence_class=item_class,
                attribution_quality=first.attribution_quality,
                confidence_milli=(850 if applicable else 350),
            )
            (preferred if applicable else conditional).append(blueprint_action)
        preferred = preferred[:3]
        conditional = conditional[:3]

        root_causes = []
        if priors:
            root_causes.append(BlueprintClaim(
                claim=(
                    "Curated prior knowledge identifies conditional physical "
                    "repair mechanisms for this bounded scope."
                ),
                evidence_ids=[item.experience_id for item in priors[:4]],
            ))
        if successes:
            root_causes.append(BlueprintClaim(
                claim="Analogous fresh verified evidence supports a bounded edit.",
                evidence_ids=[item.experience_id for item in successes[:3]],
            ))

        warnings = []
        if failures:
            direct = [
                item for item in failures
                if item.attribution_quality in {
                    "DIRECT_CANDIDATE", "DIRECT_SMALL_BUNDLE", "DELTA_DEBUGGED"
                }
            ]
            warnings.append(BlueprintWarning(
                warning=(
                    "Direct tool evidence reports a failure in this context."
                    if direct else
                    "Bundle-level evidence reports a failure without "
                    "candidate-specific causality."
                ),
                evidence_ids=[
                    item.experience_id for item in (direct or failures)[:4]
                ],
            ))
        restricted = [
            item for item in priors
            if "UNSUPPORTED_ACTIONS_PRESENT" in item.transfer_risks
            or "DIAGNOSTIC_ONLY" in item.transfer_risks
        ]
        if restricted:
            warnings.append(BlueprintWarning(
                warning=(
                    "Backend-restricted prior actions are diagnostic only and "
                    "must not be lowered on this backend."
                ),
                evidence_ids=[item.experience_id for item in restricted[:4]],
            ))

        missing = []
        if not successes:
            missing.append("verified_success")
        if not failures:
            missing.append("contrastive_failure")
        if conditional:
            missing.append("conditional_action_predicates")
        blueprint = RepairBlueprint(
            blueprint_id=f"blueprint_{stable_hash([query, evidence])[:20]}",
            knowledge_cutoff=query.knowledge_cutoff,
            symbolic_lessons=list({lesson.prototype_id:lesson
                for item in reversed(evidence.items)
                if item.applicability_status != ApplicabilityStatus.INAPPLICABLE
                for lesson in item.symbolic_lessons
                if set(lesson.rule_ids).intersection(query.signature.rule_histogram)
            }.values())[:8],
            subgraph_id=query.subgraph_id,
            planning_scope_id=scope_id,
            planning_scope_type=query.planning_scope_type,
            view_ids=query.view_ids,
            graph_version="v2",
            query_signature_hash=evidence.query_hash,
            root_causes=root_causes,
            preferred_actions=preferred,
            conditional_actions=conditional,
            avoid_conditions=warnings,
            coordination_requirements=query.current_coordination_requirements,
            verification_order=[
                VerificationStep(name=name)
                for name in ["syntax", "layout", "drc", "connectivity"]
            ],
            rollback_triggers=[
                RollbackTrigger(code=code)
                for code in ["tool_failure", "new_drc", "connectivity_failure"]
            ],
            supporting_experience_ids=sorted({
                item.experience_id for item in evidence.items
                if item.kind != "trial"
            }),
            supporting_trial_ids=sorted({
                item.experience_id for item in trials
            }),
            missing_evidence=missing,
            confidence_milli=min(
                900, 100 + len(priors) * 30 + len(successes) * 120
                + len(trials) * 80,
            ),
            generated_by="DETERMINISTIC",
            validation_status=(
                "VALID_WITH_MISSING_EVIDENCE" if missing else "VALID"
            ),
        )
        blueprint.validate_citations(
            evidence.evidence_ids, query.allowed_action_families,
        )
        return blueprint

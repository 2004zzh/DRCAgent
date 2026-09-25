from __future__ import annotations

from collections import defaultdict
from math import exp

import networkx as nx
from pydantic import BaseModel, Field

from drc_agent.config.loader import AgentGraphConfig
from drc_agent.schemas.common import stable_hash
from drc_agent.schemas.state import (
    AgentEdge, AgentSubgraph, EdgeEvidence, NeighborMessage, RegionState,
    StructuredConstraint, PotentialPhysicalAccessSummary,
)


class AgentGraph(BaseModel):
    regions: dict[str, RegionState]
    boundary_dependencies: list = Field(default_factory=list)
    planning_views: list = Field(default_factory=list)
    hierarchy_scopes: list = Field(default_factory=list)
    active_scopes: list[AgentSubgraph] = Field(default_factory=list)
    observability: dict = Field(default_factory=dict)
    potential_access_summaries: dict[
        str, PotentialPhysicalAccessSummary] = Field(default_factory=dict)
    edges: list[AgentEdge]

    def to_networkx(self) -> nx.MultiGraph:
        graph = nx.MultiGraph()
        for region_id, region in sorted(self.regions.items()):
            graph.add_node(region_id, region=region)
        for edge in sorted(self.edges, key=lambda item: item.edge_id):
            graph.add_edge(edge.u, edge.v, key=edge.edge_id, edge=edge,
                           weight=edge.score, relation=edge.relation, hard=edge.hard)
        return graph


class DesignContext(BaseModel):
    iteration: int = 0
    timing_enabled: bool = False


class AgentGraphBuilder:
    def __init__(self, *, geometry_edges: bool = True, shared_net_edges: bool = True,
                 resource_edges: bool = True, timing_edges: bool = False,
                 proximity_only_score_cap: float = 0.34):
        self.enabled = {"geometry": geometry_edges, "shared_net": shared_net_edges,
                        "resource": resource_edges, "timing": timing_edges}
        self.proximity_only_score_cap = proximity_only_score_cap

    @staticmethod
    def _edge(u: str, v: str, relation: str, score: float, hard: bool,
              evidence: list[EdgeEvidence], iteration: int, directed: bool = False) -> AgentEdge:
        a, b = (u, v) if directed or u < v else (v, u)
        edge_id = f"edge_{relation}_{stable_hash([a, b, relation, [e.evidence_id for e in evidence]])[:16]}"
        return AgentEdge(edge_id=edge_id, u=a, v=b, relation=relation, score=min(max(score, 0), 1),
                         hard=hard, directed=directed, evidence=evidence, confidence=1.0,
                         created_iteration=iteration, last_validated_iteration=iteration)

    @staticmethod
    def _physical_dependency(
        left: PotentialPhysicalAccessSummary,
        right: PotentialPhysicalAccessSummary,
    ) -> tuple[list[EdgeEvidence], bool]:
        evidence: list[EdgeEvidence] = []
        hard = False
        legacy = (
            left.schema_version == right.schema_version == "1.0"
            and not left.access_evidence and not right.access_evidence
        )
        left_exact = (
            set(left.exact_authorized_access_ids) if not legacy else
            set(left.writable_occurrence_ids)
            | set(left.writable_source_target_ids)
            | set(left.potential_write_geometry_ids)
        )
        right_exact = (
            set(right.exact_authorized_access_ids) if not legacy else
            set(right.writable_occurrence_ids)
            | set(right.writable_source_target_ids)
            | set(right.potential_write_geometry_ids)
        )
        left_proven = (
            set(left.proven_physical_relation_ids) if not legacy else
            set(left.protected_read_geometry_ids)
            | set(left.protected_relation_ids)
            | set(left.merged_boundary_contributor_ids)
        )
        right_proven = (
            set(right.proven_physical_relation_ids) if not legacy else
            set(right.protected_read_geometry_ids)
            | set(right.protected_relation_ids)
            | set(right.merged_boundary_contributor_ids)
        )
        predicates = [
            ("same_writable_occurrence", (
                set(left.writable_occurrence_ids)
                & set(right.writable_occurrence_ids)
                & left_exact & right_exact
            )),
            ("same_writable_source_target", (
                set(left.writable_source_target_ids)
                & set(right.writable_source_target_ids)
                & left_exact & right_exact
            )),
            ("shared_merged_boundary_contributor", (
                set(left.merged_boundary_contributor_ids)
                & set(right.merged_boundary_contributor_ids)
                & left_proven & right_proven
            )),
            ("potential_write_protected_read", (
                set(left.potential_write_geometry_ids) & left_exact
                & set(right.protected_read_geometry_ids) & right_proven
            ) | (
                set(right.potential_write_geometry_ids) & right_exact
                & set(left.protected_read_geometry_ids) & left_proven
            )),
            ("shared_protected_landing_contact", (
                set(left.protected_relation_write_ids)
                & set(right.protected_relation_ids)
                & left_proven & right_proven
            ) | (
                set(right.protected_relation_write_ids)
                & set(left.protected_relation_ids)
                & left_proven & right_proven
            )),
        ]
        for kind, identities in predicates:
            if not identities:
                continue
            hard = True
            values = sorted(identities)
            evidence.append(EdgeEvidence(
                evidence_id=f"physical:{kind}:{stable_hash(values)[:12]}",
                kind=kind,
                details={
                    "identities": values,
                    "evidence_quality": (
                        "legacy_exact" if legacy else
                        "exact_or_proven_physical"
                    ),
                },
            ))
        broad = (
            set(left.broad_phase_relation_ids)
            & set(right.broad_phase_relation_ids)
        )
        if broad:
            values = sorted(broad)
            evidence.append(EdgeEvidence(
                evidence_id="physical:broad-phase:" + stable_hash(values)[:12],
                kind="broad_phase_physical_candidate",
                details={"identities": values, "evidence_quality": "broad_phase"},
            ))
        shared_unknown = (
            set(left.unknown_access_ids) & set(right.unknown_access_ids)
        )
        if shared_unknown:
            values = sorted(shared_unknown)
            evidence.append(EdgeEvidence(
                evidence_id="physical:unknown-access:" + stable_hash(values)[:12],
                kind="unknown_physical_access",
                details={"identities": values, "evidence_quality": "unknown"},
            ))
        shared_definitions = (
            set(left.source_definition_ids) & set(right.source_definition_ids)
        )
        if shared_definitions and not hard:
            values = sorted(shared_definitions)
            evidence.append(EdgeEvidence(
                evidence_id=f"physical:shared-definition:{stable_hash(values)[:12]}",
                kind="shared_source_definition_distinct_or_unknown_occurrence",
                details={
                    "source_definition_ids": values,
                    "evidence_quality": "unknown",
                },
            ))
        return evidence, hard

    def build(self, regions: list[RegionState], context: DesignContext | None = None,
              *, potential_access_summaries: dict[str, PotentialPhysicalAccessSummary] | None = None) -> AgentGraph:

        context = context or DesignContext()
        edges = []
        summaries = potential_access_summaries or {}
        ordered = sorted(regions, key=lambda item: item.region_id)
        for index, left in enumerate(ordered):
            for right in ordered[index + 1:]:
                shared_objects = set(left.editable_object_ids) & set(right.editable_object_ids)
                overlap = left.edit_halo_dbu.intersection_area(right.edit_halo_dbu)
                min_area = min(left.edit_halo_dbu.area, right.edit_halo_dbu.area)
                halo_score = overlap / min_area if min_area else 0
                gap = left.edit_halo_dbu.gap(right.edit_halo_dbu)
                href = max(min(left.edit_halo_dbu.width, right.edit_halo_dbu.width), 1)
                gap_score = exp(-gap / href)
                physical_evidence: list[EdgeEvidence] = []
                physical_hard = False
                if left.region_id in summaries and right.region_id in summaries:
                    physical_evidence, physical_hard = self._physical_dependency(
                        summaries[left.region_id], summaries[right.region_id],
                    )
                shared_object_hard = bool(shared_objects) and not summaries
                if self.enabled["geometry"] and (
                    overlap or gap_score >= .35 or shared_objects or physical_evidence
                ):
                    evidence = [EdgeEvidence(evidence_id=f"geo:{left.region_id}:{right.region_id}",
                                             kind="halo_overlap" if overlap else "halo_gap",
                                             details={"overlap_area": overlap, "gap_dbu": gap})]
                    if shared_objects:
                        evidence.append(EdgeEvidence(evidence_id=f"obj:{stable_hash(sorted(shared_objects))[:12]}",
                                                     kind="shared_editable_object",
                                                     details={"object_ids": sorted(shared_objects),
                                                              "hard_by_occurrence": shared_object_hard or physical_hard}))
                    evidence.extend(physical_evidence)
                    edges.append(self._edge(left.region_id, right.region_id, "geometry",
                                            (1.0 if (shared_object_hard or physical_hard) else min(
                                                max(halo_score, gap_score), self.proximity_only_score_cap)),
                                            shared_object_hard or physical_hard, evidence, context.iteration))
                shared_nets = set(left.net_ids) & set(right.net_ids)
                if self.enabled["shared_net"] and shared_nets:
                    exact = left.net_mapping_quality == right.net_mapping_quality == "exact"
                    evidence = [EdgeEvidence(
                        evidence_id=f"net:{stable_hash(sorted(shared_nets))[:12]}",
                        kind=(
                            "shared_named_net" if exact
                            else "shared_connectivity_component"
                        ),
                        details={
                            "net_ids": sorted(shared_nets),
                            "evidence_level": (
                                "exact" if exact else "connectivity_component"
                            ),
                            "mapping_quality_u": left.net_mapping_quality,
                            "mapping_quality_v": right.net_mapping_quality,
                        },
                    )]
                    edges.append(self._edge(left.region_id, right.region_id, "shared_net",
                                            .9 if exact else .6, exact, evidence, context.iteration))
                bins = set(left.resource_context.occupied_track_bins) & set(right.resource_context.occupied_track_bins)
                if self.enabled["resource"] and bins:
                    denom = min(len(left.resource_context.occupied_track_bins), len(right.resource_context.occupied_track_bins))
                    score = len(bins) / max(denom, 1)
                    evidence = [EdgeEvidence(evidence_id=f"resource:{stable_hash(sorted(bins))[:12]}",
                                             kind="geometry_proxy_bins", details={"bin_ids": sorted(bins)})]
                    edges.append(self._edge(left.region_id, right.region_id, "resource", score, False,
                                            evidence, context.iteration))
                if self.enabled["timing"] and context.timing_enabled and left.timing_context and right.timing_context:
                    paths = set(left.timing_context.affected_path_ids) & set(right.timing_context.affected_path_ids)
                    if paths and left.timing_context.source == right.timing_context.source == "sta":
                        evidence = [EdgeEvidence(evidence_id=f"timing:{stable_hash(sorted(paths))[:12]}",
                                                 kind="shared_sta_path", details={"path_ids": sorted(paths)})]
                        edges.append(self._edge(left.region_id, right.region_id, "timing", 1.0, False,
                                                evidence, context.iteration, directed=True))
        if not self.enabled["timing"] and any(edge.relation == "timing" for edge in edges):
            raise AssertionError("timing edge created while disabled")
        unknown = sum(len(item.unknown_reasons) for item in summaries.values())
        physical_edges = sum(any(e.evidence_id.startswith("physical:") for e in edge.evidence)
                             for edge in edges)
        evidence_counts = defaultdict(int)
        for summary in summaries.values():
            for item in summary.access_evidence:
                evidence_counts[item.category] += 1
        return AgentGraph(
            regions={region.region_id: region for region in ordered},
            potential_access_summaries=summaries,
            observability={
                "potential_access_region_count": len(summaries),
                "potential_access_unknown_count": unknown,
                "potential_physical_dependency_edge_count": physical_edges,
                "potential_access_evidence_counts": dict(sorted(
                    evidence_counts.items()
                )),
                "hard_physical_dependency_edge_count": sum(
                    edge.hard and any(
                        item.evidence_id.startswith("physical:")
                        for item in edge.evidence
                    )
                    for edge in edges
                ),
            },
            edges=sorted(edges, key=lambda item: item.edge_id))


def _aggregate_score(edges: list[AgentEdge]) -> float:
    remaining = 1.0
    for edge in edges:
        remaining *= 1 - edge.confidence * edge.score
    return 1 - remaining


class AgentGraphPruner:
    def prune(self, graph: AgentGraph, cfg: AgentGraphConfig) -> AgentGraph:
        by_relation: defaultdict[tuple[str, str], list[AgentEdge]] = defaultdict(list)
        hard = []
        from drc_agent.graphs.scopes import PlanningScopeController
        return PlanningScopeController().prune(graph, cfg)
        for edge in graph.edges:
            if edge.hard:
                hard.append(edge)
            elif edge.score >= cfg.soft_edge_threshold:
                by_relation[(edge.u, edge.relation)].append(edge)
                by_relation[(edge.v, edge.relation)].append(edge)
        accepted_ids = {edge.edge_id for edge in hard}
        for values in by_relation.values():
            for edge in sorted(values, key=lambda item: (-item.score, item.edge_id))[:cfg.max_soft_neighbors_per_relation]:
                accepted_ids.add(edge.edge_id)
        candidates = [edge for edge in graph.edges if edge.edge_id in accepted_ids]
        simple = nx.Graph()
        simple.add_nodes_from(graph.regions)
        pair_edges: defaultdict[tuple[str, str], list[AgentEdge]] = defaultdict(list)
        for edge in candidates:
            pair_edges[tuple(sorted([edge.u, edge.v]))].append(edge)
        for pair, values in pair_edges.items():
            simple.add_edge(*pair, weight=_aggregate_score(values), hard=any(edge.hard for edge in values))
        if cfg.preserve_maximum_spanning_forest:
            for component in nx.connected_components(simple):
                subgraph = simple.subgraph(component)
                for u, v in nx.maximum_spanning_edges(subgraph, weight="weight", data=False):
                    accepted_ids.update(edge.edge_id for edge in pair_edges[tuple(sorted([u, v]))])
        pruned = [edge for edge in candidates if edge.edge_id in accepted_ids]
        audit = nx.Graph()
        audit.add_nodes_from(graph.regions)
        audit.add_edges_from((edge.u, edge.v, {"hard": edge.hard, "score": edge.score}) for edge in pruned)
        for component in list(nx.connected_components(audit)):
            while len(component) > cfg.max_active_subgraph_regions:
                bridges = list(nx.bridges(audit.subgraph(component)))
                removable = []
                for u, v in bridges:
                    values = [edge for edge in pruned if {edge.u, edge.v} == {u, v}]
                    if values and not any(edge.hard for edge in values):
                        removable.append((max(edge.score for edge in values), u, v, values))
                if not removable:
                    break
                _, u, v, values = min(removable, key=lambda item: (item[0], item[1], item[2]))
                remove_ids = {edge.edge_id for edge in values}
                pruned = [edge for edge in pruned if edge.edge_id not in remove_ids]
                audit.remove_edge(u, v)
                component = max(nx.connected_components(audit.subgraph(component)), key=len)
        original_hard = {edge.edge_id for edge in graph.edges if edge.hard}
        if not original_hard.issubset({edge.edge_id for edge in pruned}):
            raise AssertionError("hard dependency was pruned")
        return AgentGraph(regions=graph.regions, edges=sorted(pruned, key=lambda item: item.edge_id))

    @staticmethod
    def active_subgraphs(graph: AgentGraph, max_regions: int = 12) -> list[AgentSubgraph]:
        nx_graph = graph.to_networkx()
        results = []
        if graph.active_scopes:
            return list(graph.active_scopes)
        # Compatibility for unpruned graphs used by older smoke tests.
        for component in sorted(nx.connected_components(nx.Graph(nx_graph)), key=lambda ids: sorted(ids)):
            region_ids = sorted(component)
            edge_ids = sorted(edge.edge_id for edge in graph.edges if edge.u in component and edge.v in component)
            results.append(AgentSubgraph(
                subgraph_id=f"subgraph_{stable_hash(region_ids)[:16]}", region_ids=region_ids,
                edge_ids=edge_ids, hierarchical=len(region_ids) > max_regions,
            ))
        return results


class MessagePasser:
    """Bounded relation-specific protocol: facts first, intent summaries second."""

    @staticmethod
    def _constraint(edge: AgentEdge, graph: AgentGraph) -> StructuredConstraint:
        left, right = graph.regions[edge.u], graph.regions[edge.v]
        if edge.relation == "geometry":
            parameters = {
                "hard": edge.hard,
                "halo_u_dbu": left.edit_halo_dbu.model_dump(mode="json"),
                "halo_v_dbu": right.edit_halo_dbu.model_dump(mode="json"),
                "shared_editable_object_ids": sorted(
                    set(left.editable_object_ids) & set(right.editable_object_ids)
                ),
                "forbidden_overlap": True,
            }
            kind = "geometry_boundary_constraint"
        elif edge.relation == "shared_net":
            parameters = {
                "hard": edge.hard,
                "shared_net_ids": sorted(set(left.net_ids) & set(right.net_ids)),
                "mapping_quality_u": left.net_mapping_quality,
                "mapping_quality_v": right.net_mapping_quality,
                "preserve_topology": True,
            }
            kind = "shared_net_topology_constraint"
        elif edge.relation == "resource":
            parameters = {
                "hard": edge.hard,
                "shared_proxy_bin_ids": sorted(
                    set(left.resource_context.occupied_track_bins) &
                    set(right.resource_context.occupied_track_bins)
                ),
                "source": "geometry_proxy",
                "capacity_known": False,
            }
            kind = "resource_geometry_proxy_constraint"
        else:
            parameters = {
                "hard": edge.hard,
                "timing_disabled": not (
                    left.timing_context is not None and right.timing_context is not None
                ),
            }
            kind = "timing_constraint"
        physical_evidence = [
            item.model_dump(mode="json") for item in edge.evidence
            if item.evidence_id.startswith("physical:")
        ]
        access = {}
        for endpoint, region_id in (("u", edge.u), ("v", edge.v)):
            summary = graph.potential_access_summaries.get(region_id)
            if summary is None:
                continue
            access[endpoint] = {
                "summary_id": summary.summary_id,
                "snapshot_id": summary.snapshot_id,
                "region_id": summary.region_id,
                "writable_occurrence_ids": summary.writable_occurrence_ids,
                "writable_source_target_ids": summary.writable_source_target_ids,
                "potential_write_geometry_ids": summary.potential_write_geometry_ids,
                "potential_write_layers": summary.potential_write_layers,
                "protected_relation_ids": summary.protected_relation_ids,
                "protected_read_geometry_ids": summary.protected_read_geometry_ids,
                "merged_boundary_contributor_ids":
                    summary.merged_boundary_contributor_ids,
                "landing_contact_relation_ids":
                    summary.landing_contact_relation_ids,
                "connectivity_evidence_quality":
                    summary.connectivity_evidence_quality,
                "unknown_reasons": summary.unknown_reasons,
            }
        if physical_evidence or access:
            parameters["physical_dependency_evidence"] = physical_evidence
            parameters["potential_physical_access"] = access

        return StructuredConstraint(kind=kind, parameters=parameters)

    def round_one(self, subgraph: AgentSubgraph, graph: AgentGraph) -> list[NeighborMessage]:
        relevant = [edge for edge in graph.edges if edge.edge_id in set(subgraph.edge_ids)]
        messages = []
        for edge in relevant:
            constraint = self._constraint(edge, graph)
            for sender, receiver in [(edge.u, edge.v), (edge.v, edge.u)]:
                content = {
                    "round": 1, "sender": sender, "receiver": receiver,
                    "relation": edge.relation, "edge": edge.edge_id,
                    "evidence_ids": [item.evidence_id for item in edge.evidence],
                    "constraints": [constraint.model_dump(mode="json")],
                }
                messages.append(NeighborMessage(
                    message_id=f"msg_{stable_hash(content)[:20]}", round=1,
                    sender_region_id=sender, receiver_region_id=receiver,
                    relation=edge.relation,
                    evidence_ids=[item.evidence_id for item in edge.evidence],
                    constraints=[constraint], requested_response=["FACT_ACK"],
                    content_hash=stable_hash(content),
                ))
        return sorted(messages, key=lambda item: item.message_id)

    def round_two(
        self, subgraph: AgentSubgraph, graph: AgentGraph,
        intents_by_region: dict[str, list] | None = None,
    ) -> list[NeighborMessage]:
        intents_by_region = intents_by_region or {}
        relevant = [edge for edge in graph.edges if edge.edge_id in set(subgraph.edge_ids)]
        messages = []
        for edge in relevant:
            constraint = self._constraint(edge, graph)
            for sender, receiver in [(edge.u, edge.v), (edge.v, edge.u)]:
                summaries = intents_by_region.get(sender, [])
                content = {
                    "round": 2, "sender": sender, "receiver": receiver,
                    "relation": edge.relation, "edge": edge.edge_id,
                    "proposed_intents": [
                        item.model_dump(mode="json") if hasattr(item, "model_dump") else item
                        for item in summaries
                    ],
                }
                messages.append(NeighborMessage(
                    message_id=f"msg_{stable_hash(content)[:20]}", round=2,
                    sender_region_id=sender, receiver_region_id=receiver,
                    relation=edge.relation,
                    evidence_ids=[item.evidence_id for item in edge.evidence],
                    constraints=[constraint], proposed_intents=summaries,
                    requested_response=["INTENT_COMPATIBILITY"],
                    content_hash=stable_hash(content),
                ))
        return sorted(messages, key=lambda item: item.message_id)

    def round_three(
        self, subgraph: AgentSubgraph, graph: AgentGraph,
        conflict_region_pairs: set[tuple[str, str]],
        intents_by_region: dict[str, list] | None = None,
    ) -> list[NeighborMessage]:
        intents_by_region = intents_by_region or {}
        messages = []
        for message in self.round_two(subgraph, graph, intents_by_region):
            pair = tuple(sorted((message.sender_region_id, message.receiver_region_id)))
            if pair not in conflict_region_pairs:
                continue
            content = {
                "round": 3, "sender": message.sender_region_id,
                "receiver": message.receiver_region_id,
                "relation": message.relation,
                "evidence_ids": message.evidence_ids,
            }
            messages.append(message.model_copy(update={
                "message_id": f"msg_{stable_hash(content)[:20]}",
                "round": 3,
                "requested_response": ["COORDINATED_REVISION_OR_NO_OP"],
                "content_hash": stable_hash(content),
            }))
        return sorted(messages, key=lambda item: item.message_id)

    def run(
        self, subgraph: AgentSubgraph, graph: AgentGraph, rounds: int = 2,
        intents_by_region: dict[str, list] | None = None,
        conflict_region_pairs: set[tuple[str, str]] | None = None,
    ) -> list[NeighborMessage]:
        rounds = min(max(rounds, 0), 3)
        messages = self.round_one(subgraph, graph) if rounds >= 1 else []
        if rounds >= 2:
            messages.extend(self.round_two(subgraph, graph, intents_by_region))
        if rounds >= 3 and conflict_region_pairs:
            messages.extend(self.round_three(
                subgraph, graph, conflict_region_pairs, intents_by_region,
            ))
        return messages

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path

from pydantic import Field

from drc_agent.backends.dac26 import DAC26ReportAdapter
from drc_agent.backends.klayout import KLayoutBackend
from drc_agent.config.loader import AppConfig
from drc_agent.experience.store import NullExperienceStore
from drc_agent.graphs.agent import AgentGraphBuilder
from drc_agent.regions.builder import RegionBuilder
from drc_agent.regions.connectivity import map_connectivity_components
from drc_agent.regions.parser import (
    SourceObjectMapper, ViolationParser, load_rule_catalog,
)
from drc_agent.regions.physical import SourceHierarchyPhysicalBuilder
from drc_agent.regions.topology import LocalTopologyBuilder
from drc_agent.schemas.action import DesignState
from drc_agent.schemas.state import AgentSubgraph
from drc_agent.schemas.workflow import DesignSnapshotRef
from drc_agent.utils.artifacts import ArtifactStore
from drc_agent.workflow.planning import SubgraphPlanner
from drc_agent.workflow.sandbox_runtime import RuntimeSandboxService
from drc_agent.rules.catalog import RulePredicateRegistry
from drc_agent.rules.witness import RuleWitnessBuilder
from drc_agent.schemas.common import StrictModel, file_sha256, utc_now


DEFAULT_RULES = (
    "M2.S.7", "V2.M3.AUX.2", "V1.M1.EN.1", "M4.AUX.1", "M4.AUX.2",
)
LIVE_DEFAULT_RULES = (
    "M2.S.7", "V2.M3.AUX.2", "V4.M5.AUX.2",
    "V0.M1.AUX.3", "V1.M1.EN.1", "M4.AUX.1",
    "M4.AUX.2", "M1.S.2", "M4.S.5",
)
RUN4_POSITIVE_CANDIDATE = "candidate_d40b236a8a769b8b8524"
RUN4_POSITIVE_VIOLATION = "vio_e29580ba307c0ba0"


class RuleKernelMetrics(StrictModel):
    rule_id: str
    extracted_violation_count: int = 0
    witness_id: str | None = None
    witness_evidence_quality: str = "UNAVAILABLE"
    witness_predicate_fidelity: str = "UNAVAILABLE"
    query_count: int = 0
    repair_program_count: int = 0
    duplicate_physical_effect_count: int = 0
    compile_failure_count: int = 0
    check_failure_count: int = 0
    predicate_probe_pass_count: int = 0
    predicate_probe_reject_count: int = 0
    tool_runtime_seconds: float = 0.0
    participating_physical_geometry_count: int = 0
    editable_contributor_count: int = 0
    connectivity_quality: str = "unavailable"
    semantic_intent_count: int = 0
    compiled_repair_program_count: int = 0
    valid_candidate_count: int = 0
    preview_clean_count: int = 0
    sandbox_clean_count: int = 0
    target_violation_clean_count: int = 0
    new_violation_count: int = 0
    connectivity_failure_count: int = 0
    revision_count: int = 0
    llm_calls: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_calls_per_clean_repair: float | None = None
    llm_tokens_per_clean_repair: float | None = None
    evidence_mode: str = "RECORDED_REAL_SANDBOX_REPLAY"
    clean_candidate_ids: list[str] = Field(default_factory=list)
    verification_artifact_paths: list[str] = Field(default_factory=list)
    unresolved_stage: str | None = None


class RepairKernelReport(StrictModel):
    schema_version: str = "1.0"
    generated_at: str
    development_only: bool = True
    mode: str
    source_run_id: str
    source_run_path: str
    frozen_deck_path: str
    frozen_deck_sha256: str
    source_snapshot_script: str
    source_snapshot_drc: str
    facts: dict
    rules: list[RuleKernelMetrics]
    totals: dict
    notes: list[str] = Field(default_factory=list)


def _json_lines(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            values.append(json.loads(line))
    return values


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class RepairKernelMicrobenchmark:
    """Fail-closed replay over real planning and KLayout sandbox artifacts.

    Replay mode deliberately does not claim that legacy run4 candidates used
    the new RepairProgram compiler. It rebuilds Graph Action v3 witnesses from
    the frozen source snapshot, then audits only recorded sandbox evidence with
    attempted source/GDS/DRC hashes and connectivity results. A later live mode
    can use the same report schema without conflating ordinary tests with EDA
    evidence.
    """

    # Phase 2 supersedes this module's legacy live path with
    # development.repair_kernel_bench. Replay remains compatibility-only.

    def __init__(self, project_root: Path):
        self.project_root = project_root.resolve()

    @staticmethod
    def live_gate(report: RepairKernelReport) -> dict:
        if report.mode != "LIVE_ISOLATED_REPAIR_KERNEL":
            return {
                "passed": False,
                "reason": "REPLAY_IS_NOT_LIVE_EVIDENCE",
                "clean_rule_families": [],
            }
        clean = [
            item for item in report.rules
            if item.target_violation_clean_count > 0
            and item.new_violation_count == 0
            and item.connectivity_failure_count == 0
            and item.sandbox_clean_count > 0
        ]
        families = sorted(set(
            report.facts.get("rule_families", {}).get(item.rule_id)
            for item in clean
            if report.facts.get("rule_families", {}).get(item.rule_id)
        ))
        positive = any(
            item.rule_id == "M2.S.7" and item.sandbox_clean_count > 0
            for item in clean
        )
        duplicates = sum(
            item.duplicate_physical_effect_count for item in report.rules
        )
        replay_safe = bool(report.facts.get(
            "iteration2_replay_regression_passed", False
        ))
        passed = positive and len(families) >= 3 and not duplicates and replay_safe
        return {
            "passed": passed, "positive_control": positive,
            "clean_rule_families": families,
            "duplicate_physical_effect_count": duplicates,
            "iteration2_replay_regression_passed": replay_safe,
            "reason": None if passed else "LIVE_GATE_NOT_MET",
        }

    def run_live_case(
        self, *, case_id: str, run_id: str, config: AppConfig,
        llm, rules: tuple[str, ...] = LIVE_DEFAULT_RULES,
        samples_per_rule: int = 2,
    ) -> RepairKernelReport:
        """Run isolated real-EDA Region kernels without a master transaction.

        The caller must provide an explicit audited LLM client/config. This
        method never resolves credentials implicitly and never invokes the
        full ResearchRuntime workflow.
        """
        if llm is None or not config.llm.enabled:
            raise ValueError(
                "live microbenchmark requires an explicit enabled LLM client"
            )
        if config.features.experience_graph:
            raise ValueError("live B5 kernel requires Experience Graph OFF")
        if samples_per_rule < 1:
            raise ValueError("samples_per_rule must be positive")
        return asyncio.run(self._run_live_case(
            case_id=case_id, run_id=run_id, config=config, llm=llm,
            rules=rules, samples_per_rule=samples_per_rule,
        ))

    async def _run_live_case(
        self, *, case_id: str, run_id: str, config: AppConfig,
        llm, rules: tuple[str, ...], samples_per_rule: int,
    ) -> RepairKernelReport:
        benchmark = (
            self.project_root / "benchmarks/EvoDRC/DAC26_DRC_Benchmark"
        )
        block = benchmark / "testcase/asap7/block"
        paths = {
            "script": block / f"layout_script/{case_id}.py",
            "gds": block / f"gds/{case_id}.gds",
            "drc": block / f"drc_report/{case_id}.drc.json",
            "connectivity": block / f"connectivity/{case_id}.json",
            "rule_deck": benchmark / "testcase/asap7/asap7.lydrc",
            "sanity_helper": benchmark / "evaluator/sanity_check.py",
            "connectivity_helper": (
                benchmark / "evaluator/check_connectivity.py"
            ),
        }
        missing = [str(value) for value in paths.values() if not value.is_file()]
        if missing:
            raise FileNotFoundError(f"live microbenchmark inputs missing: {missing}")
        run_dir = self.project_root / "runs" / run_id
        if run_dir.exists():
            raise FileExistsError(f"microbenchmark run already exists: {run_dir}")
        store = ArtifactStore(run_dir)
        script_ref = store.copy(
            paths["script"], f"baseline/{case_id}.py",
            producer="live_microbenchmark", media_type="text/x-python",
        )
        gds_ref = store.copy(
            paths["gds"], f"baseline/{case_id}.gds",
            producer="live_microbenchmark", media_type="application/gds",
        )
        drc_ref = store.copy(
            paths["drc"], f"baseline/{case_id}.drc.json",
            producer="live_microbenchmark", media_type="application/json",
        )
        connectivity_ref = store.copy(
            paths["connectivity"], f"baseline/{case_id}.connectivity.json",
            producer="live_microbenchmark", media_type="application/json",
        )
        snapshot = DesignSnapshotRef(
            snapshot_id="snapshot_" + file_sha256(paths["script"])[:20],
            script_ref=script_ref, gds_ref=gds_ref, drc_ref=drc_ref,
            connectivity_ref=connectivity_ref,
        )
        catalog = load_rule_catalog(
            self.project_root / "configs/rules/asap7_rule_catalog.yaml"
        )
        violations = ViolationParser().parse_dac26_json(
            paths["drc"], case_id=case_id, rule_catalog=catalog,
        )
        objects = SourceObjectMapper().build_map(paths["script"])
        objects, connectivity_mapping = map_connectivity_components(
            objects, paths["connectivity"],
        )
        registry = RulePredicateRegistry(
            paths["rule_deck"], require_frozen_hash=True,
        )
        supported_rules = set(rules) & registry.supported_rule_ids
        supported = [
            item for item in violations if item.rule_id in supported_rules
        ]
        layers = {
            layer for item in supported
            for layer in registry.get(item.rule_id).predicate.involved_layers
        }
        physical = SourceHierarchyPhysicalBuilder().build(
            paths["script"], objects, layer_filter=layers,
        )
        witness_builder = RuleWitnessBuilder(registry)
        witnesses = [
            witness_builder.build(item, physical, objects)
            for item in supported
        ]
        witness_map = {
            item.violation_id: item.model_dump(mode="json")
            for item in witnesses
        }
        packs = {
            rule_id: registry.get(rule_id).model_dump(mode="json")
            for rule_id in sorted(supported_rules)
        }
        design = DesignState(
            case_id=case_id, legal_layers={item.layer for item in objects},
            objects={
                item.object_id: item.model_dump(mode="json") for item in objects
            },
            violations={
                item.violation_id: item.model_dump(mode="json")
                for item in violations
            },
            local_topology=LocalTopologyBuilder(
                dbu_per_um=config.region.dbu_per_um,
            ).build(violations, objects).model_dump(mode="json"),
            physical_geometries={
                item.geometry_id: item.model_dump(mode="json")
                for item in physical
            },
            rule_predicates={
                rule_id: value["predicate"] for rule_id, value in packs.items()
            },
            rule_witnesses=witness_map, rule_knowledge_packs=packs,
            manufacturing_grid_dbu=config.backend.manufacturing_grid_dbu,
            dbu_per_um=config.region.dbu_per_um,
            source_hashes={
                item.source_anchor_id: item.source_span.source_hash
                for item in objects
                if item.source_anchor_id and item.source_span
            },
        )
        built = RegionBuilder(config.region).build(
            violations, objects, catalog,
        )
        by_rule = defaultdict(list)
        for region in built.regions:
            for rule_id in region.rule_ids:
                if rule_id in supported_rules:
                    by_rule[rule_id].append(region)
        selected = []
        selected_ids = set()
        for rule_id in rules:
            for region in by_rule.get(rule_id, [])[:samples_per_rule]:
                if region.region_id not in selected_ids:
                    selected.append(region)
                    selected_ids.add(region.region_id)

        backend = KLayoutBackend(
            self.project_root, image=config.backend.image,
            stage_root=config.backend.stage_root,
            artifact_retention_mode=config.backend.artifact_retention_mode,
            container_workspace_mode=config.backend.container_workspace_mode,
        )
        adapter = DAC26ReportAdapter(benchmark)
        service = RuntimeSandboxService(
            config=config, backend=backend, adapter=adapter, paths=paths,
            run_id=run_id, case_id=case_id, run_store=store,
            experience_store=NullExperienceStore(),
            baseline_snapshot=lambda: snapshot,
            current_snapshot=lambda: snapshot,
            rule_deck_hash=file_sha256(paths["rule_deck"]),
            evaluator_hash=adapter.evaluator_hash,
        )
        planner = SubgraphPlanner(
            config=config, llm=llm,
            experience_store=NullExperienceStore(),
            artifact_store=store, run_id=run_id,
            candidate_sandbox=service.candidate,
            bundle_sandbox=None,
        )
        plans = []
        runtimes = defaultdict(float)
        try:
            for index, region in enumerate(selected):
                graph = AgentGraphBuilder(
                    geometry_edges=False, shared_net_edges=False,
                    resource_edges=False,
                ).build([region])
                subgraph = AgentSubgraph(
                    subgraph_id=f"kernel_{index:03d}_{region.region_id}",
                    region_ids=[region.region_id], edge_ids=[],
                    physical_scope_id=region.region_id,
                )
                started = time.monotonic()
                plan = await planner.plan(
                    iteration=1, subgraph=subgraph, graph=graph,
                    violations=violations, objects=objects, design=design,
                    catalog=catalog, script=paths["script"],
                    current_snapshot_id=snapshot.snapshot_id,
                )
                elapsed = time.monotonic() - started
                plans.append(plan)
                for rule_id in region.rule_ids:
                    runtimes[rule_id] += elapsed
        finally:
            service.close()

        witness_by_rule = defaultdict(list)
        for witness in witnesses:
            witness_by_rule[witness.rule_id].append(witness)
        rule_results = []
        families = {}
        for rule_id in rules:
            local_plans = [
                plan for plan in plans
                if rule_id in plan.subgraph.region_ids
                or any(
                    rule_id in built_region.rule_ids
                    for region_id in plan.subgraph.region_ids
                    for built_region in built.regions
                    if built_region.region_id == region_id
                )
            ]
            local_candidates = [
                candidate for plan in local_plans for candidate in plan.candidates
                if not candidate.is_noop
                and any(
                    (design.violations.get(identifier) or {}).get("rule_id")
                    == rule_id
                    for identifier in candidate.target_violation_ids
                )
            ]
            traces = [
                trace for plan in local_plans
                for trace in plan.repair_programming_traces
                if any(
                    (design.violations.get(identifier) or {}).get("rule_id")
                    == rule_id
                    for identifier in trace.target_violation_ids
                )
            ]
            attempts = [item for trace in traces for item in trace.attempts]
            evidence = [
                candidate.benefit_evidence.sandbox
                for candidate in local_candidates
                if candidate.benefit_evidence.sandbox is not None
            ]
            clean = [
                candidate for candidate in local_candidates
                if candidate.benefit_evidence.sandbox.status.value
                == "CLEAN_PROGRESS"
            ]
            target_ids = {
                item.violation_id for item in supported
                if item.rule_id == rule_id
            }
            target_clean = sum(
                len(
                    set(candidate.benefit_evidence.sandbox
                        .target_violation_removed_ids) & target_ids
                )
                for candidate in clean
            )
            first = next(iter(witness_by_rule.get(rule_id, [])), None)
            spec, _ = catalog.lookup(rule_id)
            families[rule_id] = spec.family.value
            rule_results.append(RuleKernelMetrics(
                rule_id=rule_id,
                extracted_violation_count=len(witness_by_rule.get(rule_id, [])),
                witness_id=first.witness_id if first else None,
                witness_evidence_quality=(
                    first.evidence_quality.value if first else "UNAVAILABLE"
                ),
                witness_predicate_fidelity=(
                    first.predicate_fidelity.value if first else "UNAVAILABLE"
                ),
                query_count=sum(trace.query_steps for trace in traces),
                repair_program_count=len(attempts),
                duplicate_physical_effect_count=sum(
                    trace.duplicate_physical_effect_count for trace in traces
                ),
                compiled_repair_program_count=sum(
                    item.candidate_id is not None for item in attempts
                ),
                compile_failure_count=sum(
                    item.feedback.status == "COMPILE_FAIL" for item in attempts
                ),
                check_failure_count=sum(
                    item.feedback.status == "CHECK_FAIL" for item in attempts
                ),
                predicate_probe_pass_count=sum(
                    item.feedback.verification_artifact_path is not None
                    and item.feedback.before_rule_witness is not None
                    for item in attempts
                ),
                predicate_probe_reject_count=sum(
                    item.feedback.status == "PREDICATE_STILL_VIOLATED"
                    for item in attempts
                ),
                tool_runtime_seconds=round(runtimes[rule_id], 3),
                participating_physical_geometry_count=(
                    len(first.participating_physical_geometry_ids)
                    if first else 0
                ),
                editable_contributor_count=(
                    len(first.editable_contributor_ids) if first else 0
                ),
                connectivity_quality=(
                    first.connectivity_evidence.quality
                    if first else "unavailable"
                ),
                semantic_intent_count=sum(
                    len(result.intents)
                    for plan in local_plans for result in plan.region_results
                    if rule_id in next(
                        region.rule_ids for region in built.regions
                        if region.region_id == result.region_id
                    )
                ),
                valid_candidate_count=len(local_candidates),
                preview_clean_count=len(clean),
                sandbox_clean_count=len(clean),
                target_violation_clean_count=target_clean,
                new_violation_count=sum(
                    item.new_violation_count for item in evidence
                ),
                connectivity_failure_count=sum(
                    item.connectivity_preserved is False for item in evidence
                ),
                revision_count=sum(
                    max(0, len(trace.attempts) - 1) for trace in traces
                ),
                llm_calls=sum(trace.llm_calls for trace in traces),
                evidence_mode=(
                    "LEGACY_LIVE_ISOLATED_REPAIR_KERNEL_UNATTESTED"
                ),
                clean_candidate_ids=[
                    item.candidate_id for item in clean
                ],
                verification_artifact_paths=[
                    item.benefit_evidence.sandbox.verification_ref.path
                    for item in clean
                    if item.benefit_evidence.sandbox.verification_ref
                ],
                unresolved_stage=(
                    None if clean else
                    "program_synthesis" if not local_candidates else
                    "sandbox_DRC"
                ),
            ))
        report = RepairKernelReport(
            generated_at=utc_now().isoformat(),
            mode="LEGACY_LIVE_ISOLATED_REPAIR_KERNEL_UNATTESTED",
            source_run_id=run_id,
            source_run_path=str(run_dir),
            frozen_deck_path=str(paths["rule_deck"]),
            frozen_deck_sha256=file_sha256(paths["rule_deck"]),
            source_snapshot_script=str(paths["script"]),
            source_snapshot_drc=str(paths["drc"]),
            facts={
                "case_id": case_id,
                "rule_families": families,
                "selected_region_count": len(selected),
                "samples_per_rule": samples_per_rule,
                "experience_graph": False,
                "master_transaction": False,
                "iteration2_replay_regression_passed": None,
                "iteration2_replay_regression_evidence": "NOT_RUN",
                "connectivity_mapping": (
                    connectivity_mapping.model_dump(mode="json")
                ),
            },
            rules=rule_results,
            totals={
                "semantic_intents": sum(
                    item.semantic_intent_count for item in rule_results
                ),
                "compiled_repair_programs": sum(
                    item.compiled_repair_program_count for item in rule_results
                ),
                "sandbox_trials": service.metrics["trials"],
                "sandbox_clean": service.metrics["clean_progress"],
                "llm_calls": sum(item.llm_calls for item in rule_results),
            },
            notes=[
                "Live mode runs one bounded Region at a time against frozen inputs.",
                "All geometry edits pass compiler/checker and isolated KLayout/connectivity.",
                "No master transaction or full Block workflow is executed.",
            ],
        )
        gate = self.live_gate(report)
        report.facts["live_gate"] = gate
        store.write_json(
            "analysis/repair_kernel_live.json", report,
            producer="RepairKernelMicrobenchmark.run_live_case",
            schema_name="RepairKernelReport", schema_version="1.0",
        )
        store.write_json(
            "manifest.json", {
                "run_id": run_id, "case_id": case_id,
                "status": "COMPLETED_MICROBENCHMARK",
                "mode": report.mode, "live_gate": gate,
            },
            producer="RepairKernelMicrobenchmark.run_live_case",
            schema_name="DevelopmentMicrobenchmarkManifest",
        )
        store.write_sha256sums()
        return report

    def replay_run4(
        self, run_dir: Path, *, rules: tuple[str, ...] = DEFAULT_RULES,
    ) -> RepairKernelReport:
        run_dir = run_dir.resolve()
        manifest = _json(run_dir / "manifest.json")
        benchmark = (
            self.project_root
            / "benchmarks/EvoDRC/DAC26_DRC_Benchmark"
        )
        block = benchmark / "testcase/asap7/block"
        script = block / "layout_script/Block1.py"
        baseline_drc = run_dir / "baseline/Block1.drc.json"
        deck = benchmark / "testcase/asap7/asap7.lydrc"
        catalog = load_rule_catalog(
            self.project_root / "configs/rules/asap7_rule_catalog.yaml"
        )
        violations = ViolationParser().parse_dac26_json(
            baseline_drc, case_id="Block1", rule_catalog=catalog,
        )
        source_objects = SourceObjectMapper().build_map(script)
        selected_layers = set()
        registry = RulePredicateRegistry(deck, require_frozen_hash=True)
        for rule_id in rules:
            selected_layers.update(
                registry.get(rule_id).predicate.involved_layers
            )
        physical = SourceHierarchyPhysicalBuilder().build(
            script, source_objects, top_cell="cell_Block1",
            layer_filter=selected_layers,
        )
        witness_builder = RuleWitnessBuilder(registry)
        violation_by_id = {item.violation_id: item for item in violations}
        ids_by_rule = defaultdict(set)
        for violation in violations:
            ids_by_rule[violation.rule_id].add(violation.violation_id)

        # Include both iteration snapshots because run4 iteration 2 started
        # after the first strict commit and therefore has fresh marker IDs.
        region_rules: dict[str, set[str]] = defaultdict(set)
        for path in sorted(run_dir.glob("iterations/iter_*/state/regions.json")):
            state_violations_path = path.with_name("violations.json")
            state_violations = (
                _json(state_violations_path)
                if state_violations_path.is_file() else []
            )
            state_rule = {
                item["violation_id"]: item["rule_id"]
                for item in state_violations
            }
            for item in _json(path):
                for violation_id in item.get("violation_ids", []):
                    if violation_id in state_rule:
                        region_rules[item["region_id"]].add(
                            state_rule[violation_id]
                        )
                        ids_by_rule[state_rule[violation_id]].add(violation_id)

        intents = []
        candidates = []
        for path in sorted(run_dir.glob(
            "iterations/iter_*/planning/candidate_intents.jsonl"
        )):
            intents.extend(_json_lines(path))
        for path in sorted(run_dir.glob(
            "iterations/iter_*/planning/candidates.jsonl"
        )):
            candidates.extend(_json_lines(path))
        llm_calls = _json_lines(run_dir / "logs/llm_calls.jsonl")

        results = []
        clean_all = set()
        for rule_id in rules:
            rule_ids = ids_by_rule[rule_id]
            local_intents = [
                item for item in intents
                if rule_ids & set(item.get("target_violation_ids", []))
            ]
            local_candidates = [
                item for item in candidates
                if rule_ids & set(item.get("target_violation_ids", []))
            ]
            valid = [
                item for item in local_candidates
                if item.get("validation_status") in {
                    "VALID", "VALID_REQUIRES_SANDBOX",
                }
            ]
            clean = []
            new_count = 0
            connectivity_failures = 0
            target_clean = 0
            verification_paths = []
            for item in valid:
                evidence = (item.get("benefit_evidence") or {}).get(
                    "sandbox"
                ) or {}
                if evidence.get("connectivity_preserved") is False:
                    connectivity_failures += 1
                new_count += int(evidence.get("new_violation_count") or 0)
                if evidence.get("status") != "CLEAN_PROGRESS":
                    continue
                required = (
                    evidence.get("attempted_script_sha256"),
                    evidence.get("attempted_gds_sha256"),
                    evidence.get("attempted_drc_sha256"),
                )
                verification = evidence.get("verification_ref") or {}
                path = Path(verification.get("path", ""))
                if not all(required) or not path.is_file():
                    continue
                if verification.get("sha256") != file_sha256(path):
                    continue
                clean.append(item)
                clean_all.add(item["candidate_id"])
                verification_paths.append(str(path))
                target_clean += len(
                    set(evidence.get("target_violation_removed_ids", []))
                    & rule_ids
                )
            local_calls = [
                item for item in llm_calls
                if rule_id in region_rules.get(
                    (item.get("context") or {}).get("region_id", ""), set()
                )
            ]
            input_tokens = sum(
                int((item.get("usage") or {}).get("input_tokens") or 0)
                for item in local_calls
            )
            output_tokens = sum(
                int((item.get("usage") or {}).get("output_tokens") or 0)
                for item in local_calls
            )
            first = next(
                (item for item in violations if item.rule_id == rule_id), None
            )
            witness = (
                witness_builder.build(first, physical, source_objects)
                if first is not None else None
            )
            unresolved = None
            if not local_intents:
                unresolved = "program_synthesis"
            elif not valid:
                unresolved = "compilation"
            elif not clean:
                statuses = {
                    ((item.get("benefit_evidence") or {}).get("sandbox") or {})
                    .get("status", "NOT_RUN")
                    for item in valid
                }
                unresolved = (
                    "connectivity" if connectivity_failures > 0
                    else "sandbox_DRC"
                )
            count = len(clean)
            results.append(RuleKernelMetrics(
                rule_id=rule_id,
                extracted_violation_count=len(ids_by_rule[rule_id]),
                witness_id=witness.witness_id if witness else None,
                witness_evidence_quality=(
                    witness.evidence_quality.value if witness else "UNAVAILABLE"
                ),
                witness_predicate_fidelity=(
                    witness.predicate_fidelity.value
                    if witness else "UNAVAILABLE"
                ),
                participating_physical_geometry_count=(
                    len(witness.participating_physical_geometry_ids)
                    if witness else 0
                ),
                editable_contributor_count=(
                    len(witness.editable_contributor_ids) if witness else 0
                ),
                connectivity_quality=(
                    witness.connectivity_evidence.quality
                    if witness else "unavailable"
                ),
                semantic_intent_count=len(local_intents),
                compiled_repair_program_count=0,
                valid_candidate_count=len(valid),
                preview_clean_count=len(clean),
                sandbox_clean_count=len(clean),
                target_violation_clean_count=target_clean,
                new_violation_count=new_count,
                connectivity_failure_count=connectivity_failures,
                revision_count=0,
                llm_calls=len(local_calls),
                llm_input_tokens=input_tokens,
                llm_output_tokens=output_tokens,
                llm_calls_per_clean_repair=(
                    len(local_calls) / count if count else None
                ),
                llm_tokens_per_clean_repair=(
                    (input_tokens + output_tokens) / count if count else None
                ),
                clean_candidate_ids=[item["candidate_id"] for item in clean],
                verification_artifact_paths=verification_paths,
                unresolved_stage=unresolved,
            ))

        positive = next(
            item for item in candidates
            if item.get("candidate_id") == RUN4_POSITIVE_CANDIDATE
        )
        positive_evidence = positive["benefit_evidence"]["sandbox"]
        transaction = _json(
            run_dir / "iterations/iter_0001/transaction/transaction.json"
        )
        verification = _json(
            run_dir / "iterations/iter_0001/verification/verification.json"
        )
        iteration_metrics = _json_lines(
            run_dir / "metrics/iteration_metrics.jsonl"
        )
        baseline_metric = next(
            item for item in iteration_metrics if item["iteration"] == 0
        )
        facts = {
            "baseline": baseline_metric["committed_total_drv"],
            "iteration_1_attempted": verification["residual_violation_count"],
            "iteration_1_committed": verification["residual_violation_count"],
            "removed_original": verification["removed_original_count"],
            "new": verification["new_violation_count"],
            "connectivity": verification["connectivity_preserved"],
            "selected_candidate": positive["candidate_id"],
            "target_violation": RUN4_POSITIVE_VIOLATION,
            "rule_id": violation_by_id[RUN4_POSITIVE_VIOLATION].rule_id,
            "sandbox_status": positive_evidence["status"],
            "transaction_status": transaction["status"],
            "iteration_2_started": (
                run_dir / "iterations/iter_0002/planning/candidates.jsonl"
            ).is_file(),
        }
        if facts != {
            "baseline": 244, "iteration_1_attempted": 243,
            "iteration_1_committed": 243, "removed_original": 1,
            "new": 0, "connectivity": True,
            "selected_candidate": RUN4_POSITIVE_CANDIDATE,
            "target_violation": RUN4_POSITIVE_VIOLATION,
            "rule_id": "M2.S.7", "sandbox_status": "CLEAN_PROGRESS",
            "transaction_status": "COMMITTED", "iteration_2_started": True,
        }:
            raise AssertionError(f"run4 positive-control drift: {facts}")

        total_input = sum(
            int((item.get("usage") or {}).get("input_tokens") or 0)
            for item in llm_calls
        )
        total_output = sum(
            int((item.get("usage") or {}).get("output_tokens") or 0)
            for item in llm_calls
        )
        return RepairKernelReport(
            generated_at=utc_now().isoformat(),
            mode="RECORDED_REAL_SANDBOX_REPLAY",
            source_run_id=manifest["run_id"],
            source_run_path=str(run_dir),
            frozen_deck_path=str(deck),
            frozen_deck_sha256=file_sha256(deck),
            source_snapshot_script=str(script),
            source_snapshot_drc=str(baseline_drc),
            facts=facts, rules=results,
            totals={
                "semantic_intents": len(intents),
                "legacy_candidates": len(candidates),
                "compiled_repair_programs": 0,
                "verified_clean_candidate_ids": sorted(clean_all),
                "llm_calls": len(llm_calls),
                "llm_input_tokens": total_input,
                "llm_output_tokens": total_output,
            },
            notes=[
                "This mode rebuilds RuleWitness from the frozen Block1 source snapshot.",
                "CLEAN counts require recorded isolated sandbox hashes, connectivity=true, and a hash-valid verification artifact.",
                "run4 predates RepairProgram v3; compiled_repair_program_count and revision_count are therefore zero, not inferred.",
                "No new paid LLM or full Block1 run was started.",
            ],
        )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="repair-kernel-microbenchmark")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = RepairKernelMicrobenchmark(args.project_root).replay_run4(
        args.run_dir
    )
    payload = report.model_dump_json(indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

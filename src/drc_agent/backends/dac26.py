from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel

from drc_agent.schemas.common import ArtifactRef, stable_hash


class DAC26CaseMeta(BaseModel):
    case_id: str
    design_type: str = "block"
    layout_script: Path | None = None


class OfficialRepairScore(BaseModel):
    repair_rate: float
    new_violation_rate: float
    original_violations: int
    final_violations: int
    removed_violations: int
    new_violations: int
    original_rules_violated: int
    final_rules_violated: int
    evaluator_hash: str


def _load_frozen_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class DAC26ReportAdapter:
    def __init__(
        self,
        benchmark_root: Path,
        *,
        runtime_roots: Iterable[Path] | None = None,
    ):
        self.benchmark_root = benchmark_root.resolve(strict=True)
        self.project_root = self.benchmark_root.parents[2]
        self.runtime_roots = tuple(
            path.resolve()
            for path in (
                runtime_roots
                if runtime_roots is not None
                else (Path("/tmp/2510878-drc-agent"),)
            )
        )
        self.converter_path = self.benchmark_root / "evaluator" / "process_klayout_reports.py"
        self.scorer_path = self.benchmark_root / "evaluator" / "score_repair.py"

    @property
    def evaluator_hash(self) -> str:
        files = sorted((self.benchmark_root / "evaluator").glob("*.py"))
        return stable_hash({path.name: ArtifactRef.from_path(
            path, producer="DAC26ReportAdapter", media_type="text/x-python").sha256 for path in files})

    def convert_lyrpt_to_json(self, lyrpt: ArtifactRef, case_meta: DAC26CaseMeta,
                              output_path: Path) -> ArtifactRef:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        allowed_roots = [
            self.project_root, output_path.parent, *self.runtime_roots,
        ]
        report_path = lyrpt.verify(allowed_roots)
        module = _load_frozen_module("drc_agent_frozen_process_klayout_reports", self.converter_path)
        module.process_single_file(
            str(report_path), str(output_path), case_meta.case_id, case_meta.design_type,
            layout_script_path=str(case_meta.layout_script) if case_meta.layout_script else None,
        )
        return ArtifactRef.from_path(output_path, producer="DAC26ReportAdapter.convert_lyrpt_to_json",
                                     media_type="application/json", schema_name="DAC26DRCReport",
                                     schema_version="1.0")

    def score_repair(self, original_drc_json: ArtifactRef, repaired_report: ArtifactRef) -> OfficialRepairScore:
        allowed_roots = [self.project_root, *self.runtime_roots]
        original = original_drc_json.verify(allowed_roots)
        repaired = repaired_report.verify(allowed_roots)
        module = _load_frozen_module("drc_agent_frozen_score_repair", self.scorer_path)
        result = module.score_repair(str(original), str(repaired))
        result["evaluator_hash"] = self.evaluator_hash
        return OfficialRepairScore.model_validate(result)

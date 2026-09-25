from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class RunSpec(BaseModel):
    run_id: str
    experiment_id: str
    case_id: str
    method: str
    seed: int
    max_iterations: int


def expand_matrix(path: Path) -> list[RunSpec]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    experiment_id = raw["experiment_id"]
    results = []
    for case_id in raw["cases"]:
        for method in raw["methods"]:
            for seed in raw["seeds"]:
                results.append(RunSpec(
                    run_id=f"{experiment_id}-{method.lower()}-{case_id.lower()}-seed{seed}",
                    experiment_id=experiment_id, case_id=case_id, method=method, seed=seed,
                    max_iterations=int(raw["budgets"]["max_iterations"]),
                ))
    ids = [item.run_id for item in results]
    if len(ids) != len(set(ids)):
        raise ValueError("experiment matrix generated duplicate run IDs")
    return results


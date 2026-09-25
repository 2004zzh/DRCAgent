#!/usr/bin/env python3
"""Public launcher for the complete method and its two documented ablations."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "complete": "configs/experiments/release/complete/Block{case}.yaml",
    "without-experience": (
        "configs/experiments/p5_ablations_20260917/"
        "block{case}_without_experience.yaml"
    ),
    "without-region-messaging": (
        "configs/experiments/p5_ablations_20260917/"
        "block{case}_without_region_messaging.yaml"
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--case", choices=[f"Block{i}" for i in range(1, 7)], required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--extend-max-iterations", type=int)
    args = parser.parse_args()

    if args.verify_only and not args.resume:
        parser.error("--verify-only requires --resume")
    if args.extend_max_iterations is not None and not args.resume:
        parser.error("--extend-max-iterations requires --resume")
    if not args.verify_only and not os.environ.get("SILICONFLOW_API_KEY"):
        parser.error("SILICONFLOW_API_KEY is not set")

    for relative in ("runs", "tmp/eda", "tmp/global-resource"):
        (ROOT / relative).mkdir(parents=True, exist_ok=True)

    command = [sys.executable, "-m", "drc_agent.cli"]
    if args.resume:
        command.extend([
            "resume", "--config", "configs/release_locator.yaml",
            "--run-id", args.run_id,
        ])
        if args.verify_only:
            command.append("--verify-only")
        if args.extend_max_iterations is not None:
            command.extend([
                "--extend-max-iterations", str(args.extend_max_iterations),
            ])
    else:
        case_number = args.case.removeprefix("Block")
        profile = ROOT / VARIANTS[args.variant].format(case=case_number)
        command.extend([
            "run", "--config", "configs/base.yaml",
            "--experiment-config", str(profile.relative_to(ROOT)),
            "--run-id", args.run_id,
        ])

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    return subprocess.run(command, cwd=ROOT, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())

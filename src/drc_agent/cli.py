from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
from pathlib import Path

import yaml

from drc_agent.backends.validation_budget import managed_validation_session
from drc_agent.backends.dac26 import DAC26ReportAdapter
from drc_agent.config.loader import load_config
from drc_agent.config.methods import method_tokens
from drc_agent.experiment.matrix import expand_matrix
from drc_agent.infrastructure import (
    configured_resource_environment,
    require_storage_health_from_environment,
)
from drc_agent.llm import LLMAuditLogger, OpenAICompatibleClient
from drc_agent.observability import (
    RunProgressLogger, build_health_summary_from_config,
    format_summary, summarize_run,
)
from drc_agent.schemas.common import (
    ArtifactRef, file_sha256, stable_hash, utc_now,
)
from drc_agent.utils.artifacts import ArtifactStore
from drc_agent.workflow.checkpoints import CheckpointStore
from drc_agent.workflow.runtime import ResearchRuntime
from drc_agent.repair_kernel_integration.gate import verify_formal_integration_gate


def _add_llm_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", choices=["openrouter", "siliconflow"])
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--effort")
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--max-retries", type=int)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--thinking-budget", type=int)
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument("--enable-thinking", action="store_true", default=None)
    thinking.add_argument("--disable-thinking", action="store_false", dest="enable_thinking")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="drc-agent")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run")
    run.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    run.add_argument("--overlay", type=Path, action="append", default=[])
    run.add_argument("--experiment-config", type=Path)
    run.add_argument("--case")
    run.add_argument("--run-id", required=True)
    run.add_argument("--experiment-id", default="development")
    run.add_argument("--seed", type=int)
    run.add_argument("--method")
    run.add_argument("--max-iterations", type=int)
    run.add_argument("--max-no-progress-iterations", type=int)
    run.add_argument("--test-mode", action="store_true", help=argparse.SUPPRESS)
    run.add_argument(
        "--validation-ledger", type=Path,
        help="existing shared validation ledger (never initialized or reset)",
    )
    run.add_argument(
        "--validation-allowance",
        help="exact pre-authorized allowance in the shared validation ledger",
    )
    _add_llm_overrides(run)

    llm_check = sub.add_parser("llm-check")
    llm_check.add_argument(
        "--config", type=Path, default=Path("configs/base.yaml"),
    )
    llm_check.add_argument("--overlay", type=Path, action="append", default=[])
    llm_check.add_argument(
        "--output-log", type=Path,
        default=Path("/tmp/drc-agent-llm-preflight.jsonl"),
    )
    _add_llm_overrides(llm_check)
    llm_check.add_argument(
        "--validation-ledger", type=Path,
        help="existing shared validation ledger (never initialized or reset)",
    )
    llm_check.add_argument(
        "--validation-allowance",
        help="exact pre-authorized allowance in the shared validation ledger",
    )

    summary = sub.add_parser("summarize-run")
    summary.add_argument("--run-id", required=True)
    summary.add_argument("--project-root", type=Path, default=Path.cwd())
    summary.add_argument("--json", action="store_true", dest="as_json")

    health = sub.add_parser(
        "health-summary",
        help="read one bounded, non-mutating P5 run health/progress snapshot",
    )
    health.add_argument("--run-id", required=True)
    health.add_argument("--project-root", type=Path, default=Path.cwd())
    health.add_argument(
        "--validation-ledger", type=Path,
        help="optional shared engineering ledger for in-flight/usage counts",
    )
    health.add_argument(
        "--include-docker-probe", action="store_true",
        help="add bounded read-only Docker context/info queries",
    )

    resume = sub.add_parser(
        "resume",
        help="resume a safe checkpoint or explicitly extend a completed run",
    )
    resume.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    resume.add_argument("--run-id", required=True)
    resume.add_argument(
        "--extend-max-iterations", type=int,
        help=(
            "explicitly raise the authorized total iteration limit (maximum 5); "
            "only COMPLETED_MAX_ITERATIONS may receive a new extension"
        ),
    )
    resume.add_argument(
        "--continuation-nonce",
        help=(
            "single-use idempotency nonce; defaults deterministically from the "
            "requested total iteration limit"
        ),
    )
    resume.add_argument(
        "--extend-max-http-attempts", type=int,
        help="explicit new cumulative HTTP-attempt ceiling (increase only)",
    )
    resume.add_argument(
        "--extend-max-eda-evaluations", type=int,
        help="explicit new cumulative physical-attempt ceiling (increase only)",
    )
    resume.add_argument(
        "--extend-active-seconds", type=float,
        help="explicit new cumulative active-run seconds ceiling (increase only)",
    )
    resume.add_argument(
        "--requester-authorization", default="EXPLICIT_USER_CLI",
        help="audit label identifying the explicit extension authorization",
    )
    resume.add_argument(
        "--verify-only", action="store_true",
        help=(
            "verify config/code/input/checkpoint/GE/memory/frontier and report "
            "the next iteration without invoking LLM or EDA"
        ),
    )
    resume.add_argument("--test-mode", action="store_true", help=argparse.SUPPRESS)
    resume.add_argument(
        "--validation-ledger", type=Path,
        help="existing shared validation ledger (never initialized or reset)",
    )
    resume.add_argument(
        "--validation-allowance",
        help="exact pre-authorized allowance in the shared validation ledger",
    )

    inspect = sub.add_parser("inspect")
    inspect.add_argument("--run-id", required=True)
    inspect.add_argument("--project-root", type=Path, default=Path.cwd())

    integration_check = sub.add_parser(
        "integration-check",
        help="verify the fail-closed formal repair-kernel integration gate",
    )
    integration_check.add_argument(
        "--project-root", type=Path, default=Path.cwd(),
    )
    integration_check.add_argument(
        "--contract", type=Path,
        help="optional gate contract path (defaults to contracts/formal_integration_gate.json)",
    )
    integration_check.add_argument(
        "--verify-only", action="store_true",
        help="perform metadata/hash checks only; never invokes LLM or KLayout",
    )

    inspect_config = sub.add_parser("inspect-config")
    inspect_config.add_argument("--config", type=Path, required=True)
    inspect_config.add_argument("--experiment-config", type=Path)
    inspect_config.add_argument(
        "--overlay", type=Path, action="append", default=[],
    )
    _add_llm_overrides(inspect_config)

    experiment = sub.add_parser("experiment")
    experiment.add_argument("--matrix", type=Path, required=True)
    experiment.add_argument(
        "--config", type=Path, default=Path("configs/base.yaml"),
    )
    experiment.add_argument("--overlay", type=Path, action="append", default=[])
    experiment.add_argument("--execute", action="store_true")
    experiment.add_argument(
        "--validation-ledger", type=Path,
        help="existing shared validation ledger (never initialized or reset)",
    )

    score = sub.add_parser("score-official")
    score.add_argument("--run-id", required=True)
    score.add_argument("--project-root", type=Path, default=Path.cwd())

    export = sub.add_parser("export-dac26-agent")
    export.add_argument("--output", type=Path, required=True)
    export.add_argument(
        "--source", type=Path,
        default=Path("integrations/dac26_agent_stock/agent"),
    )

    init = sub.add_parser("init-knowledge")
    init.add_argument(
        "--path", type=Path,
        default=Path("knowledge/base_experience_graph/v1"),
    )
    return parser


def _profile(path: Path | None) -> tuple[dict, dict]:
    if path is None:
        return {}, {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    allowed = {
        "profile_version", "experiment_id", "case_id", "method", "seed",
        "config",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown experiment profile fields: {sorted(unknown)}")
    run = {
        key: raw[key] for key in [
            "experiment_id", "case_id", "method", "seed"
        ] if key in raw
    }
    profile_method = str(run.get("method", "")).upper()
    encoded = method_tokens(path.stem)
    if profile_method and encoded and encoded != {profile_method}:
        raise ValueError(
            "METHOD_IDENTITY_MISMATCH: experiment profile filename "
            f"{path.name!r} encodes {sorted(encoded)}, but method is "
            f"{profile_method}"
        )
    config = raw.get("config") or {}
    if not isinstance(config, dict):
        raise ValueError("experiment profile config must be a mapping")
    return run, config


def _runtime_overrides(args, profile_config: dict | None = None) -> dict:
    result = dict(profile_config or {})
    llm = dict(result.get("llm") or {})
    llm_fields = {
        "provider": getattr(args, "provider", None),
        "model": getattr(args, "model", None),
        "base_url": getattr(args, "base_url", None),
        "api_key_env": getattr(args, "api_key_env", None),
        "temperature": getattr(args, "temperature", None),
        "top_p": getattr(args, "top_p", None),
        "effort": getattr(args, "effort", None),
        "timeout_seconds": getattr(args, "timeout_seconds", None),
        "max_retries": getattr(args, "max_retries", None),
        "max_output_tokens": getattr(args, "max_output_tokens", None),
        "thinking_budget": getattr(args, "thinking_budget", None),
        "enable_thinking": getattr(args, "enable_thinking", None),
    }
    for key, value in llm_fields.items():
        if value is not None:
            llm[key] = value
    if any(value is not None for value in llm_fields.values()):
        llm["enabled"] = True
        llm["allow_fallback"] = False
    if llm:
        result["llm"] = llm
    workflow = dict(result.get("workflow") or {})
    if getattr(args, "max_iterations", None) is not None:
        workflow["max_iterations"] = args.max_iterations
    if getattr(args, "max_no_progress_iterations", None) is not None:
        workflow["max_no_progress_iterations"] = (
            args.max_no_progress_iterations
        )
    if workflow:
        result["workflow"] = workflow
    return result


def _resume(args) -> dict:
    locator = load_config(args.config)
    run_dir = locator.project_root / "runs" / args.run_id
    resolved = yaml.safe_load(
        (run_dir / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    from drc_agent.config.loader import AppConfig
    cfg = AppConfig.model_validate(resolved)
    # ``resource_control`` is intentionally excluded from the persisted
    # scientific config. Reattach the launcher's validated operational
    # policy for resume so the runtime gate sees the same HTTP/EDA limits as
    # a new run while the scientific/content hashes remain unchanged.
    if locator.resource_control is not None:
        cfg = cfg.model_copy(update={
            "resource_control": locator.resource_control,
        })
    operational_config = cfg
    with configured_resource_environment(operational_config):
        if not args.verify_only:
            require_storage_health_from_environment(stage="RESUME_START")
        result = ResearchRuntime(
            cfg, test_mode=getattr(args, "test_mode", False),
        ).resume(
            args.run_id,
            extend_max_iterations=args.extend_max_iterations,
            extend_max_http_attempts_per_run=(
                args.extend_max_http_attempts
            ),
            extend_max_eda_evaluations_per_run=(
                args.extend_max_eda_evaluations
            ),
            extend_whole_run_budget_seconds=args.extend_active_seconds,
            continuation_nonce_value=args.continuation_nonce,
            requester_authorization=args.requester_authorization,
            verify_only=args.verify_only,
        )
    return result.model_dump(mode="json")


def _init_knowledge(path: Path) -> dict:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"knowledge base already initialized: {path}")
    path.mkdir(parents=True, exist_ok=True)
    database = path / "experience.sqlite"
    connection = sqlite3.connect(database)
    from drc_agent.experience.store import _SCHEMA
    connection.executescript(_SCHEMA)
    connection.execute(
        "INSERT INTO graph_versions(version,parent_hash,created_at) "
        "VALUES(?,?,?)",
        (path.name, None, utc_now().isoformat()),
    )
    connection.commit()
    connection.close()
    (path / "graph.jsonl").write_text("", encoding="utf-8")
    (path / "prior_knowledge.jsonl").write_text("", encoding="utf-8")
    (path / "schema_versions.json").write_text(
        json.dumps(
            {"experience": "1.0", "prior": "1.0"},
            sort_keys=True, indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    hashes = {
        item.name: file_sha256(item)
        for item in sorted(path.iterdir()) if item.is_file()
    }
    manifest = {
        "version": path.name, "created_at": utc_now().isoformat(),
        "files": hashes,
    }
    (path / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    hashes["manifest.json"] = file_sha256(path / "manifest.json")
    (path / "SHA256SUMS").write_text(
        "".join(
            f"{digest}  {name}\n"
            for name, digest in sorted(hashes.items())
        ),
        encoding="ascii",
    )
    return {"path": str(path.resolve()), "hash": stable_hash(hashes)}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            run_profile, profile_config = _profile(args.experiment_config)
            cfg = load_config(
                args.config, overlays=args.overlay,
                overrides=_runtime_overrides(args, profile_config),
            )
            case_id = args.case or run_profile.get("case_id")
            profile_method = run_profile.get("method")
            if (
                args.method and profile_method
                and args.method.upper() != str(profile_method).upper()
            ):
                raise ValueError(
                    "METHOD_IDENTITY_MISMATCH: --method conflicts with "
                    f"experiment profile method ({args.method} != "
                    f"{profile_method})"
                )
            method = args.method or profile_method or "NO_OP"
            seed = args.seed if args.seed is not None else int(
                run_profile.get("seed", 0)
            )
            experiment_id = (
                args.experiment_id
                if args.experiment_id != "development"
                else run_profile.get("experiment_id", "development")
            )
            if not case_id:
                raise ValueError(
                    "--case is required unless experiment profile defines case_id"
                )
            with configured_resource_environment(cfg):
                require_storage_health_from_environment(stage="RUN_START")
                with managed_validation_session(
                    args.validation_ledger, label=f"cli-run:{args.run_id}",
                    allowance_id=args.validation_allowance,
                ):
                    result = ResearchRuntime(
                        cfg, test_mode=args.test_mode,
                    ).run(
                        case_id, args.run_id, experiment_id=experiment_id,
                        seed=seed, method=method,
                    )
                    payload = result.model_dump(mode="json")
        elif args.command == "llm-check":
            cfg = load_config(
                args.config, overlays=args.overlay,
                overrides=_runtime_overrides(args, {"llm": {"enabled": True}}),
            )
            progress_log = args.output_log.with_name(
                args.output_log.stem + ".events.jsonl"
            )
            progress = RunProgressLogger(progress_log)
            progress.emit(
                "provider_preflight_started", "LLM provider preflight started",
                provider=cfg.llm.provider, model=cfg.llm.model,
                total_timeout_seconds=cfg.llm.timeout_seconds,
                max_retries=cfg.llm.max_retries,
            )
            client = OpenAICompatibleClient(
                cfg.llm, LLMAuditLogger(
                    args.output_log, progress=progress,
                ),
            )
            with configured_resource_environment(cfg):
                require_storage_health_from_environment(
                    stage="LLM_PREFLIGHT",
                )
                with managed_validation_session(
                    args.validation_ledger, label="cli-llm-check",
                    allowance_id=args.validation_allowance,
                ):
                    probe = asyncio.run(client.preflight())
            progress.emit(
                "provider_preflight_completed", "LLM provider preflight completed",
                provider=probe.provider, model=probe.model, ok=probe.ok,
            )
            payload = {
                **probe.model_dump(mode="json"),
                "audit_log": str(args.output_log.resolve()),
                "progress_log": str(progress_log.resolve()),
            }
        elif args.command == "summarize-run":
            summary = summarize_run(
                args.project_root.resolve() / "runs" / args.run_id
            )
            if args.as_json:
                print(json.dumps(
                    summary, sort_keys=True, indent=2, ensure_ascii=True,
                ))
            else:
                print(format_summary(summary), end="")
            return 0
        elif args.command == "health-summary":
            root = args.project_root.resolve()
            run_dir = root / "runs" / args.run_id
            resolved = yaml.safe_load(
                (run_dir / "resolved_config.yaml").read_text(
                    encoding="utf-8"
                )
            )
            from drc_agent.config.loader import AppConfig
            cfg = AppConfig.model_validate(resolved)
            ledger = (
                args.validation_ledger.resolve()
                if args.validation_ledger is not None else None
            )
            payload = build_health_summary_from_config(
                run_dir, cfg,
                validation_ledger_path=ledger,
                include_docker_probe=args.include_docker_probe,
            )
        elif args.command == "resume":
            if args.verify_only:
                payload = _resume(args)
            else:
                with managed_validation_session(
                    args.validation_ledger, label=f"cli-resume:{args.run_id}",
                    allowance_id=args.validation_allowance,
                ):
                    payload = _resume(args)
        elif args.command == "integration-check":
            root = args.project_root.resolve()
            result = verify_formal_integration_gate(
                root,
                contract_path=(args.contract.resolve() if args.contract else None),
            )
            payload = {
                "status": "PASS",
                "verify_only": bool(args.verify_only),
                "formal_integration_gate": result,
            }
        elif args.command == "inspect":
            manifest = (
                args.project_root / "runs" / args.run_id / "manifest.json"
            )
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        elif args.command == "inspect-config":
            _, profile_config = _profile(args.experiment_config)
            cfg = load_config(
                args.config, overlays=args.overlay,
                overrides=_runtime_overrides(args, profile_config),
            )
            payload = {
                "config_hash": cfg.content_hash,
                "resolved": cfg.model_dump(mode="json"),
            }
        elif args.command == "experiment":
            specs = expand_matrix(args.matrix)
            if args.execute:
                base_cfg = load_config(args.config, overlays=args.overlay)
                results = []
                with managed_validation_session(
                    args.validation_ledger,
                    label=f"cli-experiment:{args.matrix.name}",
                ):
                    for spec in specs:
                        cfg = base_cfg.model_copy(update={
                            "workflow": base_cfg.workflow.model_copy(update={
                                "max_iterations": spec.max_iterations,
                            }),
                        })
                        results.append(ResearchRuntime(cfg).run(
                            spec.case_id, spec.run_id,
                            experiment_id=spec.experiment_id,
                            seed=spec.seed, method=spec.method,
                        ).model_dump(mode="json"))
                payload = {"run_count": len(results), "runs": results}
            else:
                payload = {
                    "run_count": len(specs),
                    "runs": [item.model_dump() for item in specs],
                }
        elif args.command == "score-official":
            root = args.project_root.resolve()
            run_dir = root / "runs" / args.run_id
            manifest = json.loads(
                (run_dir / "manifest.json").read_text(encoding="utf-8")
            )
            baseline = ArtifactRef.model_validate(
                manifest["baseline_snapshot"]["drc_ref"]
                if "baseline_snapshot" in manifest
                else json.loads(
                    (run_dir / "manifest.json").read_text()
                )["best_snapshot"]["drc_ref"]
            )
            best = ArtifactRef.model_validate(
                manifest["best_snapshot"]["drc_ref"]
            )
            adapter = DAC26ReportAdapter(
                root / "benchmarks" / "EvoDRC" / "DAC26_DRC_Benchmark"
            )
            payload = adapter.score_repair(
                baseline, best,
            ).model_dump(mode="json")
            score_ref = ArtifactStore(run_dir).write_json(
                "score/official.json", payload,
                producer="score_official",
                schema_name="OfficialRepairScore",
            )
            manifest["official_score_ref"] = score_ref.model_dump(mode="json")
            ArtifactStore(run_dir).write_json(
                "manifest.json", manifest,
                producer="score_official", schema_name="RunManifest",
            )
        elif args.command == "export-dac26-agent":
            import shutil
            if args.output.exists():
                raise FileExistsError(args.output)
            shutil.copytree(args.source, args.output)
            payload = {"output": str(args.output.resolve())}
        elif args.command == "init-knowledge":
            payload = _init_knowledge(args.path)
        else:
            raise AssertionError(args.command)
    except Exception as exc:
        print(json.dumps(
            {"status": "FAILED", "error": str(exc)}, ensure_ascii=True,
        ), file=sys.stderr)
        return 1
    print(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

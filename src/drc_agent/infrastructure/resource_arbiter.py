"""Crash-safe cross-process HTTP/EDA concurrency arbitration.

The formal run-slot lock answers "how many case processes may exist". This
module answers the different question "how many provider/EDA calls may be in
flight across those processes". A lease is represented by an advisory flock
held only for the lifetime of the dispatched operation. The small JSON body is
audit metadata; it is never authority without checking the lock and any
recorded worker/container identity.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
from typing import IO, Any, Callable, Literal
import uuid

from drc_agent.infrastructure.file_lock import (
    acquire_exclusive_file_lock,
    exclusive_file_lock,
    release_exclusive_file_lock,
)
from drc_agent.schemas.common import utc_now


RESOURCE_PROTOCOL_VERSION = "p5-global-resource-v1"
DEFAULT_GLOBAL_LIMITS = {"http": 8, "eda": 4}
RESOURCE_ROOT_ENV = "DRC_GLOBAL_RESOURCE_ROOT"
RESOURCE_OWNER_ENV = "DRC_GLOBAL_RESOURCE_OWNER"
RESOURCE_HTTP_LIMIT_ENV = "DRC_GLOBAL_HTTP_LIMIT"
RESOURCE_EDA_LIMIT_ENV = "DRC_GLOBAL_EDA_LIMIT"
RESOURCE_QUEUE_TIMEOUT_ENV = "DRC_GLOBAL_RESOURCE_QUEUE_TIMEOUT_SECONDS"
RESOURCE_POLL_INTERVAL_ENV = "DRC_GLOBAL_RESOURCE_POLL_INTERVAL_SECONDS"
ResourceKind = Literal["http", "eda"]


class ResourceQueueTimeout(RuntimeError):
    """No global lease became available before the queue deadline."""

    def __init__(self, kind: str, timeout_seconds: float, wait_seconds: float):
        super().__init__(
            f"GLOBAL_{kind.upper()}_QUEUE_TIMEOUT after {wait_seconds:.3f}s "
            f"(limit {timeout_seconds:.3f}s)"
        )
        self.kind = kind
        self.timeout_seconds = timeout_seconds
        self.wait_seconds = wait_seconds


class ResourceAcquireCancelled(RuntimeError):
    """The caller cancelled while waiting; no attempt was admitted."""


def _pid_stat(pid: int) -> tuple[int, str] | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        # comm may contain spaces. Field 3 follows the last ')'; starttime is
        # field 22, hence index 19 in the remainder.
        remainder = raw[raw.rfind(")") + 2:].split()
        return int(remainder[19]), str(remainder[0])
    except (OSError, ValueError, IndexError):
        return None


def _pid_start_ticks(pid: int) -> int | None:
    """Return Linux process start ticks, which disambiguate PID reuse."""
    value = _pid_stat(pid)
    return value[0] if value is not None else None


def _boot_id() -> str | None:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="utf-8"
        ).strip() or None
    except OSError:
        return None


def process_owner_id() -> str:
    configured = os.environ.get(RESOURCE_OWNER_ENV)
    if configured:
        return configured
    return "process:" + ":".join((
        socket.gethostname(), str(os.getpid()),
        str(_pid_start_ticks(os.getpid()) or "unknown"),
    ))


@contextmanager
def configured_resource_environment(config):
    """Temporarily activate one config's operational resource settings.

    This deliberately leaves pre-existing settings alone when the supplied
    config has no enabled resource control.  A formal resume child may obtain
    the frozen operational settings from its launcher because legacy resolved
    scientific configs intentionally exclude them.
    """
    control = getattr(config, "resource_control", None)
    updates = (
        control.environment()
        if control is not None and getattr(control, "enabled", False)
        else {}
    )
    missing = object()
    prior = {name: os.environ.get(name, missing) for name in updates}
    os.environ.update(updates)
    try:
        yield updates
    finally:
        for name, value in prior.items():
            if value is missing:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _process_identity_alive(
    *, hostname: str | None, pid: int | None,
    start_ticks: int | None, boot_id: str | None,
) -> bool | None:
    if not pid or hostname != socket.gethostname():
        return None
    current_boot = _boot_id()
    if boot_id and current_boot and boot_id != current_boot:
        return False
    current = _pid_stat(pid)
    if current is None or current[1] == "Z":
        return False
    return start_ticks is None or current[0] == start_ticks


def _unlock_close(stream: IO[str], local_lock: threading.Lock) -> None:
    try:
        release_exclusive_file_lock(stream, local_lock)
    finally:
        stream.close()


def _owned_container_active(metadata: dict[str, Any]) -> bool:
    name = metadata.get("container_name")
    if not name:
        return False
    try:
        value = subprocess.check_output(
            [
                "docker", "inspect",
                "--format={{.State.Status}}|"
                "{{index .Config.Labels \"drc_agent.attempt_id\"}}|"
                "{{index .Config.Labels \"drc_agent.owner_token_hash\"}}",
                str(name),
            ],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        ).strip()
    except (
        OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired,
    ):
        return False
    status, separator, identities = value.partition("|")
    attempt_id, second, owner_hash = identities.partition("|")
    if not separator or not second:
        return False
    if attempt_id != str(metadata.get("attempt_id") or ""):
        return False
    expected_hash = str(metadata.get("container_owner_token_hash") or "")
    if expected_hash and owner_hash != expected_hash:
        return False
    return status in {"created", "running", "restarting", "paused"}


def _default_orphan_probe(metadata: dict[str, Any]) -> bool:
    """Conservatively decide whether unlocked metadata still owns work."""
    for prefix in ("worker", "owner"):
        alive = _process_identity_alive(
            hostname=metadata.get(prefix + "_hostname"),
            pid=metadata.get(prefix + "_pid"),
            start_ticks=metadata.get(prefix + "_pid_start_ticks"),
            boot_id=metadata.get(prefix + "_boot_id"),
        )
        if alive is True:
            return True
    if _owned_container_active(metadata):
        return True
    if (
        metadata.get("kind") == "eda"
        and metadata.get("worker_pid")
        and metadata.get("worker_hostname") != socket.gethostname()
    ):
        return True
    return False


def _read_stream_json(stream: IO[str]) -> dict[str, Any]:
    stream.seek(0)
    raw = stream.read()
    if not raw.strip():
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"metadata_error": "INVALID_JSON"}
    return value if isinstance(value, dict) else {
        "metadata_error": "NOT_AN_OBJECT",
    }


def _write_stream_json(stream: IO[str], value: dict[str, Any]) -> None:
    stream.seek(0)
    stream.truncate()
    json.dump(value, stream, sort_keys=True, indent=2)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())


@dataclass
class GlobalResourceLease:
    arbiter: "GlobalResourceArbiter"
    kind: ResourceKind
    slot_id: int
    path: Path
    stream: IO[str]
    local_lock: threading.Lock
    metadata: dict[str, Any]
    wait_seconds: float
    _closed: bool = False
    _context_token: Token | None = field(default=None, init=False)

    def fileno(self) -> int:
        return self.stream.fileno()

    def activate(self) -> None:
        if self._closed:
            raise RuntimeError("cannot activate a released resource lease")
        if self._context_token is None:
            self._context_token = _CURRENT_RESOURCE_LEASE.set(self)

    def update_worker(
        self, *, pid: int, container_name: str | None = None,
        container_owner_token_hash: str | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("cannot update a released resource lease")
        self.metadata.update({
            "worker_hostname": socket.gethostname(),
            "worker_pid": pid,
            "worker_pid_start_ticks": _pid_start_ticks(pid),
            "worker_boot_id": _boot_id(),
            "container_name": container_name,
            "container_owner_token_hash": container_owner_token_hash,
            "worker_updated_at": utc_now().isoformat(),
        })
        _write_stream_json(self.stream, self.metadata)

    def clear_worker(self) -> None:
        if self._closed:
            return
        self.metadata.update({
            "worker_pid": None,
            "worker_pid_start_ticks": None,
            "container_name": None,
            "container_owner_token_hash": None,
            "worker_cleared_at": utc_now().isoformat(),
        })
        _write_stream_json(self.stream, self.metadata)

    def close(self, *, outcome: str = "RELEASED") -> None:
        if self._closed:
            return
        try:
            self.metadata.update({
                "state": "RELEASED",
                "outcome": outcome,
                "released_at": utc_now().isoformat(),
                "worker_pid": None,
                "worker_pid_start_ticks": None,
                "container_name": None,
                "container_owner_token_hash": None,
            })
            _write_stream_json(self.stream, self.metadata)
            self.arbiter._event("RELEASED", self.metadata)
        finally:
            # ContextVar tokens are context-specific.  A defensive release
            # must never leave the OS lock held merely because a caller
            # finishes a reservation from a different context.
            if self._context_token is not None:
                try:
                    _CURRENT_RESOURCE_LEASE.reset(self._context_token)
                except (RuntimeError, ValueError):
                    pass
                self._context_token = None
            try:
                release_exclusive_file_lock(
                    self.stream, self.local_lock,
                )
            finally:
                try:
                    self.stream.close()
                finally:
                    self._closed = True

    def __enter__(self) -> "GlobalResourceLease":
        self.activate()
        return self

    def __exit__(self, exc_type, _value, _traceback) -> None:
        self.close(outcome="RAISED" if exc_type is not None else "RELEASED")


_CURRENT_RESOURCE_LEASE: ContextVar[GlobalResourceLease | None] = ContextVar(
    "p5_current_global_resource_lease", default=None,
)


def current_resource_lease(
    kind: ResourceKind | None = None,
) -> GlobalResourceLease | None:
    lease = _CURRENT_RESOURCE_LEASE.get()
    if lease is None or (kind is not None and lease.kind != kind):
        return None
    return lease


class GlobalResourceArbiter:
    """A fixed-limit pool whose flock files are shared by OS processes."""

    def __init__(
        self,
        root: Path,
        *,
        limits: dict[str, int] | None = None,
        poll_interval_seconds: float = 0.05,
        orphan_probe: Callable[[dict[str, Any]], bool] | None = None,
    ) -> None:
        requested = dict(DEFAULT_GLOBAL_LIMITS if limits is None else limits)
        if set(requested) != set(DEFAULT_GLOBAL_LIMITS) or any(
            not isinstance(value, int) or value < 1
            for value in requested.values()
        ):
            raise ValueError(
                "resource limits require positive http and eda values"
            )
        if poll_interval_seconds <= 0:
            raise ValueError("resource poll interval must be positive")
        self.root = root.resolve()
        self.limits = requested
        self.poll_interval_seconds = poll_interval_seconds
        self.orphan_probe = orphan_probe or _default_orphan_probe
        self.root.mkdir(parents=True, exist_ok=True)
        self._initialize_policy()
        for kind in DEFAULT_GLOBAL_LIMITS:
            (self.root / "slots" / kind).mkdir(parents=True, exist_ok=True)

    def _initialize_policy(self) -> None:
        policy = self.root / "policy.json"
        lock_path = self.root / "policy.lock"
        with lock_path.open("a+", encoding="utf-8") as stream:
            with exclusive_file_lock(stream, lock_path):
                if policy.is_file():
                    current = json.loads(policy.read_text(encoding="utf-8"))
                    if (
                        current.get("protocol_version")
                        != RESOURCE_PROTOCOL_VERSION
                        or current.get("limits") != self.limits
                    ):
                        raise ValueError(
                            "GLOBAL_RESOURCE_POLICY_DRIFT: "
                            "existing limits differ"
                        )
                    return
                partial = policy.with_name(policy.name + ".partial")
                partial.write_text(json.dumps({
                    "protocol_version": RESOURCE_PROTOCOL_VERSION,
                    "limits": self.limits,
                    "created_at": utc_now().isoformat(),
                }, sort_keys=True, indent=2) + "\n", encoding="utf-8")
                with partial.open("rb") as value:
                    os.fsync(value.fileno())
                os.replace(partial, policy)

    @property
    def event_path(self) -> Path:
        return self.root / "resource_events.jsonl"

    def _event(
        self, event: str, metadata: dict[str, Any], **extra: Any,
    ) -> None:
        payload = {
            "protocol_version": RESOURCE_PROTOCOL_VERSION,
            "event": event,
            "recorded_at": utc_now().isoformat(),
            "kind": metadata.get("kind"),
            "slot_id": metadata.get("slot_id"),
            "lease_id": metadata.get("lease_id"),
            "owner_id": metadata.get("owner_id"),
            "run_id": metadata.get("run_id"),
            "attempt_id": metadata.get("attempt_id"),
            "purpose": metadata.get("purpose"),
            **extra,
        }
        encoded = (
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"),
                ensure_ascii=True,
            ) + "\n"
        ).encode("utf-8")
        lock_path = self.event_path.with_suffix(".jsonl.lock")
        with lock_path.open("a+") as lock:
            with exclusive_file_lock(lock, lock_path):
                with self.event_path.open("ab", buffering=0) as stream:
                    stream.write(encoded)
                    os.fsync(stream.fileno())

    def _try_acquire(
        self, kind: ResourceKind, *, owner_id: str, run_id: str,
        attempt_id: str, purpose: str, queued_at: float,
    ) -> GlobalResourceLease | None:
        for slot_id in range(self.limits[kind]):
            path = self.root / "slots" / kind / f"slot{slot_id}.lock"
            stream = path.open("a+", encoding="utf-8")
            local_lock = acquire_exclusive_file_lock(
                stream, path, blocking=False,
            )
            if local_lock is None:
                stream.close()
                continue
            try:
                previous = _read_stream_json(stream)
            except BaseException:
                _unlock_close(stream, local_lock)
                raise
            if (
                previous.get("state") == "HELD"
                and self.orphan_probe(previous)
            ):
                _unlock_close(stream, local_lock)
                continue
            if previous.get("state") == "HELD":
                try:
                    self._event(
                        "RECLAIMED", previous,
                        reason="NO_ACTIVE_OWNER_OR_WORKER",
                    )
                except BaseException:
                    _unlock_close(stream, local_lock)
                    raise
            waited = max(0.0, time.monotonic() - queued_at)
            metadata = {
                "protocol_version": RESOURCE_PROTOCOL_VERSION,
                "state": "HELD",
                "kind": kind,
                "slot_id": slot_id,
                "lease_id": "lease_" + uuid.uuid4().hex,
                "owner_id": owner_id,
                "owner_hostname": socket.gethostname(),
                "owner_pid": os.getpid(),
                "owner_pid_start_ticks": _pid_start_ticks(os.getpid()),
                "owner_boot_id": _boot_id(),
                "run_id": run_id,
                "attempt_id": attempt_id,
                "purpose": purpose,
                "acquired_at": utc_now().isoformat(),
                "queue_wait_seconds": waited,
                "worker_pid": None,
                "container_name": None,
            }
            try:
                _write_stream_json(stream, metadata)
            except BaseException:
                _unlock_close(stream, local_lock)
                raise
            lease = GlobalResourceLease(
                arbiter=self, kind=kind, slot_id=slot_id, path=path,
                stream=stream, local_lock=local_lock,
                metadata=metadata, wait_seconds=waited,
            )
            try:
                self._event("ACQUIRED", metadata, queue_wait_seconds=waited)
            except BaseException:
                # A lease is not admitted unless its audit event is durable.
                # Mark the slot reusable before unlocking when possible so a
                # still-live owner PID cannot make failed admission metadata
                # look like an orphaned in-flight operation.
                try:
                    metadata.update({
                        "state": "RELEASED",
                        "outcome": "ACQUIRE_EVENT_FAILED",
                        "released_at": utc_now().isoformat(),
                    })
                    _write_stream_json(stream, metadata)
                except BaseException:
                    pass
                _unlock_close(stream, local_lock)
                lease._closed = True
                raise
            return lease
        return None

    def acquire(
        self,
        kind: ResourceKind,
        *,
        owner_id: str,
        run_id: str,
        attempt_id: str,
        purpose: str,
        timeout_seconds: float,
        cancel_check: Callable[[], bool] | None = None,
    ) -> GlobalResourceLease:
        if kind not in DEFAULT_GLOBAL_LIMITS:
            raise ValueError(f"unknown resource kind: {kind}")
        if timeout_seconds <= 0:
            raise ValueError("resource queue timeout must be positive")
        queued_at = time.monotonic()
        queued_event_written = False
        while True:
            if cancel_check is not None and cancel_check():
                raise ResourceAcquireCancelled(
                    f"GLOBAL_{kind.upper()}_QUEUE_CANCELLED"
                )
            lease = self._try_acquire(
                kind, owner_id=owner_id, run_id=run_id,
                attempt_id=attempt_id, purpose=purpose,
                queued_at=queued_at,
            )
            if lease is not None:
                return lease
            waited = max(0.0, time.monotonic() - queued_at)
            if not queued_event_written:
                self._event("QUEUED", {
                    "kind": kind, "slot_id": None, "lease_id": None,
                    "owner_id": owner_id, "run_id": run_id,
                    "attempt_id": attempt_id, "purpose": purpose,
                })
                queued_event_written = True
            if waited >= timeout_seconds:
                self._event("QUEUE_TIMEOUT", {
                    "kind": kind, "slot_id": None, "lease_id": None,
                    "owner_id": owner_id, "run_id": run_id,
                    "attempt_id": attempt_id, "purpose": purpose,
                }, queue_wait_seconds=waited)
                raise ResourceQueueTimeout(kind, timeout_seconds, waited)
            time.sleep(min(
                self.poll_interval_seconds,
                max(0.0, timeout_seconds - waited),
            ))

    def status(self) -> dict[str, Any]:
        return read_resource_status(
            self.root, limits=self.limits, orphan_probe=self.orphan_probe,
        )


def read_resource_status(
    root: Path, *, limits: dict[str, int] | None = None,
    orphan_probe: Callable[[dict[str, Any]], bool] | None = None,
) -> dict[str, Any]:
    """Inspect an existing pool without creating or modifying any path."""
    root = root.resolve()
    configured = dict(limits or {})
    policy_path = root / "policy.json"
    if not configured and policy_path.is_file():
        try:
            policy = json.loads(policy_path.read_text(encoding="utf-8"))
            configured = {
                kind: int(value)
                for kind, value in (policy.get("limits") or {}).items()
                if kind in DEFAULT_GLOBAL_LIMITS
            }
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            configured = {}
    if set(configured) != set(DEFAULT_GLOBAL_LIMITS) or any(
        value < 1 for value in configured.values()
    ):
        return {
            "protocol_version": RESOURCE_PROTOCOL_VERSION,
            "root": str(root),
            "status": "NOT_CONFIGURED",
            "resources": {},
        }
    probe = orphan_probe or _default_orphan_probe
    resources: dict[str, Any] = {}
    for kind, limit in configured.items():
        slots: list[dict[str, Any]] = []
        for slot_id in range(limit):
            path = root / "slots" / kind / f"slot{slot_id}.lock"
            if not path.is_file():
                slots.append({
                    "slot_id": slot_id,
                    "active": False,
                    "lock_held": False,
                    "orphan_worker_active": False,
                    "metadata": None,
                    "path": str(path),
                })
                continue
            locked = False
            metadata: dict[str, Any] = {}
            try:
                with path.open("r+", encoding="utf-8") as stream:
                    local_lock = acquire_exclusive_file_lock(
                        stream, path, blocking=False,
                    )
                    if local_lock is None:
                        locked = True
                    metadata = _read_stream_json(stream)
                    if local_lock is not None:
                        release_exclusive_file_lock(stream, local_lock)
            except OSError as exc:
                metadata = {
                    "metadata_error": type(exc).__name__,
                    "message": str(exc)[:256],
                }
            orphan_active = bool(
                not locked
                and metadata.get("state") == "HELD"
                and probe(metadata)
            )
            slots.append({
                "slot_id": slot_id,
                "active": locked or orphan_active,
                "lock_held": locked,
                "orphan_worker_active": orphan_active,
                "metadata": metadata if locked or orphan_active else None,
                "path": str(path),
            })
        resources[kind] = {
            "limit": limit,
            "in_flight": sum(bool(item["active"]) for item in slots),
            "slots": slots,
        }
    return {
        "protocol_version": RESOURCE_PROTOCOL_VERSION,
        "root": str(root),
        "status": "AVAILABLE",
        "resources": resources,
    }


def arbiter_from_environment() -> GlobalResourceArbiter | None:
    raw_root = os.environ.get(RESOURCE_ROOT_ENV)
    if not raw_root:
        return None
    limits = {
        "http": int(os.environ.get(
            RESOURCE_HTTP_LIMIT_ENV, DEFAULT_GLOBAL_LIMITS["http"],
        )),
        "eda": int(os.environ.get(
            RESOURCE_EDA_LIMIT_ENV, DEFAULT_GLOBAL_LIMITS["eda"],
        )),
    }
    poll = float(os.environ.get(RESOURCE_POLL_INTERVAL_ENV, "0.05"))
    return GlobalResourceArbiter(
        Path(raw_root), limits=limits, poll_interval_seconds=poll,
    )


def acquire_from_environment(
    kind: ResourceKind, *, run_id: str, attempt_id: str, purpose: str,
    timeout_seconds: float | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> GlobalResourceLease | None:
    try:
        arbiter = arbiter_from_environment()
        if arbiter is None:
            return None
        timeout = timeout_seconds
        if timeout is None:
            timeout = float(os.environ.get(RESOURCE_QUEUE_TIMEOUT_ENV, "600"))
        return arbiter.acquire(
            kind, owner_id=process_owner_id(), run_id=run_id,
            attempt_id=attempt_id, purpose=purpose,
            timeout_seconds=timeout, cancel_check=cancel_check,
        )
    except (ResourceQueueTimeout, ResourceAcquireCancelled):
        raise
    except OSError as exc:
        from drc_agent.infrastructure.storage_health import (
            record_storage_incident_from_environment,
            storage_pause_from_exception,
        )
        pause = storage_pause_from_exception(
            exc, stage="GLOBAL_" + kind.upper() + "_ADMISSION",
            provider="P5_GLOBAL_RESOURCE_ARBITER",
            model=kind,
        )
        if pause is not None:
            pause.health_snapshot = (
                record_storage_incident_from_environment(
                    code=pause.original_exception_type or "STORAGE_EXHAUSTED",
                    stage=pause.failure_stage,
                    details={"resource_root": os.environ.get(RESOURCE_ROOT_ENV)},
                ) or {}
            )
            raise pause from exc
        from drc_agent.reliability import FailureCode, InfrastructurePause
        raise InfrastructurePause(
            f"global {kind} resource admission failed: {exc}",
            failure_code=FailureCode.WORKSPACE_OWNERSHIP,
            retryable=False,
            provider="P5_GLOBAL_RESOURCE_ARBITER",
            model=kind,
            original_exception_type=type(exc).__name__,
            failure_stage="GLOBAL_" + kind.upper() + "_ADMISSION",
            pause_scope="SHARED_RESOURCE",
        ) from exc
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        from drc_agent.reliability import FailureCode, InfrastructurePause
        raise InfrastructurePause(
            f"global {kind} resource policy is invalid: {exc}",
            failure_code=FailureCode.CONFIGURATION,
            retryable=False,
            provider="P5_GLOBAL_RESOURCE_ARBITER",
            model=kind,
            original_exception_type=type(exc).__name__,
            failure_stage="GLOBAL_" + kind.upper() + "_ADMISSION",
            pause_scope="SHARED_RESOURCE",
        ) from exc


async def acquire_from_environment_async(
    kind: ResourceKind, *, run_id: str, attempt_id: str, purpose: str,
    timeout_seconds: float | None = None,
) -> GlobalResourceLease | None:
    """Cancellation-safe async bridge for the blocking flock queue."""
    cancelled = threading.Event()
    task = asyncio.create_task(asyncio.to_thread(
        acquire_from_environment,
        kind, run_id=run_id, attempt_id=attempt_id, purpose=purpose,
        timeout_seconds=timeout_seconds, cancel_check=cancelled.is_set,
    ))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        cancelled.set()
        try:
            lease = await asyncio.shield(task)
        except BaseException:
            lease = None
        if lease is not None:
            lease.close(outcome="QUEUE_CALLER_CANCELLED")
        raise

"""Cron scheduler — JSON5-backed, smart-timer, async-native.

Design principles
-----------------
* **One source of truth**: system jobs live in ``system_cron.json5`` (managed
  by code / ops), agent-created jobs in ``jobs.json5`` (managed at runtime via
  tool calls).  Both files are JSON5 so humans can add comments.

* **Smart timer**: instead of fixed-interval polling the scheduler computes
  the next due job and arms an ``asyncio`` timer to that instant.  A 60 s cap
  guards against missed wakeups.

* **Concurrency cap + error backoff**: at most ``MAX_CONCURRENT_JOBS`` tasks
  in flight; exponential backoff on repeated failures with auto-disable after
  ``DISABLE_THRESHOLD`` consecutive errors.
"""

from __future__ import annotations

import asyncio
import os
import random
import shutil
import time
import zoneinfo
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mirai.agent.agent_loop import AgentLoop
    from mirai.agent.im.base import BaseIMProvider

import json5
from croniter import croniter  # type: ignore[import-untyped]

from mirai.logging import get_logger

log = get_logger("mirai.cron")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_TIMER_DELAY = 60  # seconds; cap to avoid drift / long sleep
MAX_CONCURRENT_JOBS = 3
WARN_THRESHOLD = 3  # consecutive errors before curator alert
DISABLE_THRESHOLD = 5  # consecutive errors before auto-disable
MISSED_GRACE_SECONDS = 3600  # recover missed jobs within 1 hour
EMPTY_STORE: dict[str, Any] = {"version": 1, "jobs": []}

# Exponential backoff for consecutive errors (indexed by error count - 1)
ERROR_BACKOFF_MS = [
    30_000,  # 1st error  →  30 s
    60_000,  # 2nd error  →   1 min
    5 * 60_000,  # 3rd error  →   5 min
    15 * 60_000,  # 4th error  →  15 min
    60 * 60_000,  # 5th+ error →  60 min
]


def _error_backoff_ms(consecutive_errors: int) -> int:
    """Return backoff delay in ms for the given error count."""
    idx = min(consecutive_errors - 1, len(ERROR_BACKOFF_MS) - 1)
    return ERROR_BACKOFF_MS[max(0, idx)]


# ---------------------------------------------------------------------------
# Data model helpers
# ---------------------------------------------------------------------------


def _now_ms() -> int:
    """Current time in epoch milliseconds."""
    return int(time.time() * 1000)


def _ms_to_dt(ms: int) -> datetime:
    """Epoch millis → timezone-aware datetime (UTC)."""
    return datetime.fromtimestamp(ms / 1000, tz=UTC)


def _dt_to_ms(dt: datetime) -> int:
    """Datetime → epoch millis."""
    return int(dt.timestamp() * 1000)


# ---------------------------------------------------------------------------
# Schedule computation
# ---------------------------------------------------------------------------


def compute_next_run(schedule: dict[str, Any], after_ms: int | None = None) -> int | None:
    """Compute next run time (epoch ms) for a schedule dict.

    Supports:
      - {"kind": "cron", "expr": "0 9 * * 1-5", "tz": "Asia/Shanghai"}
      - {"kind": "every", "everyMs": 60000}
      - {"kind": "at", "at": "2026-02-15T14:00:00+08:00"}
    """
    after_ms = after_ms or _now_ms()
    kind = schedule.get("kind", "")

    if kind == "cron":
        expr = schedule["expr"]
        tz_name = schedule.get("tz")
        tz = None
        if tz_name:
            try:
                tz = zoneinfo.ZoneInfo(tz_name)
            except Exception:
                pass  # fallback to UTC
        base = _ms_to_dt(after_ms)
        if tz:
            base = base.astimezone(tz)
        cron = croniter(expr, base)
        next_dt: datetime = cron.get_next(datetime)
        return _dt_to_ms(next_dt)

    if kind == "every":
        interval_ms: int = int(schedule.get("everyMs", 60000))
        return after_ms + interval_ms

    if kind == "at":
        at_str = schedule["at"]
        at_dt = datetime.fromisoformat(at_str)
        at_ms = _dt_to_ms(at_dt)
        return at_ms if at_ms > after_ms else None  # already passed

    return None


# ---------------------------------------------------------------------------
# Store I/O: load / save / atomic write
# ---------------------------------------------------------------------------


def _load_json5(path: Path) -> dict[str, Any]:
    """Load a JSON5 file, returning empty store if file is missing or corrupt."""
    if not path.exists():
        return {"version": 1, "jobs": []}
    try:
        text = path.read_text(encoding="utf-8")
        data: dict[str, Any] = json5.loads(text)
        return data
    except Exception as exc:
        log.warning("cron_store_load_failed", path=str(path), error=str(exc))
        return {"version": 1, "jobs": []}


def _save_json5_atomic(path: Path, data: dict[str, Any]) -> None:
    """Atomic write: write to tmp then rename (POSIX atomic).  Best-effort .bak."""
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_name = f"{path.name}.{os.getpid()}.{random.randint(0, 0xFFFF):04x}.tmp"
    tmp_path = path.parent / tmp_name
    text = json5.dumps(data, indent=2, ensure_ascii=False)
    tmp_path.write_text(text + "\n", encoding="utf-8")
    os.replace(str(tmp_path), str(path))  # atomic on POSIX

    # Best-effort backup
    try:
        bak_path = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(str(path), str(bak_path))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# CronStore — data ownership and persistence
# ---------------------------------------------------------------------------


class CronStore:
    """Owns cron job data and JSON5 file persistence.

    Responsibilities:
      - File paths and mtime tracking
      - In-memory job lists (system + agent)
      - Load / save / hot-reload from disk
      - Startup recovery and schedule computation
    """

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = Path(state_dir)
        self._system_path = self.state_dir / "system.json5"
        self._jobs_path = self.state_dir / "jobs.json5"

        # In-memory job lists
        self.system_jobs: list[dict[str, Any]] = []
        self.agent_jobs: list[dict[str, Any]] = []

        # Mtime tracking for hot-reload
        self._last_mtime: float = 0.0
        self._system_mtime: float = 0.0

    @property
    def jobs_path(self) -> Path:
        """Path to agent jobs file (for tests / external access)."""
        return self._jobs_path

    @property
    def all_jobs(self) -> list[dict[str, Any]]:
        """Combined list of system + agent jobs."""
        return self.system_jobs + self.agent_jobs

    def load(self) -> None:
        """Load both system and agent job stores from disk."""
        sys_data = _load_json5(self._system_path)
        self.system_jobs = sys_data.get("jobs", [])
        if self._system_path.exists():
            self._system_mtime = self._system_path.stat().st_mtime

        agent_data = _load_json5(self._jobs_path)
        self.agent_jobs = agent_data.get("jobs", [])
        if self._jobs_path.exists():
            self._last_mtime = self._jobs_path.stat().st_mtime

        total = len(self.system_jobs) + len(self.agent_jobs)
        log.info(
            "cron_stores_loaded",
            system_jobs=len(self.system_jobs),
            agent_jobs=len(self.agent_jobs),
            total=total,
        )

    def save_agent_jobs(self) -> None:
        """Persist agent jobs to disk (system.json5 is never written at runtime)."""
        data: dict[str, Any] = {"version": 1, "jobs": self.agent_jobs}
        _save_json5_atomic(self._jobs_path, data)
        self._last_mtime = self._jobs_path.stat().st_mtime

    def maybe_reload(self) -> bool:
        """Reload from disk if files have been externally edited.

        Returns True if any file was reloaded.
        """
        reloaded = False

        if self._jobs_path.exists():
            mtime = self._jobs_path.stat().st_mtime
            if mtime != self._last_mtime:
                agent_data = _load_json5(self._jobs_path)
                self.agent_jobs = agent_data.get("jobs", [])
                self._last_mtime = mtime
                reloaded = True
                log.info("cron_agent_jobs_reloaded")

        if self._system_path.exists():
            mtime = self._system_path.stat().st_mtime
            if mtime != self._system_mtime:
                sys_data = _load_json5(self._system_path)
                self.system_jobs = sys_data.get("jobs", [])
                self._system_mtime = mtime
                reloaded = True
                log.info("cron_system_jobs_reloaded")

        if reloaded:
            self.ensure_next_runs()

        return reloaded

    def startup_recovery(self) -> None:
        """Recompute nextRunAtMs for all jobs; recover missed ones."""
        now = _now_ms()
        grace = MISSED_GRACE_SECONDS * 1000

        for job in self.all_jobs:
            if not job.get("enabled", True):
                continue
            state = job.setdefault("state", {})
            next_run = state.get("nextRunAtMs")

            if next_run is None:
                computed = compute_next_run(job.get("schedule", {}), after_ms=now)
                if computed:
                    state["nextRunAtMs"] = computed
            elif next_run < now - grace:
                computed = compute_next_run(job.get("schedule", {}), after_ms=now)
                if computed:
                    state["nextRunAtMs"] = computed
                    log.info("cron_job_skip_stale", job_id=job["id"])
            elif next_run < now:
                log.info("cron_job_missed_recovery", job_id=job["id"])

        self.save_agent_jobs()

    def ensure_next_runs(self) -> None:
        """Ensure all enabled jobs have a nextRunAtMs."""
        now = _now_ms()
        for job in self.all_jobs:
            if not job.get("enabled", True):
                continue
            state = job.setdefault("state", {})
            if state.get("nextRunAtMs") is None:
                computed = compute_next_run(job.get("schedule", {}), after_ms=now)
                if computed:
                    state["nextRunAtMs"] = computed


# ---------------------------------------------------------------------------
# CronScheduler
# ---------------------------------------------------------------------------


@dataclass
class _RunningJob:
    """Tracks a currently-executing job."""

    job_id: str
    task: asyncio.Task[Any]
    started_ms: int = field(default_factory=_now_ms)


class CronScheduler:
    """JSON5-backed scheduler with smart timer.

    Uses ``asyncio`` timers armed to the next due job instead of fixed-interval
    polling.  Falls back to a 60-second cap to guard against drift.

    Delegates data persistence to :class:`CronStore`.
    """

    def __init__(
        self,
        state_dir: Path,
        agent: AgentLoop | None = None,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.agent = agent
        self.im_provider: BaseIMProvider | None = None  # injected by bootstrap

        # Delegate data ownership to CronStore
        self.store = CronStore(state_dir)

        # Concurrency tracking
        self._running: dict[str, _RunningJob] = {}

        # Smart timer state
        self._timer_handle: asyncio.TimerHandle | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stopped = False

    # Backward-compatible property aliases
    @property
    def _system_jobs(self) -> list[dict[str, Any]]:
        return self.store.system_jobs

    @_system_jobs.setter
    def _system_jobs(self, value: list[dict[str, Any]]) -> None:
        self.store.system_jobs = value

    @property
    def _agent_jobs(self) -> list[dict[str, Any]]:
        return self.store.agent_jobs

    @_agent_jobs.setter
    def _agent_jobs(self, value: list[dict[str, Any]]) -> None:
        self.store.agent_jobs = value

    @property
    def _jobs_path(self) -> Path:
        return self.store.jobs_path

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Load stores and arm the first timer."""
        self._loop = loop or asyncio.get_running_loop()
        self._stopped = False
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.store.load()
        self.store.startup_recovery()
        self._arm_timer()
        log.info(
            "CronScheduler_started",
            jobs=len(self.store.all_jobs),
            next_wake_ms=self._next_wake_ms(),
        )

    def stop(self) -> None:
        """Cancel the pending timer."""
        self._stopped = True
        if self._timer_handle:
            self._timer_handle.cancel()
            self._timer_handle = None
        log.info("CronScheduler_stopped")

    # ------------------------------------------------------------------
    # Smart Timer (replaces fixed-interval polling)
    # ------------------------------------------------------------------

    def _next_wake_ms(self) -> int | None:
        """Earliest ``nextRunAtMs`` across all enabled jobs, or *None*."""
        earliest: int | None = None
        for job in self.store.all_jobs:
            if not job.get("enabled", True):
                continue
            nr = job.get("state", {}).get("nextRunAtMs")
            if nr is not None and (earliest is None or nr < earliest):
                earliest = nr
        return earliest

    def _arm_timer(self) -> None:
        """Schedule ``_on_timer`` at the next due time (capped at 60s)."""
        if self._stopped or self._loop is None:
            return
        if self._timer_handle:
            self._timer_handle.cancel()
            self._timer_handle = None

        next_at = self._next_wake_ms()
        if next_at is None:
            # No jobs — still wake periodically to catch hot-reloaded files
            delay: float = float(MAX_TIMER_DELAY)
        else:
            now = _now_ms()
            delay = max(float(next_at - now) / 1000.0, 0.0)
            delay = min(delay, float(MAX_TIMER_DELAY))

        self._timer_handle = self._loop.call_later(delay, self._schedule_on_timer)
        log.debug("cron_timer_armed", delay_s=round(delay, 2), next_wake_ms=next_at)

    def _schedule_on_timer(self) -> None:
        """Bridge from ``call_later`` (sync) to the async ``_on_timer``."""
        if not self._stopped and self._loop is not None:
            self._loop.create_task(self._on_timer())

    async def _on_timer(self) -> None:
        """Timer callback — detect changes, fire due jobs, re-arm."""
        if self._stopped:
            return

        try:
            # 1. Detect external file edits
            self.store.maybe_reload()

            # 2. Reap completed running jobs
            self._reap_finished()

            # 3. Find and fire due jobs
            now = _now_ms()
            fired_any = False

            for job in self.store.all_jobs:
                if not job.get("enabled", True):
                    continue

                next_run = job.get("state", {}).get("nextRunAtMs")
                if next_run is None:
                    continue
                if next_run > now:
                    continue

                # Concurrency cap
                if len(self._running) >= MAX_CONCURRENT_JOBS:
                    log.debug("cron_concurrent_cap", running=len(self._running))
                    break

                # Reentrance guard
                job_id = job["id"]
                if job_id in self._running:
                    continue

                await self._fire_job(job)
                fired_any = True

            # 4. Persist if anything changed
            if fired_any:
                self.store.save_agent_jobs()
        except Exception as exc:
            log.error("cron_timer_tick_failed", error=str(exc))
        finally:
            self._arm_timer()

    # ------------------------------------------------------------------
    # Job firing
    # ------------------------------------------------------------------

    async def _fire_job(self, job: dict[str, Any]) -> None:
        """Spawn an asyncio task to execute the job's prompt."""
        job_id = job["id"]
        prompt = job.get("payload", {}).get("prompt", "")
        if not prompt:
            log.warning("cron_job_no_prompt", job_id=job_id)
            return

        log.info("cron_job_firing", job_id=job_id, name=job.get("name"))

        task = asyncio.get_running_loop().create_task(
            self._run_job(job, prompt),
        )
        self._running[job_id] = _RunningJob(job_id=job_id, task=task)

    async def _run_job(self, job: dict[str, Any], prompt: str) -> None:
        """Execute a single job and update its state.

        The cron layer is a pure timer — it does NOT deliver the agent's
        response.  The agent has its own IM tools to communicate.
        """
        job_id = job["id"]
        state = job.setdefault("state", {})
        started_ms = _now_ms()

        try:
            if self.agent:
                await self.agent.run(prompt)

            # Success
            state["lastStatus"] = "ok"
            state["consecutiveErrors"] = 0
            state["runCount"] = state.get("runCount", 0) + 1
            log.info("cron_job_ok", job_id=job_id, run_count=state["runCount"])

        except Exception as exc:
            errors = state.get("consecutiveErrors", 0) + 1
            state["consecutiveErrors"] = errors
            state["lastStatus"] = "error"
            state["lastError"] = str(exc)[:200]
            log.warning("cron_job_failed", job_id=job_id, errors=errors, error=str(exc))

            # Progressive error handling
            if errors >= WARN_THRESHOLD:
                await self._alert_curator(f"⚠️ Cron job `{job_id}` failed {errors} times consecutively.")

            if errors >= DISABLE_THRESHOLD:
                job["enabled"] = False
                log.error("cron_job_auto_disabled", job_id=job_id)
                await self._alert_curator(f"🛑 Cron job `{job_id}` auto-disabled after {errors} consecutive failures.")

        finally:
            now = _now_ms()
            state["lastRunAtMs"] = now
            state["lastDurationMs"] = now - started_ms

            # Handle one-shot jobs
            if job.get("deleteAfterRun"):
                self._agent_jobs = [j for j in self._agent_jobs if j["id"] != job_id]
                log.info("cron_job_deleted_oneshot", job_id=job_id)
            else:
                # Compute next run, applying error backoff if needed
                base_next = compute_next_run(job.get("schedule", {}), after_ms=now)
                errors = state.get("consecutiveErrors", 0)
                if errors > 0 and base_next is not None:
                    backoff = _error_backoff_ms(errors)
                    state["nextRunAtMs"] = max(base_next, now + backoff)
                    log.info(
                        "cron_job_error_backoff",
                        job_id=job_id,
                        errors=errors,
                        backoff_ms=backoff,
                        next_run_ms=state["nextRunAtMs"],
                    )
                elif base_next:
                    state["nextRunAtMs"] = base_next

            # Remove from running set
            self._running.pop(job_id, None)

            # Persist immediately after job completes
            self.store.save_agent_jobs()

            # Re-arm timer in case next-run changed
            self._arm_timer()

    # ------------------------------------------------------------------
    # Backward-compatible shims (delegate to CronStore)
    # ------------------------------------------------------------------

    def _load_stores(self) -> None:
        """Shim — delegates to ``self.store.load()``."""
        self.store.load()

    def _save_agent_jobs(self) -> None:
        """Shim — delegates to ``self.store.save_agent_jobs()``."""
        self.store.save_agent_jobs()

    def _maybe_reload(self) -> bool:
        """Shim — delegates to ``self.store.maybe_reload()``."""
        return self.store.maybe_reload()

    def _startup_recovery(self) -> None:
        """Shim — delegates to ``self.store.startup_recovery()``."""
        self.store.startup_recovery()

    # ------------------------------------------------------------------
    # Concurrency management
    # ------------------------------------------------------------------

    def _reap_finished(self) -> None:
        """Clean up completed tasks from the running set."""
        done = [jid for jid, rj in self._running.items() if rj.task.done()]
        for jid in done:
            rj = self._running.pop(jid)
            if rj.task.exception():
                log.warning("cron_job_task_exception", job_id=jid, error=str(rj.task.exception()))

    # ------------------------------------------------------------------
    # Curator alerts
    # ------------------------------------------------------------------

    async def _alert_curator(self, message: str) -> None:
        """Send an alert message to the curator via IM (best-effort)."""
        try:
            if self.im_provider:
                await self.im_provider.send_markdown(
                    content=message,
                    title="Cron Alert",
                )
        except Exception as exc:
            log.warning("cron_alert_failed", error=str(exc))


# ---------------------------------------------------------------------------
# System jobs initializer (called from bootstrap)
# ---------------------------------------------------------------------------


def ensure_system_jobs(state_dir: Path) -> None:
    """Create default system.json5 if it doesn't exist.

    System jobs are framework-level periodic tasks that agents cannot modify.
    """
    system_path = state_dir / "system.json5"
    if system_path.exists():
        return

    state_dir.mkdir(parents=True, exist_ok=True)
    default_system: dict[str, Any] = {
        "version": 1,
        "jobs": [
            {
                "id": "sys:registry-refresh",
                "name": "registry-refresh",
                "enabled": False,  # disabled until migrated from bootstrap loop
                "schedule": {"kind": "every", "everyMs": 300_000},
                "payload": {
                    "kind": "agentTurn",
                    "prompt": "Refresh model registry from all sources.",
                },
                "state": {},
            },
            {
                "id": "sys:health-check",
                "name": "health-check",
                "enabled": False,  # disabled until migrated from bootstrap loop
                "schedule": {"kind": "every", "everyMs": 600_000},
                "payload": {
                    "kind": "agentTurn",
                    "prompt": "Probe all configured free provider health.",
                },
                "state": {},
            },
        ],
    }
    _save_json5_atomic(system_path, default_system)
    log.info("cron_system_jobs_created", path=str(system_path))

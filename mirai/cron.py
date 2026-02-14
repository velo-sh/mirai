"""Cron Service — JSON5 file-based agent-driven task scheduler (RFC 0005).

Provides lightweight periodic scheduling with:
- JSON5 file persistence (human-readable, commentable, Git-friendly)
- System vs agent job separation (two files)
- Atomic write with .bak backup
- External edit detection (mtime-based reload)
- Progressive error handling (warn → disable)
- Missed job recovery on startup
"""

from __future__ import annotations

import asyncio
import os
import random
import shutil
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import json5
from croniter import croniter  # type: ignore[import-untyped]

from mirai.logging import get_logger
from mirai.utils.service import BaseBackgroundService

log = get_logger("mirai.cron")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TICK_INTERVAL = 15  # seconds
MAX_CONCURRENT_JOBS = 3
WARN_THRESHOLD = 3  # consecutive errors before curator alert
DISABLE_THRESHOLD = 5  # consecutive errors before auto-disable
MISSED_GRACE_SECONDS = 3600  # recover missed jobs within 1 hour
EMPTY_STORE: dict[str, Any] = {"version": 1, "jobs": []}


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
                import zoneinfo

                tz = zoneinfo.ZoneInfo(tz_name)
            except Exception:
                pass  # fallback to UTC
        base = _ms_to_dt(after_ms)
        if tz:
            base = base.astimezone(tz)
        cron = croniter(expr, base)
        next_dt: datetime = cron.get_next(datetime)
        return _dt_to_ms(next_dt)

    elif kind == "every":
        interval_ms: int = int(schedule.get("everyMs", 60000))
        return after_ms + interval_ms

    elif kind == "at":
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
# CronScheduler
# ---------------------------------------------------------------------------


@dataclass
class _RunningJob:
    """Tracks a currently-executing job."""

    job_id: str
    task: asyncio.Task[Any]
    started_ms: int = field(default_factory=_now_ms)


class CronScheduler(BaseBackgroundService):
    """JSON5-backed scheduler with 15-second tick loop.

    Loads two job files:
    - ``system.json5`` — bootstrap-managed, agent read-only
    - ``jobs.json5``   — agent-editable
    """

    def __init__(
        self,
        state_dir: Path,
        agent: Any = None,
        im_provider: Any = None,
    ) -> None:
        super().__init__(TICK_INTERVAL)
        self.state_dir = Path(state_dir)
        self.agent = agent
        self.im_provider = im_provider

        self._system_path = self.state_dir / "system.json5"
        self._jobs_path = self.state_dir / "jobs.json5"

        # In-memory merged list (system + agent jobs)
        self._system_jobs: list[dict[str, Any]] = []
        self._agent_jobs: list[dict[str, Any]] = []
        self._last_mtime: float = 0.0
        self._system_mtime: float = 0.0

        # Concurrency tracking
        self._running: dict[str, _RunningJob] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Load stores and start the tick loop."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._load_stores()
        self._startup_recovery()
        super().start(loop)

    # ------------------------------------------------------------------
    # BaseBackgroundService.tick()
    # ------------------------------------------------------------------

    async def tick(self) -> None:
        """Single scheduler tick — detect changes, fire due jobs, persist."""
        # 1. Detect external file edits
        self._maybe_reload()

        # 2. Reap completed running jobs
        self._reap_finished()

        # 3. Find and fire due jobs
        now = _now_ms()
        all_jobs = self._system_jobs + self._agent_jobs
        fired_any = False

        for job in all_jobs:
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
            self._save_agent_jobs()

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
        """Execute a single job and update its state."""
        job_id = job["id"]
        state = job.setdefault("state", {})

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

            # Handle one-shot jobs
            if job.get("deleteAfterRun"):
                self._agent_jobs = [j for j in self._agent_jobs if j["id"] != job_id]
                log.info("cron_job_deleted_oneshot", job_id=job_id)
            else:
                # Compute next run
                next_run = compute_next_run(job.get("schedule", {}), after_ms=now)
                if next_run:
                    state["nextRunAtMs"] = next_run

            # Remove from running set
            self._running.pop(job_id, None)

            # Persist immediately after job completes
            self._save_agent_jobs()

    # ------------------------------------------------------------------
    # Store management
    # ------------------------------------------------------------------

    def _load_stores(self) -> None:
        """Load both system and agent job stores from disk."""
        sys_data = _load_json5(self._system_path)
        self._system_jobs = sys_data.get("jobs", [])
        if self._system_path.exists():
            self._system_mtime = self._system_path.stat().st_mtime

        agent_data = _load_json5(self._jobs_path)
        self._agent_jobs = agent_data.get("jobs", [])
        if self._jobs_path.exists():
            self._last_mtime = self._jobs_path.stat().st_mtime

        total = len(self._system_jobs) + len(self._agent_jobs)
        log.info(
            "cron_stores_loaded",
            system_jobs=len(self._system_jobs),
            agent_jobs=len(self._agent_jobs),
            total=total,
        )

    def _maybe_reload(self) -> None:
        """Reload from disk if files have been externally edited."""
        reloaded = False

        if self._jobs_path.exists():
            mtime = self._jobs_path.stat().st_mtime
            if mtime != self._last_mtime:
                agent_data = _load_json5(self._jobs_path)
                self._agent_jobs = agent_data.get("jobs", [])
                self._last_mtime = mtime
                reloaded = True
                log.info("cron_agent_jobs_reloaded")

        if self._system_path.exists():
            mtime = self._system_path.stat().st_mtime
            if mtime != self._system_mtime:
                sys_data = _load_json5(self._system_path)
                self._system_jobs = sys_data.get("jobs", [])
                self._system_mtime = mtime
                reloaded = True
                log.info("cron_system_jobs_reloaded")

        if reloaded:
            # Recompute next-run for newly loaded jobs that lack it
            self._ensure_next_runs()

    def _save_agent_jobs(self) -> None:
        """Persist agent jobs to disk (system.json5 is never written at runtime)."""
        data: dict[str, Any] = {"version": 1, "jobs": self._agent_jobs}
        _save_json5_atomic(self._jobs_path, data)
        self._last_mtime = self._jobs_path.stat().st_mtime

    # ------------------------------------------------------------------
    # Startup recovery
    # ------------------------------------------------------------------

    def _startup_recovery(self) -> None:
        """Recompute nextRunAtMs for all jobs; recover missed ones."""
        now = _now_ms()
        grace = MISSED_GRACE_SECONDS * 1000

        for job in self._system_jobs + self._agent_jobs:
            if not job.get("enabled", True):
                continue
            state = job.setdefault("state", {})
            next_run = state.get("nextRunAtMs")

            if next_run is None:
                # Never scheduled — compute initial
                computed = compute_next_run(job.get("schedule", {}), after_ms=now)
                if computed:
                    state["nextRunAtMs"] = computed
            elif next_run < now - grace:
                # Missed by more than grace period — skip, advance
                computed = compute_next_run(job.get("schedule", {}), after_ms=now)
                if computed:
                    state["nextRunAtMs"] = computed
                    log.info("cron_job_skip_stale", job_id=job["id"])
            elif next_run < now:
                # Missed within grace — fire immediately (keep nextRunAtMs as-is)
                log.info("cron_job_missed_recovery", job_id=job["id"])
                # nextRunAtMs stays in the past so tick() picks it up

        self._save_agent_jobs()

    def _ensure_next_runs(self) -> None:
        """Ensure all enabled jobs have a nextRunAtMs."""
        now = _now_ms()
        for job in self._system_jobs + self._agent_jobs:
            if not job.get("enabled", True):
                continue
            state = job.setdefault("state", {})
            if state.get("nextRunAtMs") is None:
                computed = compute_next_run(job.get("schedule", {}), after_ms=now)
                if computed:
                    state["nextRunAtMs"] = computed

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

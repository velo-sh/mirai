"""Tests for mirai.cron — JSON5-based cron scheduler (RFC 0005)."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from mirai.cron import (
    CronScheduler,
    _load_json5,
    _save_json5_atomic,
    compute_next_run,
    ensure_system_jobs,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def state_dir(tmp_path: Path) -> Path:
    """Provide a clean temp state directory for each test."""
    d = tmp_path / "cron"
    d.mkdir()
    return d


def _make_job(
    *,
    job_id: str = "test-001",
    name: str = "test-job",
    enabled: bool = True,
    schedule: dict[str, Any] | None = None,
    prompt: str = "Do something.",
    delete_after_run: bool = False,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Helper to create a job dict."""
    return {
        "id": job_id,
        "name": name,
        "enabled": enabled,
        "schedule": schedule or {"kind": "every", "everyMs": 60000},
        "payload": {"kind": "agentTurn", "prompt": prompt},
        **({"deleteAfterRun": True} if delete_after_run else {}),
        "state": state or {},
    }


def _write_jobs(path: Path, jobs: list[dict[str, Any]]) -> None:
    """Write a jobs.json5 file."""
    _save_json5_atomic(path, {"version": 1, "jobs": jobs})


# ---------------------------------------------------------------------------
# Tests: JSON5 I/O
# ---------------------------------------------------------------------------


class TestJson5IO:
    def test_load_missing_file(self, state_dir: Path) -> None:
        data = _load_json5(state_dir / "nonexistent.json5")
        assert data == {"version": 1, "jobs": []}

    def test_load_corrupt_file(self, state_dir: Path) -> None:
        p = state_dir / "bad.json5"
        p.write_text("{invalid json5!!", encoding="utf-8")
        data = _load_json5(p)
        assert data == {"version": 1, "jobs": []}

    def test_save_and_load_roundtrip(self, state_dir: Path) -> None:
        path = state_dir / "jobs.json5"
        jobs = [_make_job()]
        _save_json5_atomic(path, {"version": 1, "jobs": jobs})

        loaded = _load_json5(path)
        assert loaded["version"] == 1
        assert len(loaded["jobs"]) == 1
        assert loaded["jobs"][0]["id"] == "test-001"

    def test_atomic_write_creates_backup(self, state_dir: Path) -> None:
        path = state_dir / "jobs.json5"
        _save_json5_atomic(path, {"version": 1, "jobs": []})

        bak = path.with_suffix(".json5.bak")
        assert bak.exists()

    def test_atomic_write_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "deep" / "nested" / "jobs.json5"
        _save_json5_atomic(path, {"version": 1, "jobs": []})
        assert path.exists()


# ---------------------------------------------------------------------------
# Tests: Schedule computation
# ---------------------------------------------------------------------------


class TestScheduleComputation:
    def test_cron_schedule(self) -> None:
        """Standard cron expression returns a future time."""
        schedule = {"kind": "cron", "expr": "* * * * *"}  # every minute
        now_ms = int(time.time() * 1000)
        next_ms = compute_next_run(schedule, after_ms=now_ms)

        assert next_ms is not None
        assert next_ms > now_ms
        # Should be within ~61 seconds
        assert next_ms - now_ms <= 61_000

    def test_every_schedule(self) -> None:
        """Interval schedule adds everyMs to current time."""
        now_ms = 1_000_000_000_000
        schedule = {"kind": "every", "everyMs": 30000}
        next_ms = compute_next_run(schedule, after_ms=now_ms)

        assert next_ms == 1_000_000_030_000

    def test_at_schedule_future(self) -> None:
        """One-shot schedule returns the timestamp if in the future."""
        future = "2099-12-31T23:59:59+00:00"
        schedule = {"kind": "at", "at": future}
        next_ms = compute_next_run(schedule)

        assert next_ms is not None

    def test_at_schedule_past(self) -> None:
        """One-shot schedule returns None if already passed."""
        past = "2000-01-01T00:00:00+00:00"
        schedule = {"kind": "at", "at": past}
        next_ms = compute_next_run(schedule)

        assert next_ms is None

    def test_unknown_kind(self) -> None:
        """Unknown schedule kind returns None."""
        assert compute_next_run({"kind": "unknown"}) is None

    def test_cron_with_timezone(self) -> None:
        """Cron with explicit timezone computes correctly."""
        schedule = {"kind": "cron", "expr": "0 9 * * *", "tz": "Asia/Shanghai"}
        now_ms = int(time.time() * 1000)
        next_ms = compute_next_run(schedule, after_ms=now_ms)
        assert next_ms is not None
        assert next_ms > now_ms


# ---------------------------------------------------------------------------
# Tests: ensure_system_jobs
# ---------------------------------------------------------------------------


class TestEnsureSystemJobs:
    def test_creates_system_file(self, state_dir: Path) -> None:
        ensure_system_jobs(state_dir)
        path = state_dir / "system.json5"
        assert path.exists()

        data = _load_json5(path)
        assert data["version"] == 1
        assert len(data["jobs"]) == 2

        ids = {j["id"] for j in data["jobs"]}
        assert "sys:registry-refresh" in ids
        assert "sys:health-check" in ids

    def test_does_not_overwrite(self, state_dir: Path) -> None:
        """If system.json5 already exists, leave it alone."""
        path = state_dir / "system.json5"
        custom = {"version": 1, "jobs": [{"id": "sys:custom", "name": "custom"}]}
        _save_json5_atomic(path, custom)

        ensure_system_jobs(state_dir)

        data = _load_json5(path)
        assert len(data["jobs"]) == 1  # not overwritten
        assert data["jobs"][0]["id"] == "sys:custom"


# ---------------------------------------------------------------------------
# Tests: CronScheduler
# ---------------------------------------------------------------------------


class TestCronScheduler:
    def _create_scheduler(self, state_dir: Path, agent: Any = None) -> CronScheduler:
        """Helper to create a CronScheduler with no agent."""
        ensure_system_jobs(state_dir)
        return CronScheduler(state_dir=state_dir, agent=agent)

    def test_load_empty_stores(self, state_dir: Path) -> None:
        sched = self._create_scheduler(state_dir)
        sched._load_stores()

        # system.json5 has 2 default jobs, jobs.json5 is empty
        assert len(sched._system_jobs) == 2
        assert len(sched._agent_jobs) == 0

    def test_load_agent_jobs(self, state_dir: Path) -> None:
        jobs_path = state_dir / "jobs.json5"
        _write_jobs(jobs_path, [_make_job()])

        sched = self._create_scheduler(state_dir)
        sched._load_stores()

        assert len(sched._agent_jobs) == 1
        assert sched._agent_jobs[0]["id"] == "test-001"

    def test_external_edit_detection(self, state_dir: Path) -> None:
        """Scheduler detects mtime changes and reloads."""
        jobs_path = state_dir / "jobs.json5"
        _write_jobs(jobs_path, [_make_job(name="original")])

        sched = self._create_scheduler(state_dir)
        sched._load_stores()
        assert sched._agent_jobs[0]["name"] == "original"

        # Simulate external edit
        time.sleep(0.05)  # ensure mtime differs
        _write_jobs(jobs_path, [_make_job(name="edited")])

        sched._maybe_reload()
        assert sched._agent_jobs[0]["name"] == "edited"

    def test_startup_recovery_missed_job(self, state_dir: Path) -> None:
        """Jobs missed within grace period are scheduled to fire immediately."""
        past_ms = int(time.time() * 1000) - 60_000  # 1 minute ago
        job = _make_job(state={"nextRunAtMs": past_ms})
        _write_jobs(state_dir / "jobs.json5", [job])

        sched = self._create_scheduler(state_dir)
        sched._load_stores()
        sched._startup_recovery()

        # nextRunAtMs should still be in the past (fire immediately)
        next_run = sched._agent_jobs[0]["state"]["nextRunAtMs"]
        assert next_run <= past_ms

    def test_startup_recovery_stale_job(self, state_dir: Path) -> None:
        """Jobs missed beyond grace period are advanced to next future time."""
        stale_ms = int(time.time() * 1000) - 7200_000  # 2 hours ago
        job = _make_job(state={"nextRunAtMs": stale_ms})
        _write_jobs(state_dir / "jobs.json5", [job])

        sched = self._create_scheduler(state_dir)
        sched._load_stores()
        sched._startup_recovery()

        now_ms = int(time.time() * 1000)
        next_run = sched._agent_jobs[0]["state"]["nextRunAtMs"]
        assert next_run > now_ms  # advanced to future

    @pytest.mark.asyncio
    async def test_tick_fires_due_job(self, state_dir: Path) -> None:
        """Tick fires a job that is past due."""
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value="done")

        past_ms = int(time.time() * 1000) - 1000
        job = _make_job(state={"nextRunAtMs": past_ms})
        _write_jobs(state_dir / "jobs.json5", [job])

        sched = self._create_scheduler(state_dir, agent=mock_agent)
        sched._load_stores()

        await sched.tick()

        # Wait a moment for the spawned task to complete
        await asyncio.sleep(0.1)
        sched._reap_finished()

        mock_agent.run.assert_called_once_with("Do something.")

    @pytest.mark.asyncio
    async def test_tick_skips_disabled_job(self, state_dir: Path) -> None:
        """Disabled jobs are not fired."""
        mock_agent = AsyncMock()

        past_ms = int(time.time() * 1000) - 1000
        job = _make_job(enabled=False, state={"nextRunAtMs": past_ms})
        _write_jobs(state_dir / "jobs.json5", [job])

        sched = self._create_scheduler(state_dir, agent=mock_agent)
        sched._load_stores()

        await sched.tick()
        await asyncio.sleep(0.1)

        mock_agent.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_tick_skips_future_job(self, state_dir: Path) -> None:
        """Jobs not yet due are not fired."""
        mock_agent = AsyncMock()

        future_ms = int(time.time() * 1000) + 3_600_000  # 1 hour from now
        job = _make_job(state={"nextRunAtMs": future_ms})
        _write_jobs(state_dir / "jobs.json5", [job])

        sched = self._create_scheduler(state_dir, agent=mock_agent)
        sched._load_stores()

        await sched.tick()
        await asyncio.sleep(0.1)

        mock_agent.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_after_run(self, state_dir: Path) -> None:
        """One-shot jobs are removed after execution."""
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value="done")

        past_ms = int(time.time() * 1000) - 1000
        job = _make_job(delete_after_run=True, state={"nextRunAtMs": past_ms})
        _write_jobs(state_dir / "jobs.json5", [job])

        sched = self._create_scheduler(state_dir, agent=mock_agent)
        sched._load_stores()
        assert len(sched._agent_jobs) == 1

        await sched.tick()
        await asyncio.sleep(0.2)

        # Job should be removed
        assert len(sched._agent_jobs) == 0

        # Verify persisted
        loaded = _load_json5(state_dir / "jobs.json5")
        assert len(loaded["jobs"]) == 0

    @pytest.mark.asyncio
    async def test_progressive_error_handling(self, state_dir: Path) -> None:
        """Jobs accumulate error count and get disabled at threshold."""
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=RuntimeError("boom"))

        past_ms = int(time.time() * 1000) - 1000
        job = _make_job(state={"nextRunAtMs": past_ms, "consecutiveErrors": 4})
        _write_jobs(state_dir / "jobs.json5", [job])

        sched = self._create_scheduler(state_dir, agent=mock_agent)
        sched._load_stores()

        await sched.tick()
        await asyncio.sleep(0.2)

        # Error count should now be 5 → auto-disabled
        assert sched._agent_jobs[0]["state"]["consecutiveErrors"] == 5
        assert sched._agent_jobs[0]["enabled"] is False

    @pytest.mark.asyncio
    async def test_concurrent_job_cap(self, state_dir: Path) -> None:
        """No more than MAX_CONCURRENT_JOBS run simultaneously."""
        blocker = asyncio.Event()

        async def slow_run(prompt: str) -> str:
            await blocker.wait()
            return "done"

        slow_agent = AsyncMock()
        slow_agent.run = AsyncMock(side_effect=slow_run)

        past_ms = int(time.time() * 1000) - 1000
        jobs = [_make_job(job_id=f"job-{i}", name=f"job-{i}", state={"nextRunAtMs": past_ms}) for i in range(5)]
        _write_jobs(state_dir / "jobs.json5", jobs)

        sched = self._create_scheduler(state_dir, agent=slow_agent)
        sched._load_stores()

        await sched.tick()
        await asyncio.sleep(0.05)

        # Should have at most 3 running
        assert len(sched._running) <= 3

        # Clean up running tasks
        blocker.set()
        await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_reentrance_guard(self, state_dir: Path) -> None:
        """A job that is already running is not fired again."""
        blocker = asyncio.Event()

        async def slow_run(prompt: str) -> str:
            await blocker.wait()
            return "done"

        slow_agent = AsyncMock()
        slow_agent.run = AsyncMock(side_effect=slow_run)

        past_ms = int(time.time() * 1000) - 1000
        job = _make_job(state={"nextRunAtMs": past_ms})
        _write_jobs(state_dir / "jobs.json5", [job])

        sched = self._create_scheduler(state_dir, agent=slow_agent)
        sched._load_stores()

        # First tick fires the job
        await sched.tick()
        await asyncio.sleep(0.05)
        assert "test-001" in sched._running

        # Second tick should NOT fire it again
        sched._agent_jobs[0]["state"]["nextRunAtMs"] = past_ms
        call_count_before = slow_agent.run.call_count
        await sched.tick()
        await asyncio.sleep(0.05)
        assert slow_agent.run.call_count == call_count_before  # no additional calls

        # Clean up
        blocker.set()
        await asyncio.sleep(0.1)

    def test_save_persists_state(self, state_dir: Path) -> None:
        """Agent job state changes are persisted to disk."""
        job = _make_job(state={"runCount": 42})
        _write_jobs(state_dir / "jobs.json5", [job])

        sched = self._create_scheduler(state_dir)
        sched._load_stores()

        # Modify and save
        sched._agent_jobs[0]["state"]["runCount"] = 43
        sched._save_agent_jobs()

        loaded = _load_json5(state_dir / "jobs.json5")
        assert loaded["jobs"][0]["state"]["runCount"] == 43

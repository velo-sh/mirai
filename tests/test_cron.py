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
        sched._loop = asyncio.get_running_loop()

        await sched._on_timer()

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
        sched._loop = asyncio.get_running_loop()

        await sched._on_timer()
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
        sched._loop = asyncio.get_running_loop()

        await sched._on_timer()
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
        sched._loop = asyncio.get_running_loop()

        await sched._on_timer()
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
        sched._loop = asyncio.get_running_loop()

        await sched._on_timer()
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
        sched._loop = asyncio.get_running_loop()

        await sched._on_timer()
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
        sched._loop = asyncio.get_running_loop()

        # First tick fires the job
        await sched._on_timer()
        await asyncio.sleep(0.05)
        assert "test-001" in sched._running

        # Second tick should NOT fire it again
        sched._agent_jobs[0]["state"]["nextRunAtMs"] = past_ms
        call_count_before = slow_agent.run.call_count
        await sched._on_timer()
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


# ---------------------------------------------------------------------------
# QA: Edge cases in JSON5 I/O
# ---------------------------------------------------------------------------


class TestJson5IOEdgeCases:
    def test_load_empty_file(self, state_dir: Path) -> None:
        """Empty file should return safe default."""
        p = state_dir / "empty.json5"
        p.write_text("", encoding="utf-8")
        data = _load_json5(p)
        assert data == {"version": 1, "jobs": []}

    def test_load_valid_json_no_jobs_key(self, state_dir: Path) -> None:
        """Valid JSON5 but missing 'jobs' key."""
        p = state_dir / "nojobs.json5"
        p.write_text('{"version": 1}', encoding="utf-8")
        data = _load_json5(p)
        assert data["version"] == 1
        assert "jobs" not in data  # key genuinely missing

    def test_unicode_content_roundtrip(self, state_dir: Path) -> None:
        """Job with Unicode characters survives round-trip."""
        path = state_dir / "unicode.json5"
        job = _make_job(name="定时任务-🕐", prompt="执行备份 → 完成 ✅")
        _save_json5_atomic(path, {"version": 1, "jobs": [job]})

        loaded = _load_json5(path)
        assert loaded["jobs"][0]["name"] == "定时任务-🕐"
        assert loaded["jobs"][0]["payload"]["prompt"] == "执行备份 → 完成 ✅"

    def test_large_job_list_roundtrip(self, state_dir: Path) -> None:
        """100 jobs survive round-trip without corruption."""
        path = state_dir / "large.json5"
        jobs = [_make_job(job_id=f"job-{i:03d}", name=f"bulk-{i}") for i in range(100)]
        _save_json5_atomic(path, {"version": 1, "jobs": jobs})

        loaded = _load_json5(path)
        assert len(loaded["jobs"]) == 100
        assert loaded["jobs"][42]["id"] == "job-042"

    def test_save_overwrites_cleanly(self, state_dir: Path) -> None:
        """Second save fully replaces first save content."""
        path = state_dir / "overwrite.json5"
        _save_json5_atomic(path, {"version": 1, "jobs": [_make_job(job_id="old")]})
        _save_json5_atomic(path, {"version": 1, "jobs": [_make_job(job_id="new")]})

        loaded = _load_json5(path)
        assert len(loaded["jobs"]) == 1
        assert loaded["jobs"][0]["id"] == "new"

    def test_backup_reflects_latest_save(self, state_dir: Path) -> None:
        """Backup file contains the latest saved content."""
        path = state_dir / "bak_test.json5"
        _save_json5_atomic(path, {"version": 1, "jobs": [_make_job(job_id="v1")]})
        _save_json5_atomic(path, {"version": 1, "jobs": [_make_job(job_id="v2")]})

        bak = path.with_suffix(".json5.bak")
        loaded = _load_json5(bak)
        assert loaded["jobs"][0]["id"] == "v2"


# ---------------------------------------------------------------------------
# QA: Schedule computation edge cases
# ---------------------------------------------------------------------------


class TestScheduleComputationEdgeCases:
    def test_empty_schedule_dict(self) -> None:
        """Empty schedule returns None."""
        assert compute_next_run({}) is None

    def test_missing_kind_key(self) -> None:
        """Schedule dict without 'kind' returns None."""
        assert compute_next_run({"expr": "* * * * *"}) is None

    def test_every_zero_interval(self) -> None:
        """Zero interval returns current time (edge but valid)."""
        now_ms = 1_000_000_000_000
        result = compute_next_run({"kind": "every", "everyMs": 0}, after_ms=now_ms)
        assert result == now_ms  # fires immediately

    def test_every_very_large_interval(self) -> None:
        """Very large interval doesn't overflow."""
        now_ms = 1_000_000_000_000
        big_interval = 365 * 24 * 3600 * 1000  # 1 year in ms
        result = compute_next_run({"kind": "every", "everyMs": big_interval}, after_ms=now_ms)
        assert result == now_ms + big_interval

    def test_cron_complex_expression(self) -> None:
        """Complex cron (weekday, specific hours) computes without error."""
        schedule = {"kind": "cron", "expr": "30 9,17 * * 1-5"}
        now_ms = int(time.time() * 1000)
        result = compute_next_run(schedule, after_ms=now_ms)
        assert result is not None
        assert result > now_ms

    def test_cron_invalid_timezone_fallback(self) -> None:
        """Invalid timezone falls back to UTC without crashing."""
        schedule = {"kind": "cron", "expr": "* * * * *", "tz": "Invalid/Timezone"}
        now_ms = int(time.time() * 1000)
        result = compute_next_run(schedule, after_ms=now_ms)
        assert result is not None  # should succeed with UTC fallback

    def test_at_exact_now_boundary(self) -> None:
        """@at schedule at exactly 'now' returns None (must be strictly future)."""
        # Use a time just barely in the past
        barely_past = int(time.time() * 1000) - 1
        schedule = {"kind": "at", "at": "2000-01-01T00:00:00+00:00"}
        result = compute_next_run(schedule, after_ms=barely_past)
        assert result is None

    def test_every_default_interval(self) -> None:
        """Omitting everyMs uses 60000ms default."""
        now_ms = 1_000_000_000_000
        result = compute_next_run({"kind": "every"}, after_ms=now_ms)
        assert result == now_ms + 60000

    def test_cron_every_5_minutes(self) -> None:
        """'*/5 * * * *' fires within 5 minutes."""
        schedule = {"kind": "cron", "expr": "*/5 * * * *"}
        now_ms = int(time.time() * 1000)
        result = compute_next_run(schedule, after_ms=now_ms)
        assert result is not None
        assert result - now_ms <= 300_000  # within 5 minutes


# ---------------------------------------------------------------------------
# QA: Error handling scenarios
# ---------------------------------------------------------------------------


class TestErrorHandlingScenarios:
    def _create_scheduler(self, state_dir: Path, agent: Any = None, im_provider: Any = None) -> CronScheduler:
        ensure_system_jobs(state_dir)
        sched = CronScheduler(state_dir=state_dir, agent=agent)
        if im_provider:
            sched.im_provider = im_provider
        return sched

    @pytest.mark.asyncio
    async def test_error_counter_resets_on_success(self, state_dir: Path) -> None:
        """After a successful run, consecutiveErrors resets to 0."""
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value="ok")

        past_ms = int(time.time() * 1000) - 1000
        job = _make_job(state={"nextRunAtMs": past_ms, "consecutiveErrors": 4})
        _write_jobs(state_dir / "jobs.json5", [job])

        sched = self._create_scheduler(state_dir, agent=mock_agent)
        sched._load_stores()
        sched._loop = asyncio.get_running_loop()

        await sched._on_timer()
        await asyncio.sleep(0.2)

        assert sched._agent_jobs[0]["state"]["consecutiveErrors"] == 0
        assert sched._agent_jobs[0]["state"]["lastStatus"] == "ok"
        assert sched._agent_jobs[0]["enabled"] is True  # NOT disabled

    @pytest.mark.asyncio
    async def test_warn_threshold_without_disable(self, state_dir: Path) -> None:
        """At 3 errors, job is warned but NOT disabled."""
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=RuntimeError("fail"))
        mock_im = AsyncMock()
        mock_im.send_markdown = AsyncMock()

        past_ms = int(time.time() * 1000) - 1000
        job = _make_job(state={"nextRunAtMs": past_ms, "consecutiveErrors": 2})
        _write_jobs(state_dir / "jobs.json5", [job])

        sched = self._create_scheduler(state_dir, agent=mock_agent, im_provider=mock_im)
        sched._load_stores()
        sched._loop = asyncio.get_running_loop()

        await sched._on_timer()
        await asyncio.sleep(0.2)

        assert sched._agent_jobs[0]["state"]["consecutiveErrors"] == 3
        assert sched._agent_jobs[0]["enabled"] is True  # still enabled
        mock_im.send_markdown.assert_called()  # alert sent

    @pytest.mark.asyncio
    async def test_alert_curator_failure_does_not_crash(self, state_dir: Path) -> None:
        """If IM provider throws, the error is swallowed gracefully."""
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=RuntimeError("fail"))
        mock_im = AsyncMock()
        mock_im.send_markdown = AsyncMock(side_effect=ConnectionError("IM down"))

        past_ms = int(time.time() * 1000) - 1000
        job = _make_job(state={"nextRunAtMs": past_ms, "consecutiveErrors": 2})
        _write_jobs(state_dir / "jobs.json5", [job])

        sched = self._create_scheduler(state_dir, agent=mock_agent, im_provider=mock_im)
        sched._load_stores()
        sched._loop = asyncio.get_running_loop()

        # Should not raise despite IM failure
        await sched._on_timer()
        await asyncio.sleep(0.2)

        assert sched._agent_jobs[0]["state"]["consecutiveErrors"] == 3

    @pytest.mark.asyncio
    async def test_last_error_is_truncated(self, state_dir: Path) -> None:
        """Error messages longer than 200 chars are truncated in state."""
        mock_agent = AsyncMock()
        long_error = "x" * 500
        mock_agent.run = AsyncMock(side_effect=RuntimeError(long_error))

        past_ms = int(time.time() * 1000) - 1000
        job = _make_job(state={"nextRunAtMs": past_ms})
        _write_jobs(state_dir / "jobs.json5", [job])

        sched = self._create_scheduler(state_dir, agent=mock_agent)
        sched._load_stores()
        sched._loop = asyncio.get_running_loop()

        await sched._on_timer()
        await asyncio.sleep(0.2)

        assert len(sched._agent_jobs[0]["state"]["lastError"]) == 200

    @pytest.mark.asyncio
    async def test_disabled_job_not_re_enabled_by_recovery(self, state_dir: Path) -> None:
        """Startup recovery does not re-enable a disabled job."""
        past_ms = int(time.time() * 1000) - 60_000
        job = _make_job(enabled=False, state={"nextRunAtMs": past_ms, "consecutiveErrors": 5})
        _write_jobs(state_dir / "jobs.json5", [job])

        sched = self._create_scheduler(state_dir)
        sched._load_stores()
        sched._startup_recovery()

        assert sched._agent_jobs[0]["enabled"] is False


# ---------------------------------------------------------------------------
# QA: Multi-job mixed state scenarios
# ---------------------------------------------------------------------------


class TestMultiJobScenarios:
    def _create_scheduler(self, state_dir: Path, agent: Any = None) -> CronScheduler:
        ensure_system_jobs(state_dir)
        return CronScheduler(state_dir=state_dir, agent=agent)

    @pytest.mark.asyncio
    async def test_mixed_due_and_future_jobs(self, state_dir: Path) -> None:
        """Only due jobs fire; future jobs remain untouched."""
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value="ok")

        now = int(time.time() * 1000)
        jobs = [
            _make_job(job_id="due-1", state={"nextRunAtMs": now - 1000}),
            _make_job(job_id="future-1", state={"nextRunAtMs": now + 3_600_000}),
            _make_job(job_id="due-2", state={"nextRunAtMs": now - 500}),
        ]
        _write_jobs(state_dir / "jobs.json5", jobs)

        sched = self._create_scheduler(state_dir, agent=mock_agent)
        sched._load_stores()
        sched._loop = asyncio.get_running_loop()

        await sched._on_timer()
        await asyncio.sleep(0.2)

        assert mock_agent.run.call_count == 2  # only 2 due jobs

    @pytest.mark.asyncio
    async def test_mixed_enabled_disabled_jobs(self, state_dir: Path) -> None:
        """Disabled jobs are skipped even if due."""
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value="ok")

        past_ms = int(time.time() * 1000) - 1000
        jobs = [
            _make_job(job_id="active", enabled=True, state={"nextRunAtMs": past_ms}),
            _make_job(job_id="disabled", enabled=False, state={"nextRunAtMs": past_ms}),
            _make_job(job_id="also-active", enabled=True, state={"nextRunAtMs": past_ms}),
        ]
        _write_jobs(state_dir / "jobs.json5", jobs)

        sched = self._create_scheduler(state_dir, agent=mock_agent)
        sched._load_stores()
        sched._loop = asyncio.get_running_loop()

        await sched._on_timer()
        await asyncio.sleep(0.2)

        assert mock_agent.run.call_count == 2

    @pytest.mark.asyncio
    async def test_oneshot_among_recurring(self, state_dir: Path) -> None:
        """One-shot job is deleted while recurring job survives."""
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value="ok")

        past_ms = int(time.time() * 1000) - 1000
        jobs = [
            _make_job(job_id="recurring", state={"nextRunAtMs": past_ms}),
            _make_job(job_id="oneshot", delete_after_run=True, state={"nextRunAtMs": past_ms}),
        ]
        _write_jobs(state_dir / "jobs.json5", jobs)

        sched = self._create_scheduler(state_dir, agent=mock_agent)
        sched._load_stores()
        sched._loop = asyncio.get_running_loop()

        await sched._on_timer()
        await asyncio.sleep(0.3)

        assert len(sched._agent_jobs) == 1
        assert sched._agent_jobs[0]["id"] == "recurring"

    @pytest.mark.asyncio
    async def test_job_with_no_next_run_is_skipped(self, state_dir: Path) -> None:
        """Job without nextRunAtMs in state is not fired."""
        mock_agent = AsyncMock()
        job = _make_job(state={})  # no nextRunAtMs
        _write_jobs(state_dir / "jobs.json5", [job])

        sched = self._create_scheduler(state_dir, agent=mock_agent)
        sched._load_stores()
        sched._loop = asyncio.get_running_loop()

        await sched._on_timer()
        await asyncio.sleep(0.1)

        mock_agent.run.assert_not_called()


# ---------------------------------------------------------------------------
# QA: No-agent and empty-prompt scenarios
# ---------------------------------------------------------------------------


class TestNoAgentScenarios:
    def _create_scheduler(self, state_dir: Path, **kwargs: Any) -> CronScheduler:
        ensure_system_jobs(state_dir)
        return CronScheduler(state_dir=state_dir, **kwargs)

    @pytest.mark.asyncio
    async def test_fire_without_agent(self, state_dir: Path) -> None:
        """Job fires without error when agent is None (no-op)."""
        past_ms = int(time.time() * 1000) - 1000
        job = _make_job(state={"nextRunAtMs": past_ms})
        _write_jobs(state_dir / "jobs.json5", [job])

        sched = self._create_scheduler(state_dir, agent=None)
        sched._load_stores()
        sched._loop = asyncio.get_running_loop()

        # Should not crash
        await sched._on_timer()
        await asyncio.sleep(0.2)

        # Job completes successfully (no-op)
        assert sched._agent_jobs[0]["state"]["lastStatus"] == "ok"
        assert sched._agent_jobs[0]["state"]["runCount"] == 1

    @pytest.mark.asyncio
    async def test_job_with_empty_prompt_is_skipped(self, state_dir: Path) -> None:
        """Job with empty prompt string is not fired."""
        mock_agent = AsyncMock()

        past_ms = int(time.time() * 1000) - 1000
        job = _make_job(prompt="", state={"nextRunAtMs": past_ms})
        _write_jobs(state_dir / "jobs.json5", [job])

        sched = self._create_scheduler(state_dir, agent=mock_agent)
        sched._load_stores()
        sched._loop = asyncio.get_running_loop()

        await sched._on_timer()
        await asyncio.sleep(0.1)

        mock_agent.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_job_with_missing_payload(self, state_dir: Path) -> None:
        """Job dict without payload key is handled gracefully."""
        mock_agent = AsyncMock()

        past_ms = int(time.time() * 1000) - 1000
        job = {
            "id": "no-payload",
            "name": "missing",
            "enabled": True,
            "schedule": {"kind": "every", "everyMs": 60000},
            "state": {"nextRunAtMs": past_ms},
            # no "payload" key
        }
        _write_jobs(state_dir / "jobs.json5", [job])

        sched = self._create_scheduler(state_dir, agent=mock_agent)
        sched._load_stores()
        sched._loop = asyncio.get_running_loop()

        # Should not crash
        await sched._on_timer()
        await asyncio.sleep(0.1)

        mock_agent.run.assert_not_called()


# ---------------------------------------------------------------------------
# QA: System job protection
# ---------------------------------------------------------------------------


class TestSystemJobProtection:
    @pytest.mark.asyncio
    async def test_system_jobs_are_not_persisted_by_save(self, state_dir: Path) -> None:
        """_save_agent_jobs() does NOT write system jobs to jobs.json5."""
        ensure_system_jobs(state_dir)
        sched = CronScheduler(state_dir=state_dir)
        sched._load_stores()

        # Mutate system job state in memory
        sched._system_jobs[0]["state"]["runCount"] = 999

        # Save should only persist agent jobs
        sched._save_agent_jobs()

        # jobs.json5 should not contain system jobs
        loaded = _load_json5(state_dir / "jobs.json5")
        job_ids = {j["id"] for j in loaded.get("jobs", [])}
        assert "sys:registry-refresh" not in job_ids

    def test_system_jobs_loaded_separately(self, state_dir: Path) -> None:
        """System and agent jobs are stored in separate lists."""
        ensure_system_jobs(state_dir)
        _write_jobs(state_dir / "jobs.json5", [_make_job(job_id="agent:myapp:task-1")])

        sched = CronScheduler(state_dir=state_dir)
        sched._load_stores()

        assert len(sched._system_jobs) == 2
        assert len(sched._agent_jobs) == 1
        assert sched._agent_jobs[0]["id"] == "agent:myapp:task-1"


# ---------------------------------------------------------------------------
# QA: ensure_system_jobs edge cases
# ---------------------------------------------------------------------------


class TestEnsureSystemJobsEdgeCases:
    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Works even if state_dir doesn't exist yet."""
        deep_dir = tmp_path / "deep" / "nested" / "cron"
        ensure_system_jobs(deep_dir)
        assert (deep_dir / "system.json5").exists()

    def test_system_jobs_all_disabled_by_default(self, state_dir: Path) -> None:
        """All default system jobs start disabled (not yet migrated)."""
        ensure_system_jobs(state_dir)
        data = _load_json5(state_dir / "system.json5")
        for job in data["jobs"]:
            assert job["enabled"] is False, f"System job {job['id']} should be disabled by default"

    def test_system_jobs_have_required_fields(self, state_dir: Path) -> None:
        """All system jobs have id, name, schedule, payload, state."""
        ensure_system_jobs(state_dir)
        data = _load_json5(state_dir / "system.json5")
        required = {"id", "name", "enabled", "schedule", "payload", "state"}
        for job in data["jobs"]:
            assert required.issubset(job.keys()), f"Job {job.get('id')} missing fields: {required - job.keys()}"


# ---------------------------------------------------------------------------
# QA: Startup recovery edge cases
# ---------------------------------------------------------------------------


class TestStartupRecoveryEdgeCases:
    def _create_scheduler(self, state_dir: Path) -> CronScheduler:
        ensure_system_jobs(state_dir)
        return CronScheduler(state_dir=state_dir)

    def test_recovery_with_no_jobs(self, state_dir: Path) -> None:
        """Recovery with empty job list completes without error."""
        sched = self._create_scheduler(state_dir)
        sched._load_stores()
        sched._startup_recovery()  # no crash

    def test_recovery_assigns_next_run_to_unscheduled(self, state_dir: Path) -> None:
        """Jobs that never ran get an initial nextRunAtMs."""
        job = _make_job(state={})  # no nextRunAtMs
        _write_jobs(state_dir / "jobs.json5", [job])

        sched = self._create_scheduler(state_dir)
        sched._load_stores()
        sched._startup_recovery()

        assert "nextRunAtMs" in sched._agent_jobs[0]["state"]
        assert sched._agent_jobs[0]["state"]["nextRunAtMs"] > int(time.time() * 1000)

    def test_recovery_skips_disabled_jobs(self, state_dir: Path) -> None:
        """Disabled jobs are not touched during recovery."""
        job = _make_job(enabled=False, state={})
        _write_jobs(state_dir / "jobs.json5", [job])

        sched = self._create_scheduler(state_dir)
        sched._load_stores()
        sched._startup_recovery()

        # No nextRunAtMs should be computed for disabled jobs
        assert sched._agent_jobs[0]["state"].get("nextRunAtMs") is None


# ---------------------------------------------------------------------------
# QA: State persistence edge cases
# ---------------------------------------------------------------------------


class TestStatePersistenceEdgeCases:
    def _create_scheduler(self, state_dir: Path, agent: Any = None) -> CronScheduler:
        ensure_system_jobs(state_dir)
        return CronScheduler(state_dir=state_dir, agent=agent)

    @pytest.mark.asyncio
    async def test_state_persists_after_error(self, state_dir: Path) -> None:
        """Error state is persisted to disk even after failure."""
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=RuntimeError("boom"))

        past_ms = int(time.time() * 1000) - 1000
        job = _make_job(state={"nextRunAtMs": past_ms})
        _write_jobs(state_dir / "jobs.json5", [job])

        sched = self._create_scheduler(state_dir, agent=mock_agent)
        sched._load_stores()
        sched._loop = asyncio.get_running_loop()

        await sched._on_timer()
        await asyncio.sleep(0.2)

        # Reload from disk and verify error state was persisted
        loaded = _load_json5(state_dir / "jobs.json5")
        assert loaded["jobs"][0]["state"]["lastStatus"] == "error"
        assert loaded["jobs"][0]["state"]["consecutiveErrors"] == 1
        assert "boom" in loaded["jobs"][0]["state"]["lastError"]

    @pytest.mark.asyncio
    async def test_run_count_increments(self, state_dir: Path) -> None:
        """runCount increments with each successful execution."""
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value="ok")

        past_ms = int(time.time() * 1000) - 1000
        job = _make_job(state={"nextRunAtMs": past_ms, "runCount": 10})
        _write_jobs(state_dir / "jobs.json5", [job])

        sched = self._create_scheduler(state_dir, agent=mock_agent)
        sched._load_stores()
        sched._loop = asyncio.get_running_loop()

        await sched._on_timer()
        await asyncio.sleep(0.2)

        assert sched._agent_jobs[0]["state"]["runCount"] == 11

    @pytest.mark.asyncio
    async def test_last_run_timestamp_updated(self, state_dir: Path) -> None:
        """lastRunAtMs is updated after execution."""
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value="ok")

        past_ms = int(time.time() * 1000) - 1000
        job = _make_job(state={"nextRunAtMs": past_ms})
        _write_jobs(state_dir / "jobs.json5", [job])

        sched = self._create_scheduler(state_dir, agent=mock_agent)
        sched._load_stores()
        sched._loop = asyncio.get_running_loop()

        before = int(time.time() * 1000)
        await sched._on_timer()
        await asyncio.sleep(0.2)
        after = int(time.time() * 1000)

        last_run = sched._agent_jobs[0]["state"]["lastRunAtMs"]
        assert before <= last_run <= after

    @pytest.mark.asyncio
    async def test_next_run_recomputed_after_success(self, state_dir: Path) -> None:
        """After successful run, nextRunAtMs is advanced to future."""
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value="ok")

        past_ms = int(time.time() * 1000) - 1000
        job = _make_job(
            schedule={"kind": "every", "everyMs": 60000},
            state={"nextRunAtMs": past_ms},
        )
        _write_jobs(state_dir / "jobs.json5", [job])

        sched = self._create_scheduler(state_dir, agent=mock_agent)
        sched._load_stores()
        sched._loop = asyncio.get_running_loop()

        await sched._on_timer()
        await asyncio.sleep(0.2)

        now = int(time.time() * 1000)
        next_run = sched._agent_jobs[0]["state"]["nextRunAtMs"]
        assert next_run > now  # advanced to future


# ---------------------------------------------------------------------------
# QA: Reap finished edge cases
# ---------------------------------------------------------------------------


class TestReapFinished:
    def _create_scheduler(self, state_dir: Path) -> CronScheduler:
        ensure_system_jobs(state_dir)
        return CronScheduler(state_dir=state_dir)

    def test_reap_empty_running(self, state_dir: Path) -> None:
        """Reap with nothing running is a no-op."""
        sched = self._create_scheduler(state_dir)
        sched._reap_finished()  # no crash
        assert len(sched._running) == 0

    @pytest.mark.asyncio
    async def test_reap_completed_task(self, state_dir: Path) -> None:
        """Completed tasks are removed from _running."""
        from mirai.cron import _RunningJob

        sched = self._create_scheduler(state_dir)

        # Create a task that completes immediately
        task = asyncio.get_running_loop().create_task(asyncio.sleep(0))
        sched._running["fake-job"] = _RunningJob(job_id="fake-job", task=task)

        await asyncio.sleep(0.05)  # let task complete
        sched._reap_finished()

        assert "fake-job" not in sched._running

    @pytest.mark.asyncio
    async def test_reap_failed_task(self, state_dir: Path) -> None:
        """Failed tasks are reaped and their exceptions are logged, not raised."""
        from mirai.cron import _RunningJob

        sched = self._create_scheduler(state_dir)

        async def always_fail() -> None:
            raise ValueError("task-level failure")

        task = asyncio.get_running_loop().create_task(always_fail())
        sched._running["fail-job"] = _RunningJob(job_id="fail-job", task=task)

        await asyncio.sleep(0.05)
        sched._reap_finished()  # should not raise

        assert "fail-job" not in sched._running


# ---------------------------------------------------------------------------
# QA: _error_backoff_ms pure function tests
# ---------------------------------------------------------------------------


class TestErrorBackoff:
    """Tests for the exponential error backoff schedule."""

    def test_first_error_30s(self) -> None:
        from mirai.cron import _error_backoff_ms

        assert _error_backoff_ms(1) == 30_000

    def test_second_error_1m(self) -> None:
        from mirai.cron import _error_backoff_ms

        assert _error_backoff_ms(2) == 60_000

    def test_third_error_5m(self) -> None:
        from mirai.cron import _error_backoff_ms

        assert _error_backoff_ms(3) == 5 * 60_000

    def test_fourth_error_15m(self) -> None:
        from mirai.cron import _error_backoff_ms

        assert _error_backoff_ms(4) == 15 * 60_000

    def test_fifth_error_60m(self) -> None:
        from mirai.cron import _error_backoff_ms

        assert _error_backoff_ms(5) == 60 * 60_000

    def test_beyond_fifth_caps_at_60m(self) -> None:
        """Errors beyond 5th still return 60 min (table cap)."""
        from mirai.cron import _error_backoff_ms

        assert _error_backoff_ms(10) == 60 * 60_000
        assert _error_backoff_ms(100) == 60 * 60_000

    def test_zero_errors_treated_as_first(self) -> None:
        """Edge case: 0 errors clamps to index 0."""
        from mirai.cron import _error_backoff_ms

        assert _error_backoff_ms(0) == 30_000


# ---------------------------------------------------------------------------
# QA: Smart timer — _next_wake_ms
# ---------------------------------------------------------------------------


class TestNextWakeMs:
    """Tests for the _next_wake_ms helper that finds the earliest due time."""

    def _create_scheduler(self, state_dir: Path) -> CronScheduler:
        ensure_system_jobs(state_dir)
        return CronScheduler(state_dir=state_dir)

    def test_no_jobs_returns_none(self, state_dir: Path) -> None:
        """With no enabled jobs having nextRunAtMs, returns None."""
        sched = self._create_scheduler(state_dir)
        sched._load_stores()
        # Default system jobs are disabled, so should return None
        assert sched._next_wake_ms() is None

    def test_single_job_returns_its_time(self, state_dir: Path) -> None:
        target_ms = 2_000_000_000_000
        job = _make_job(state={"nextRunAtMs": target_ms})
        _write_jobs(state_dir / "jobs.json5", [job])

        sched = self._create_scheduler(state_dir)
        sched._load_stores()

        assert sched._next_wake_ms() == target_ms

    def test_picks_earliest_among_multiple(self, state_dir: Path) -> None:
        """Returns the smallest nextRunAtMs among all enabled jobs."""
        jobs = [
            _make_job(job_id="a", state={"nextRunAtMs": 3000}),
            _make_job(job_id="b", state={"nextRunAtMs": 1000}),
            _make_job(job_id="c", state={"nextRunAtMs": 2000}),
        ]
        _write_jobs(state_dir / "jobs.json5", jobs)

        sched = self._create_scheduler(state_dir)
        sched._load_stores()

        assert sched._next_wake_ms() == 1000

    def test_disabled_jobs_are_excluded(self, state_dir: Path) -> None:
        """Disabled jobs are not considered for wake time."""
        jobs = [
            _make_job(job_id="disabled", enabled=False, state={"nextRunAtMs": 500}),
            _make_job(job_id="enabled", state={"nextRunAtMs": 5000}),
        ]
        _write_jobs(state_dir / "jobs.json5", jobs)

        sched = self._create_scheduler(state_dir)
        sched._load_stores()

        assert sched._next_wake_ms() == 5000  # ignores the disabled job at 500

    def test_jobs_without_next_run_are_excluded(self, state_dir: Path) -> None:
        """Jobs without nextRunAtMs in state don't affect wake time."""
        jobs = [
            _make_job(job_id="no-state", state={}),
            _make_job(job_id="has-state", state={"nextRunAtMs": 9000}),
        ]
        _write_jobs(state_dir / "jobs.json5", jobs)

        sched = self._create_scheduler(state_dir)
        sched._load_stores()

        assert sched._next_wake_ms() == 9000


# ---------------------------------------------------------------------------
# QA: Smart timer — _arm_timer lifecycle
# ---------------------------------------------------------------------------


class TestArmTimer:
    """Tests for _arm_timer scheduling behavior."""

    def _create_scheduler(self, state_dir: Path) -> CronScheduler:
        ensure_system_jobs(state_dir)
        return CronScheduler(state_dir=state_dir)

    @pytest.mark.asyncio
    async def test_arm_timer_creates_handle(self, state_dir: Path) -> None:
        """Calling _arm_timer sets _timer_handle."""
        sched = self._create_scheduler(state_dir)
        sched._load_stores()
        sched._loop = asyncio.get_running_loop()

        assert sched._timer_handle is None
        sched._arm_timer()
        assert sched._timer_handle is not None

        # Cleanup
        sched._timer_handle.cancel()

    @pytest.mark.asyncio
    async def test_arm_timer_cancels_previous(self, state_dir: Path) -> None:
        """Re-arming cancels the old handle and sets a new one."""
        sched = self._create_scheduler(state_dir)
        sched._load_stores()
        sched._loop = asyncio.get_running_loop()

        sched._arm_timer()
        first_handle = sched._timer_handle
        assert first_handle is not None

        sched._arm_timer()
        second_handle = sched._timer_handle
        assert second_handle is not first_handle
        assert first_handle.cancelled()

        # Cleanup
        if second_handle:
            second_handle.cancel()

    @pytest.mark.asyncio
    async def test_arm_timer_noop_when_stopped(self, state_dir: Path) -> None:
        """_arm_timer does nothing after stop()."""
        sched = self._create_scheduler(state_dir)
        sched._load_stores()
        sched._loop = asyncio.get_running_loop()
        sched._stopped = True

        sched._arm_timer()
        assert sched._timer_handle is None

    @pytest.mark.asyncio
    async def test_arm_timer_noop_without_loop(self, state_dir: Path) -> None:
        """_arm_timer does nothing when _loop is None."""
        sched = self._create_scheduler(state_dir)
        sched._load_stores()
        # _loop is None by default

        sched._arm_timer()
        assert sched._timer_handle is None

    @pytest.mark.asyncio
    async def test_stop_cancels_timer(self, state_dir: Path) -> None:
        """stop() cancels any pending timer handle."""
        sched = self._create_scheduler(state_dir)
        sched._load_stores()
        sched._loop = asyncio.get_running_loop()

        sched._arm_timer()
        assert sched._timer_handle is not None

        sched.stop()
        assert sched._stopped is True
        assert sched._timer_handle is None


# ---------------------------------------------------------------------------
# QA: Duration tracking (lastDurationMs)
# ---------------------------------------------------------------------------


class TestDurationTracking:
    """Tests for lastDurationMs recorded per job execution."""

    def _create_scheduler(self, state_dir: Path, agent: Any = None) -> CronScheduler:
        ensure_system_jobs(state_dir)
        return CronScheduler(state_dir=state_dir, agent=agent)

    @pytest.mark.asyncio
    async def test_duration_recorded_on_success(self, state_dir: Path) -> None:
        """lastDurationMs is set after successful run."""
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value="ok")

        past_ms = int(time.time() * 1000) - 1000
        job = _make_job(state={"nextRunAtMs": past_ms})
        _write_jobs(state_dir / "jobs.json5", [job])

        sched = self._create_scheduler(state_dir, agent=mock_agent)
        sched._load_stores()
        sched._loop = asyncio.get_running_loop()

        await sched._on_timer()
        await asyncio.sleep(0.2)

        duration = sched._agent_jobs[0]["state"].get("lastDurationMs")
        assert duration is not None
        assert duration >= 0

    @pytest.mark.asyncio
    async def test_duration_recorded_on_error(self, state_dir: Path) -> None:
        """lastDurationMs is set even when the job fails."""
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=RuntimeError("fail"))

        past_ms = int(time.time() * 1000) - 1000
        job = _make_job(state={"nextRunAtMs": past_ms})
        _write_jobs(state_dir / "jobs.json5", [job])

        sched = self._create_scheduler(state_dir, agent=mock_agent)
        sched._load_stores()
        sched._loop = asyncio.get_running_loop()

        await sched._on_timer()
        await asyncio.sleep(0.2)

        duration = sched._agent_jobs[0]["state"].get("lastDurationMs")
        assert duration is not None
        assert duration >= 0
        assert sched._agent_jobs[0]["state"]["lastStatus"] == "error"

    @pytest.mark.asyncio
    async def test_duration_persisted_to_disk(self, state_dir: Path) -> None:
        """lastDurationMs survives round-trip to disk."""
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value="ok")

        past_ms = int(time.time() * 1000) - 1000
        job = _make_job(state={"nextRunAtMs": past_ms})
        _write_jobs(state_dir / "jobs.json5", [job])

        sched = self._create_scheduler(state_dir, agent=mock_agent)
        sched._load_stores()
        sched._loop = asyncio.get_running_loop()

        await sched._on_timer()
        await asyncio.sleep(0.2)

        loaded = _load_json5(state_dir / "jobs.json5")
        assert "lastDurationMs" in loaded["jobs"][0]["state"]
        assert loaded["jobs"][0]["state"]["lastDurationMs"] >= 0


# ---------------------------------------------------------------------------
# QA: Error backoff applied to nextRunAtMs
# ---------------------------------------------------------------------------


class TestErrorBackoffApplied:
    """Tests that error backoff delays are applied to next schedule time."""

    def _create_scheduler(self, state_dir: Path, agent: Any = None) -> CronScheduler:
        ensure_system_jobs(state_dir)
        return CronScheduler(state_dir=state_dir, agent=agent)

    @pytest.mark.asyncio
    async def test_first_error_delays_next_run_by_30s(self, state_dir: Path) -> None:
        """After 1 error, nextRunAtMs is pushed back by >= 30s from now."""
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=RuntimeError("boom"))

        past_ms = int(time.time() * 1000) - 1000
        job = _make_job(
            schedule={"kind": "every", "everyMs": 1000},  # very short interval
            state={"nextRunAtMs": past_ms},
        )
        _write_jobs(state_dir / "jobs.json5", [job])

        sched = self._create_scheduler(state_dir, agent=mock_agent)
        sched._load_stores()
        sched._loop = asyncio.get_running_loop()

        await sched._on_timer()
        await asyncio.sleep(0.2)

        now_ms = int(time.time() * 1000)
        next_run = sched._agent_jobs[0]["state"]["nextRunAtMs"]
        # With 1s interval, normal next_run would be ~now+1s
        # Error backoff pushes it to at least now + 30s
        assert next_run >= now_ms + 25_000  # allow small timing tolerance

    @pytest.mark.asyncio
    async def test_no_backoff_on_success(self, state_dir: Path) -> None:
        """After success, nextRunAtMs uses normal schedule (no backoff)."""
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value="ok")

        past_ms = int(time.time() * 1000) - 1000
        job = _make_job(
            schedule={"kind": "every", "everyMs": 60_000},
            state={"nextRunAtMs": past_ms},
        )
        _write_jobs(state_dir / "jobs.json5", [job])

        sched = self._create_scheduler(state_dir, agent=mock_agent)
        sched._load_stores()
        sched._loop = asyncio.get_running_loop()

        await sched._on_timer()
        await asyncio.sleep(0.2)

        now_ms = int(time.time() * 1000)
        next_run = sched._agent_jobs[0]["state"]["nextRunAtMs"]
        # Normal schedule: ~now + 60s (not pushed further by backoff)
        assert next_run <= now_ms + 65_000  # within normal range

    @pytest.mark.asyncio
    async def test_consecutive_errors_increase_backoff(self, state_dir: Path) -> None:
        """More consecutive errors yield longer backoff delays."""
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=RuntimeError("keep failing"))

        past_ms = int(time.time() * 1000) - 1000
        job = _make_job(
            schedule={"kind": "every", "everyMs": 1000},
            state={"nextRunAtMs": past_ms, "consecutiveErrors": 2},  # already at 2
        )
        _write_jobs(state_dir / "jobs.json5", [job])

        sched = self._create_scheduler(state_dir, agent=mock_agent)
        sched._load_stores()
        sched._loop = asyncio.get_running_loop()

        await sched._on_timer()
        await asyncio.sleep(0.2)

        # Now at 3 errors → 5 min backoff
        now_ms = int(time.time() * 1000)
        next_run = sched._agent_jobs[0]["state"]["nextRunAtMs"]
        assert sched._agent_jobs[0]["state"]["consecutiveErrors"] == 3
        assert next_run >= now_ms + (4 * 60_000)  # at least ~4 min (tolerance)

    @pytest.mark.asyncio
    async def test_backoff_resets_after_success(self, state_dir: Path) -> None:
        """Success after errors resets consecutiveErrors and uses normal schedule."""
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value="recovered!")

        past_ms = int(time.time() * 1000) - 1000
        job = _make_job(
            schedule={"kind": "every", "everyMs": 60_000},
            state={"nextRunAtMs": past_ms, "consecutiveErrors": 4},
        )
        _write_jobs(state_dir / "jobs.json5", [job])

        sched = self._create_scheduler(state_dir, agent=mock_agent)
        sched._load_stores()
        sched._loop = asyncio.get_running_loop()

        await sched._on_timer()
        await asyncio.sleep(0.2)

        assert sched._agent_jobs[0]["state"]["consecutiveErrors"] == 0
        now_ms = int(time.time() * 1000)
        next_run = sched._agent_jobs[0]["state"]["nextRunAtMs"]
        # Should be ~now + 60s, not pushed by 15 min backoff
        assert next_run <= now_ms + 65_000


# ---------------------------------------------------------------------------
# QA: _on_timer re-arm and resilience
# ---------------------------------------------------------------------------


class TestOnTimerResilience:
    """Tests that _on_timer correctly re-arms even after internal failures."""

    def _create_scheduler(self, state_dir: Path, agent: Any = None) -> CronScheduler:
        ensure_system_jobs(state_dir)
        return CronScheduler(state_dir=state_dir, agent=agent)

    @pytest.mark.asyncio
    async def test_on_timer_re_arms_after_job_errors(self, state_dir: Path) -> None:
        """Timer re-arms even when all jobs fail."""
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=RuntimeError("fail"))

        past_ms = int(time.time() * 1000) - 1000
        job = _make_job(state={"nextRunAtMs": past_ms})
        _write_jobs(state_dir / "jobs.json5", [job])

        sched = self._create_scheduler(state_dir, agent=mock_agent)
        sched._load_stores()
        sched._loop = asyncio.get_running_loop()

        await sched._on_timer()
        await asyncio.sleep(0.2)

        # Timer should still be armed after the error
        assert sched._timer_handle is not None
        sched._timer_handle.cancel()

    @pytest.mark.asyncio
    async def test_on_timer_skipped_when_stopped(self, state_dir: Path) -> None:
        """_on_timer is a no-op when scheduler is stopped."""
        mock_agent = AsyncMock()

        past_ms = int(time.time() * 1000) - 1000
        job = _make_job(state={"nextRunAtMs": past_ms})
        _write_jobs(state_dir / "jobs.json5", [job])

        sched = self._create_scheduler(state_dir, agent=mock_agent)
        sched._load_stores()
        sched._loop = asyncio.get_running_loop()
        sched._stopped = True

        await sched._on_timer()
        await asyncio.sleep(0.1)

        # Nothing should have fired
        mock_agent.run.assert_not_called()
        assert sched._timer_handle is None

    @pytest.mark.asyncio
    async def test_on_timer_re_arms_with_no_due_jobs(self, state_dir: Path) -> None:
        """Timer re-arms even when there are no due jobs."""
        future_ms = int(time.time() * 1000) + 3_600_000
        job = _make_job(state={"nextRunAtMs": future_ms})
        _write_jobs(state_dir / "jobs.json5", [job])

        sched = self._create_scheduler(state_dir)
        sched._load_stores()
        sched._loop = asyncio.get_running_loop()

        await sched._on_timer()
        await asyncio.sleep(0.05)

        # Timer should be re-armed for the future job
        assert sched._timer_handle is not None
        sched._timer_handle.cancel()

    @pytest.mark.asyncio
    async def test_on_timer_re_arms_with_empty_job_list(self, state_dir: Path) -> None:
        """Timer re-arms even with zero jobs (fallback to MAX_TIMER_DELAY)."""
        sched = self._create_scheduler(state_dir)
        sched._load_stores()
        sched._loop = asyncio.get_running_loop()

        await sched._on_timer()
        await asyncio.sleep(0.05)

        assert sched._timer_handle is not None
        sched._timer_handle.cancel()


# ---------------------------------------------------------------------------
# QA: Delivery-free cron (agent uses im_tool)
# ---------------------------------------------------------------------------


class TestDeliveryFreeCron:
    """Verify cron does NOT deliver messages; agent response is discarded."""

    def _create_scheduler(self, state_dir: Path, agent: Any = None) -> CronScheduler:
        ensure_system_jobs(state_dir)
        return CronScheduler(state_dir=state_dir, agent=agent)

    @pytest.mark.asyncio
    async def test_agent_return_value_is_discarded(self, state_dir: Path) -> None:
        """Agent run() return value is not propagated or used by cron."""
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value="I want to send this to Feishu!")

        past_ms = int(time.time() * 1000) - 1000
        job = _make_job(state={"nextRunAtMs": past_ms})
        _write_jobs(state_dir / "jobs.json5", [job])

        sched = self._create_scheduler(state_dir, agent=mock_agent)
        sched._load_stores()
        sched._loop = asyncio.get_running_loop()

        await sched._on_timer()
        await asyncio.sleep(0.2)

        # Agent was called, but its return value was discarded
        mock_agent.run.assert_called_once()
        # Cron has no im_provider reference for delivery
        assert sched.im_provider is None
        # Job still marked as successful
        assert sched._agent_jobs[0]["state"]["lastStatus"] == "ok"

    @pytest.mark.asyncio
    async def test_im_provider_not_used_for_delivery(self, state_dir: Path) -> None:
        """Even if im_provider is set, it is NOT used for job result delivery."""
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value="result text")
        mock_im = AsyncMock()
        mock_im.send_message = AsyncMock()
        mock_im.send_markdown = AsyncMock()

        past_ms = int(time.time() * 1000) - 1000
        job = _make_job(state={"nextRunAtMs": past_ms})
        _write_jobs(state_dir / "jobs.json5", [job])

        sched = self._create_scheduler(state_dir, agent=mock_agent)
        sched.im_provider = mock_im
        sched._load_stores()
        sched._loop = asyncio.get_running_loop()

        await sched._on_timer()
        await asyncio.sleep(0.2)

        # Agent was called
        mock_agent.run.assert_called_once()
        # IM provider was NOT called for delivery (only for alerts on errors)
        mock_im.send_message.assert_not_called()
        mock_im.send_markdown.assert_not_called()

    @pytest.mark.asyncio
    async def test_im_provider_used_for_curator_alerts_only(self, state_dir: Path) -> None:
        """im_provider.send_markdown is only called for error alerts, not delivery."""
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=RuntimeError("critical failure"))
        mock_im = AsyncMock()
        mock_im.send_markdown = AsyncMock()

        past_ms = int(time.time() * 1000) - 1000
        # 3rd consecutive error triggers WARN_THRESHOLD alert
        job = _make_job(state={"nextRunAtMs": past_ms, "consecutiveErrors": 2})
        _write_jobs(state_dir / "jobs.json5", [job])

        sched = self._create_scheduler(state_dir, agent=mock_agent)
        sched.im_provider = mock_im
        sched._load_stores()
        sched._loop = asyncio.get_running_loop()

        await sched._on_timer()
        await asyncio.sleep(0.2)

        # send_markdown called for curator alert (not for delivery)
        mock_im.send_markdown.assert_called()
        call_args = mock_im.send_markdown.call_args
        assert "failed" in call_args.kwargs.get("content", "") or "failed" in str(call_args)

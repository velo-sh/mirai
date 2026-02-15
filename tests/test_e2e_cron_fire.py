"""E2E tests: cron fires → real LLM agent → im_tool delivery chain.

Tests the full user-facing flow from the perspective of "what happens when a
cron job triggers": the scheduler wakes, calls agent.run(), the LLM decides
to use im_tool, and a message is delivered to the (mock) IM provider.

Unlike test_e2e_cron_real_llm.py (which tests cron *tool actions* like
add/remove/list), these tests exercise the *timer-fire* path introduced
in the Phase 1 refactor:

  Smart timer arm → _on_timer → _fire_job → agent.run(prompt) → LLM
                  → im_tool.execute() → im_provider.send_message()

Requirements:
  - MINIMAX_API_KEY must be available (from .env or environment)
  - Tests are auto-skipped if no API key is present

Usage:
  pytest tests/test_e2e_cron_fire.py -v -s
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from dotenv import load_dotenv

# Load .env before importing mirai modules
load_dotenv()

from mirai.agent.agent_loop import AgentLoop  # noqa: E402
from mirai.agent.providers.factory import create_provider  # noqa: E402
from mirai.agent.tools.base import ToolContext  # noqa: E402
from mirai.agent.tools.im import IMTool  # noqa: E402
from mirai.agent.tools.system import SystemTool  # noqa: E402
from mirai.config import MiraiConfig  # noqa: E402
from mirai.cron import CronScheduler, _load_json5, _save_json5_atomic  # noqa: E402

# ---------------------------------------------------------------------------
# Skip if no LLM API key is available
# ---------------------------------------------------------------------------

_HAS_MINIMAX_KEY = bool(os.getenv("MINIMAX_API_KEY"))

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not _HAS_MINIMAX_KEY,
        reason="MINIMAX_API_KEY not set — skipping real LLM E2E tests",
    ),
]

_MODEL = "MiniMax-M2.5"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_job(
    *,
    job_id: str = "e2e:fire-test",
    name: str = "Fire Test",
    prompt: str = "Say hello to the user",
    every_ms: int = 60_000,
    enabled: bool = True,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a minimal cron job dict."""
    return {
        "id": job_id,
        "name": name,
        "enabled": enabled,
        "schedule": {"kind": "every", "everyMs": every_ms},
        "payload": {"kind": "agentTurn", "prompt": prompt},
        "state": state or {},
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cron_dir(tmp_path: Path) -> Path:
    """Create a temporary cron state directory."""
    d = tmp_path / "cron"
    d.mkdir()
    _save_json5_atomic(d / "system.json5", {"version": 1, "jobs": []})
    _save_json5_atomic(d / "jobs.json5", {"version": 1, "jobs": []})
    return d


@pytest.fixture
def mock_im() -> AsyncMock:
    """A mock IM provider that records sent messages."""
    im = AsyncMock()
    im.send_message = AsyncMock(return_value=True)
    im.send_markdown = AsyncMock()
    return im


@pytest.fixture
def provider():
    """Create a real MiniMax provider."""
    return create_provider(provider="minimax", model=_MODEL)


@pytest.fixture
def agent_with_im(provider, mock_im, tmp_path: Path) -> AgentLoop:
    """Create a real AgentLoop with im_tool wired to a mock IM provider.

    The agent has access to im_tool so the LLM can send messages,
    but the IM provider is mocked so no real messages are sent.
    """
    from mirai.db.duck import DuckDBStorage

    config = MiraiConfig.load()
    ctx = ToolContext(
        config=config,
        im_provider=mock_im,
        cron_scheduler=None,
        start_time=0.0,
    )
    im_tool = IMTool(context=ctx)
    system_tool = SystemTool(context=ctx)

    storage = DuckDBStorage(db_path=str(tmp_path / "test_e2e_fire.duckdb"))
    loop = AgentLoop(
        provider=provider,
        tools=[im_tool, system_tool],
        collaborator_id="test-e2e-cron-fire",
        l3_storage=storage,
        base_system_prompt=(
            "You are a cron-triggered agent. When woken by a cron job, "
            "you MUST use im_tool to send the result to the user. "
            "Always use im_tool with action 'send_message'. "
            "Keep messages brief."
        ),
    )
    return loop


# ---------------------------------------------------------------------------
# Tests: Full cron fire → agent → im_tool chain
# ---------------------------------------------------------------------------


class TestCronFireE2E:
    """E2E tests for the complete cron-fire → agent → im_tool chain."""

    async def test_cron_fire_triggers_agent_and_im_delivery(
        self, cron_dir: Path, agent_with_im: AgentLoop, mock_im: AsyncMock
    ):
        """Full chain: cron fires due job → agent.run → LLM → im_tool → mock IM.

        This is the most important E2E test — it proves the Phase 1
        "main session wake" architecture works end-to-end with a real LLM.
        """
        # 1. Set up a job that is already due
        past_ms = int(time.time() * 1000) - 5000
        job = _make_job(
            prompt="You have been woken by a cron job. Greet the user briefly using im_tool.",
            state={"nextRunAtMs": past_ms},
        )
        _save_json5_atomic(cron_dir / "jobs.json5", {"version": 1, "jobs": [job]})

        # 2. Create scheduler wired with the real agent
        sched = CronScheduler(state_dir=cron_dir, agent=agent_with_im)
        sched._load_stores()
        sched._loop = asyncio.get_running_loop()

        # 3. Fire the timer (simulates cron wake)
        await sched._on_timer()

        # 4. Wait for the spawned task to complete (real LLM call takes time)
        for _ in range(60):
            await asyncio.sleep(0.5)
            sched._reap_finished()
            if not sched._running:
                break

        # 5. Verify: agent was called and completed successfully
        state = sched._agent_jobs[0]["state"]
        assert state.get("lastStatus") == "ok", (
            f"Expected 'ok' but got '{state.get('lastStatus')}'. Error: {state.get('lastError')}"
        )
        assert state.get("runCount", 0) >= 1

        # 6. Verify: im_tool was called (agent sent a message)
        assert mock_im.send_message.called, (
            "Expected agent to use im_tool to send a message, but send_message was not called"
        )

    async def test_cron_fire_records_duration(self, cron_dir: Path, agent_with_im: AgentLoop, mock_im: AsyncMock):
        """After cron fires, lastDurationMs reflects actual LLM latency."""
        past_ms = int(time.time() * 1000) - 1000
        job = _make_job(
            prompt="Say 'hello' using im_tool. Be very brief.",
            state={"nextRunAtMs": past_ms},
        )
        _save_json5_atomic(cron_dir / "jobs.json5", {"version": 1, "jobs": [job]})

        sched = CronScheduler(state_dir=cron_dir, agent=agent_with_im)
        sched._load_stores()
        sched._loop = asyncio.get_running_loop()

        await sched._on_timer()

        for _ in range(60):
            await asyncio.sleep(0.5)
            sched._reap_finished()
            if not sched._running:
                break

        state = sched._agent_jobs[0]["state"]
        duration = state.get("lastDurationMs")
        assert duration is not None, "lastDurationMs should be recorded"
        # Real LLM call takes at least a few hundred ms
        assert duration > 100, f"Expected real latency > 100ms, got {duration}ms"
        # But shouldn't be more than 60s for a simple prompt
        assert duration < 60_000, f"Duration unexpectedly large: {duration}ms"

    async def test_cron_fire_persists_state_to_disk(self, cron_dir: Path, agent_with_im: AgentLoop, mock_im: AsyncMock):
        """State changes from cron fire are persisted to jobs.json5 on disk."""
        past_ms = int(time.time() * 1000) - 1000
        job = _make_job(
            prompt="Use im_tool to send 'ping' to the user.",
            state={"nextRunAtMs": past_ms},
        )
        _save_json5_atomic(cron_dir / "jobs.json5", {"version": 1, "jobs": [job]})

        sched = CronScheduler(state_dir=cron_dir, agent=agent_with_im)
        sched._load_stores()
        sched._loop = asyncio.get_running_loop()

        await sched._on_timer()

        for _ in range(60):
            await asyncio.sleep(0.5)
            sched._reap_finished()
            if not sched._running:
                break

        # Read from disk (not in-memory) to verify persistence
        loaded = _load_json5(cron_dir / "jobs.json5")
        disk_state = loaded["jobs"][0]["state"]
        assert disk_state.get("lastStatus") == "ok"
        assert "lastRunAtMs" in disk_state
        assert "lastDurationMs" in disk_state
        assert "nextRunAtMs" in disk_state
        # nextRunAtMs should be in the future
        assert disk_state["nextRunAtMs"] > int(time.time() * 1000)

    async def test_cron_fire_computes_next_run_after_success(
        self, cron_dir: Path, agent_with_im: AgentLoop, mock_im: AsyncMock
    ):
        """After successful fire, nextRunAtMs advances to next interval."""
        past_ms = int(time.time() * 1000) - 1000
        interval_ms = 120_000  # 2 minutes
        job = _make_job(
            prompt="Use im_tool to say 'scheduled message'.",
            every_ms=interval_ms,
            state={"nextRunAtMs": past_ms},
        )
        _save_json5_atomic(cron_dir / "jobs.json5", {"version": 1, "jobs": [job]})

        sched = CronScheduler(state_dir=cron_dir, agent=agent_with_im)
        sched._load_stores()
        sched._loop = asyncio.get_running_loop()

        await sched._on_timer()

        for _ in range(60):
            await asyncio.sleep(0.5)
            sched._reap_finished()
            if not sched._running:
                break

        state = sched._agent_jobs[0]["state"]
        now_ms = int(time.time() * 1000)
        next_run = state["nextRunAtMs"]
        # Next run should be approximately now + interval (within tolerance)
        assert next_run > now_ms, "nextRunAtMs should be in the future"
        assert next_run <= now_ms + interval_ms + 5000, (
            f"nextRunAtMs too far ahead: {next_run - now_ms}ms (expected ~{interval_ms}ms)"
        )


# ---------------------------------------------------------------------------
# Tests: Error handling with real LLM
# ---------------------------------------------------------------------------


class TestCronFireErrorE2E:
    """E2E tests for error handling when cron fires but agent fails."""

    async def test_error_backoff_applied_on_failure(self, cron_dir: Path, tmp_path: Path):
        """When agent.run raises, backoff delay is applied to nextRunAtMs."""
        # Use a mock agent that always fails
        failing_agent = AsyncMock()
        failing_agent.run = AsyncMock(side_effect=RuntimeError("LLM unavailable"))

        past_ms = int(time.time() * 1000) - 1000
        job = _make_job(
            every_ms=1000,  # tiny interval → backoff should dominate
            state={"nextRunAtMs": past_ms},
        )
        _save_json5_atomic(cron_dir / "jobs.json5", {"version": 1, "jobs": [job]})

        sched = CronScheduler(state_dir=cron_dir, agent=failing_agent)
        sched._load_stores()
        sched._loop = asyncio.get_running_loop()

        await sched._on_timer()
        await asyncio.sleep(0.3)
        sched._reap_finished()

        state = sched._agent_jobs[0]["state"]
        assert state["lastStatus"] == "error"
        assert state["consecutiveErrors"] == 1

        # nextRunAtMs should be pushed out by ~30s backoff (1st error)
        now_ms = int(time.time() * 1000)
        assert state["nextRunAtMs"] >= now_ms + 25_000

    async def test_consecutive_fire_failures_escalate_backoff(self, cron_dir: Path, tmp_path: Path):
        """Multiple consecutive errors increase the backoff delay."""
        failing_agent = AsyncMock()
        failing_agent.run = AsyncMock(side_effect=RuntimeError("still broken"))

        past_ms = int(time.time() * 1000) - 1000
        job = _make_job(
            every_ms=1000,
            state={"nextRunAtMs": past_ms, "consecutiveErrors": 1},  # 2nd error coming
        )
        _save_json5_atomic(cron_dir / "jobs.json5", {"version": 1, "jobs": [job]})

        sched = CronScheduler(state_dir=cron_dir, agent=failing_agent)
        sched._load_stores()
        sched._loop = asyncio.get_running_loop()

        await sched._on_timer()
        await asyncio.sleep(0.3)
        sched._reap_finished()

        state = sched._agent_jobs[0]["state"]
        assert state["consecutiveErrors"] == 2
        # 2nd error → 60s backoff
        now_ms = int(time.time() * 1000)
        assert state["nextRunAtMs"] >= now_ms + 55_000

    async def test_recovery_after_errors_resets_backoff(
        self, cron_dir: Path, agent_with_im: AgentLoop, mock_im: AsyncMock
    ):
        """When a job succeeds after errors, consecutiveErrors resets to 0."""
        past_ms = int(time.time() * 1000) - 1000
        job = _make_job(
            prompt="Use im_tool to say 'recovery success'.",
            state={"nextRunAtMs": past_ms, "consecutiveErrors": 3},
        )
        _save_json5_atomic(cron_dir / "jobs.json5", {"version": 1, "jobs": [job]})

        sched = CronScheduler(state_dir=cron_dir, agent=agent_with_im)
        sched._load_stores()
        sched._loop = asyncio.get_running_loop()

        await sched._on_timer()

        for _ in range(60):
            await asyncio.sleep(0.5)
            sched._reap_finished()
            if not sched._running:
                break

        state = sched._agent_jobs[0]["state"]
        assert state["lastStatus"] == "ok"
        assert state["consecutiveErrors"] == 0

        # nextRunAtMs should be normal interval, not pushed by backoff
        now_ms = int(time.time() * 1000)
        assert state["nextRunAtMs"] <= now_ms + 65_000  # within normal 60s schedule


# ---------------------------------------------------------------------------
# Tests: Smart timer lifecycle with real agent
# ---------------------------------------------------------------------------


class TestSmartTimerE2E:
    """E2E tests for smart timer arming and re-arming."""

    async def test_timer_re_arms_after_successful_fire(
        self, cron_dir: Path, agent_with_im: AgentLoop, mock_im: AsyncMock
    ):
        """After a job fires and completes, the timer re-arms for next run."""
        past_ms = int(time.time() * 1000) - 1000
        job = _make_job(
            prompt="Use im_tool to send 'timer test'.",
            state={"nextRunAtMs": past_ms},
        )
        _save_json5_atomic(cron_dir / "jobs.json5", {"version": 1, "jobs": [job]})

        sched = CronScheduler(state_dir=cron_dir, agent=agent_with_im)
        sched._load_stores()
        sched._loop = asyncio.get_running_loop()

        await sched._on_timer()

        for _ in range(60):
            await asyncio.sleep(0.5)
            sched._reap_finished()
            if not sched._running:
                break

        # Timer should be armed for the next run
        assert sched._timer_handle is not None, "Timer handle should be set after successful fire"
        sched._timer_handle.cancel()

    async def test_oneshot_job_deleted_after_fire(self, cron_dir: Path, agent_with_im: AgentLoop, mock_im: AsyncMock):
        """A deleteAfterRun job is removed from the scheduler after execution."""
        past_ms = int(time.time() * 1000) - 1000
        job = _make_job(
            job_id="e2e:oneshot",
            prompt="Use im_tool to say 'one-time message'. Be very brief.",
            state={"nextRunAtMs": past_ms},
        )
        job["deleteAfterRun"] = True
        _save_json5_atomic(cron_dir / "jobs.json5", {"version": 1, "jobs": [job]})

        sched = CronScheduler(state_dir=cron_dir, agent=agent_with_im)
        sched._load_stores()
        sched._loop = asyncio.get_running_loop()

        assert len(sched._agent_jobs) == 1

        await sched._on_timer()

        for _ in range(60):
            await asyncio.sleep(0.5)
            sched._reap_finished()
            if not sched._running:
                break

        # Job should be deleted from memory
        assert len(sched._agent_jobs) == 0, f"Expected 0 jobs after oneshot, got {len(sched._agent_jobs)}"

        # And from disk
        loaded = _load_json5(cron_dir / "jobs.json5")
        assert len(loaded["jobs"]) == 0

        # im_tool should still have been called
        assert mock_im.send_message.called

    async def test_disabled_job_not_fired(self, cron_dir: Path, agent_with_im: AgentLoop, mock_im: AsyncMock):
        """A disabled job is never executed, even if past due."""
        past_ms = int(time.time() * 1000) - 5000
        job = _make_job(
            enabled=False,
            prompt="This should never run!",
            state={"nextRunAtMs": past_ms},
        )
        _save_json5_atomic(cron_dir / "jobs.json5", {"version": 1, "jobs": [job]})

        sched = CronScheduler(state_dir=cron_dir, agent=agent_with_im)
        sched._load_stores()
        sched._loop = asyncio.get_running_loop()

        await sched._on_timer()
        await asyncio.sleep(1.0)

        # Agent should NOT have been called
        mock_im.send_message.assert_not_called()
        state = sched._agent_jobs[0]["state"]
        assert state.get("lastStatus") is None

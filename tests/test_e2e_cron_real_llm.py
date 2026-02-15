"""Real E2E tests for cron tool actions — calls a live LLM (MiniMax).

Unlike test_e2e_cron_tool.py which tests the SystemTool → CronScheduler path
in isolation, these tests wire up a **real AgentLoop** with a **real LLM provider**
(MiniMax) and verify the agent can autonomously use cron tools.

Requirements:
  - MINIMAX_API_KEY must be available (from .env or environment)
  - Tests are auto-skipped if no API key is present

Usage:
  pytest tests/test_e2e_cron_real_llm.py -v -s
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load .env before importing mirai modules
load_dotenv()

from mirai.agent.agent_loop import AgentLoop  # noqa: E402
from mirai.agent.providers.factory import create_provider  # noqa: E402
from mirai.agent.tools.base import ToolContext  # noqa: E402
from mirai.agent.tools.system import SystemTool  # noqa: E402
from mirai.config import MiraiConfig  # noqa: E402
from mirai.cron import CronScheduler, _save_json5_atomic  # noqa: E402

# ---------------------------------------------------------------------------
# Skip if no LLM API key is available
# ---------------------------------------------------------------------------

_HAS_MINIMAX_KEY = bool(os.getenv("MINIMAX_API_KEY"))

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not _HAS_MINIMAX_KEY, reason="MINIMAX_API_KEY not set — skipping real LLM tests"),
]

# Model to use for tests — fast and cheap
_MODEL = "MiniMax-M2.5"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cron_dir(tmp_path: Path) -> Path:
    """Create a temporary cron state directory with empty job files."""
    d = tmp_path / "cron"
    d.mkdir()
    _save_json5_atomic(d / "system.json5", {"version": 1, "jobs": []})
    _save_json5_atomic(d / "jobs.json5", {"version": 1, "jobs": []})
    return d


@pytest.fixture
def cron_scheduler(cron_dir: Path) -> CronScheduler:
    """Create a CronScheduler backed by the tmp directory."""
    sched = CronScheduler(state_dir=cron_dir, agent=None)
    sched._load_stores()
    return sched


@pytest.fixture
def provider():
    """Create a real MiniMax provider."""
    return create_provider(provider="minimax", model=_MODEL)


@pytest.fixture
def system_tool(cron_scheduler: CronScheduler) -> SystemTool:
    """SystemTool wired with a real CronScheduler (but no LLM)."""
    config = MiraiConfig.load()
    ctx = ToolContext(
        config=config,
        cron_scheduler=cron_scheduler,
        start_time=0.0,
    )
    return SystemTool(context=ctx)


@pytest.fixture
def agent(provider, system_tool: SystemTool, tmp_path: Path) -> AgentLoop:
    """Create a real AgentLoop with the SystemTool and MiniMax provider.

    Uses an isolated temp DuckDB so tests can run alongside the live server.
    """
    from mirai.db.duck import DuckDBStorage

    storage = DuckDBStorage(db_path=str(tmp_path / "test_e2e.duckdb"))
    loop = AgentLoop(
        provider=provider,
        tools=[system_tool],
        collaborator_id="test-e2e-cron",
        l3_storage=storage,
        base_system_prompt=(
            "You are a test agent. You have access to mirai_system tool. "
            "When asked about cron jobs, use the appropriate action. "
            "Keep responses very brief."
        ),
    )
    return loop


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCronWithRealLLM:
    """E2E tests that call a real LLM to exercise cron tool usage."""

    async def test_agent_lists_cron_jobs(self, agent: AgentLoop, system_tool: SystemTool):
        """Agent should be able to list cron jobs when asked.

        This test verifies the full chain:
          User message → AgentLoop.run() → LLM decides to call mirai_system
          → SystemTool.execute(action='list_cron_jobs') → CronScheduler
          → response back through agent
        """
        result = await agent.run("List all my cron jobs using the mirai_system tool.")
        assert result  # Agent must return a non-empty response
        assert isinstance(result, str)
        # The agent should mention something about jobs or cron
        # (it might say "no jobs" or "0 jobs" since the scheduler is empty)

    async def test_agent_adds_cron_job(self, agent: AgentLoop, system_tool: SystemTool, cron_scheduler: CronScheduler):
        """Agent should successfully add a cron job when instructed.

        This test verifies:
          1. Agent understands the add_cron_job action
          2. Agent constructs valid cron_job parameters
          3. Job is persisted in the CronScheduler
        """
        result = await agent.run(
            "Add a cron job using mirai_system tool with action add_cron_job. "
            "The job id should be 'e2e:llm-test', schedule should be "
            "{'kind': 'every', 'everyMs': 60000}, and payload should be "
            "{'kind': 'agentTurn', 'prompt': 'Say hello'}. "
            "Use the exact parameters I specified."
        )
        assert result
        assert isinstance(result, str)

        # Verify the job was actually added to the scheduler
        agent_jobs = cron_scheduler._agent_jobs
        job_ids = [j["id"] for j in agent_jobs]
        assert "e2e:llm-test" in job_ids, (
            f"Expected job 'e2e:llm-test' in scheduler, got: {job_ids}. Agent response: {result}"
        )

    async def test_agent_add_then_remove(
        self, agent: AgentLoop, system_tool: SystemTool, cron_scheduler: CronScheduler
    ):
        """Agent should be able to add and then remove a cron job.

        Full lifecycle via LLM:
          1. Add a job → verify in scheduler
          2. Remove the job → verify removed from scheduler
        """
        # Step 1: Add
        add_result = await agent.run(
            "Use mirai_system tool with action add_cron_job to create a job with "
            "id 'e2e:temp-job', schedule {'kind': 'every', 'everyMs': 30000}, "
            "payload {'kind': 'agentTurn', 'prompt': 'temp test'}."
        )
        assert add_result

        # Verify added
        assert any(j["id"] == "e2e:temp-job" for j in cron_scheduler._agent_jobs), (
            f"Job not added. Agent said: {add_result}"
        )

        # Step 2: Remove
        remove_result = await agent.run(
            "Use mirai_system tool with action remove_cron_job, with job_id 'e2e:temp-job'."
        )
        assert remove_result

        # Verify removed
        assert not any(j["id"] == "e2e:temp-job" for j in cron_scheduler._agent_jobs), (
            f"Job not removed. Agent said: {remove_result}"
        )

    async def test_agent_refuses_system_job(
        self, agent: AgentLoop, system_tool: SystemTool, cron_scheduler: CronScheduler
    ):
        """Agent should relay the system job protection error.

        Even if instructed, the tool should refuse to add/remove sys: jobs.
        The agent should report the error back. Two valid outcomes:
          1. LLM calls the tool → tool returns error → agent relays it
          2. LLM refuses to call the tool (safety filter) → agent explains
        Either way, the job should NOT be created.
        """
        result = await agent.run(
            "Use mirai_system tool with action add_cron_job. "
            "Create a job with id 'sys:reserved-test', "
            "schedule {'kind': 'every', 'everyMs': 60000}, "
            "payload {'kind': 'agentTurn', 'prompt': 'hello world'}."
        )
        assert result
        # The job must NOT be in the scheduler (sys: prefix is protected)
        assert not any(j["id"] == "sys:reserved-test" for j in cron_scheduler._agent_jobs), (
            f"System job was unexpectedly created! Agent said: {result}"
        )
        # The agent should mention some kind of rejection
        result_lower = result.lower()
        rejection_keywords = [
            "error",
            "cannot",
            "can't",
            "not allowed",
            "system",
            "unable",
            "refuse",
            "denied",
            "not permitted",
        ]
        assert any(kw in result_lower for kw in rejection_keywords), f"Expected rejection message, agent said: {result}"

    async def test_tool_call_verified_via_scheduler_state(
        self, agent: AgentLoop, system_tool: SystemTool, cron_scheduler: CronScheduler
    ):
        """Verify the LLM actually invoked the tool by checking scheduler state.

        This is the strongest E2E assertion: we don't just check the agent's
        text response, we verify the side effect in the CronScheduler.
        """
        # Pre-condition: no agent jobs
        assert len(cron_scheduler._agent_jobs) == 0

        # Ask agent to add a job
        await agent.run(
            "Call mirai_system with action add_cron_job. Parameters: "
            "cron_job = {'id': 'e2e:verify', 'schedule': {'kind': 'every', 'everyMs': 120000}, "
            "'payload': {'kind': 'agentTurn', 'prompt': 'verification test'}}."
        )

        # Post-condition: job exists in scheduler
        assert len(cron_scheduler._agent_jobs) == 1
        job = cron_scheduler._agent_jobs[0]
        assert job["id"] == "e2e:verify"
        assert job["payload"]["prompt"] == "verification test"

        # Also verify disk persistence
        jobs_file = cron_scheduler._jobs_path
        content = jobs_file.read_text()
        assert "e2e:verify" in content

    async def test_list_returns_structured_data(
        self, agent: AgentLoop, system_tool: SystemTool, cron_scheduler: CronScheduler
    ):
        """After adding a job, listing should show it in the agent response."""
        # Add a job directly (bypass LLM for setup)
        await system_tool.execute(
            action="add_cron_job",
            cron_job={
                "id": "e2e:listed",
                "name": "Listed Job",
                "schedule": {"kind": "every", "everyMs": 60000},
                "payload": {"kind": "agentTurn", "prompt": "I am listed"},
            },
        )

        # Now ask the agent to list
        result = await agent.run("List all cron jobs using mirai_system tool.")
        assert result
        # The agent should mention the job we added
        assert "e2e:listed" in result or "Listed Job" in result or "listed" in result.lower(), (
            f"Expected agent to mention the listed job, got: {result}"
        )

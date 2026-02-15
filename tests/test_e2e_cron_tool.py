"""E2E tests for CronTool — list_cron_jobs, add_cron_job, remove_cron_job.

Tests exercise the full path: CronTool.execute() → _list/_add/_remove → CronScheduler
with a real CronScheduler backed by a temp directory and JSON5 files on disk.
"""

import json
from pathlib import Path

import pytest

from mirai.agent.tools.base import ToolContext
from mirai.agent.tools.cron_tool import CronTool
from mirai.cron import CronScheduler, _save_json5_atomic

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeConfig:
    """Minimal stand-in for MiraiConfig."""

    class _LLM:
        default_model = "test-model"
        max_tokens = 4096

    class _Heartbeat:
        interval = 3600.0
        enabled = True

    llm = _LLM()
    heartbeat = _Heartbeat()


def _make_tool(tmp_path: Path, system_jobs: list | None = None, agent_jobs: list | None = None) -> CronTool:
    """Build a CronTool wired to a real CronScheduler in tmp_path."""
    state_dir = tmp_path / "cron"
    state_dir.mkdir(parents=True, exist_ok=True)

    # Write initial job files
    _save_json5_atomic(state_dir / "system.json5", {"version": 1, "jobs": system_jobs or []})
    _save_json5_atomic(state_dir / "jobs.json5", {"version": 1, "jobs": agent_jobs or []})

    sched = CronScheduler(state_dir=state_dir, agent=None)
    sched._load_stores()

    ctx = ToolContext(
        config=_FakeConfig(),
        cron_scheduler=sched,
        start_time=0.0,
    )
    return CronTool(context=ctx)


# ---------------------------------------------------------------------------
# list_cron_jobs
# ---------------------------------------------------------------------------


class TestListCronJobs:
    """E2E tests for the list_cron_jobs action."""

    @pytest.mark.asyncio
    async def test_list_empty(self, tmp_path: Path):
        """No jobs → empty list with total=0."""
        tool = _make_tool(tmp_path)
        raw = await tool.execute(action="list_cron_jobs")
        data = json.loads(raw)

        assert data["total"] == 0
        assert data["jobs"] == []
        assert data["running"] == []

    @pytest.mark.asyncio
    async def test_list_system_jobs(self, tmp_path: Path):
        """System jobs should appear with type='system'."""
        sys_jobs = [
            {
                "id": "sys:heartbeat",
                "name": "heartbeat",
                "enabled": True,
                "schedule": {"kind": "every", "everyMs": 3600000},
            },
            {
                "id": "sys:cleanup",
                "name": "cleanup",
                "enabled": False,
                "schedule": {"kind": "cron", "expr": "0 2 * * *"},
            },
        ]
        tool = _make_tool(tmp_path, system_jobs=sys_jobs)
        raw = await tool.execute(action="list_cron_jobs")
        data = json.loads(raw)

        assert data["total"] == 2
        ids = [j["id"] for j in data["jobs"]]
        assert "sys:heartbeat" in ids
        assert "sys:cleanup" in ids

        # Check type and enabled
        hb = next(j for j in data["jobs"] if j["id"] == "sys:heartbeat")
        assert hb["type"] == "system"
        assert hb["enabled"] is True

        cl = next(j for j in data["jobs"] if j["id"] == "sys:cleanup")
        assert cl["type"] == "system"
        assert cl["enabled"] is False

    @pytest.mark.asyncio
    async def test_list_agent_jobs(self, tmp_path: Path):
        """Agent jobs should appear with type='agent' and include payload."""
        agent_jobs = [
            {
                "id": "daily-report",
                "name": "Daily Report",
                "enabled": True,
                "schedule": {"kind": "every", "everyMs": 86400000},
                "payload": {"kind": "agentTurn", "prompt": "Generate daily report"},
            },
        ]
        tool = _make_tool(tmp_path, agent_jobs=agent_jobs)
        raw = await tool.execute(action="list_cron_jobs")
        data = json.loads(raw)

        assert data["total"] == 1
        job = data["jobs"][0]
        assert job["id"] == "daily-report"
        assert job["type"] == "agent"
        assert job["payload"]["prompt"] == "Generate daily report"

    @pytest.mark.asyncio
    async def test_list_mixed_system_and_agent(self, tmp_path: Path):
        """Mixed system + agent jobs appear together with correct types."""
        sys_jobs = [
            {"id": "sys:refresh", "name": "refresh", "enabled": True, "schedule": {"kind": "every", "everyMs": 300000}}
        ]
        agent_jobs = [
            {
                "id": "my-task",
                "name": "My Task",
                "enabled": True,
                "schedule": {"kind": "every", "everyMs": 60000},
                "payload": {"kind": "agentTurn", "prompt": "hello"},
            },
        ]
        tool = _make_tool(tmp_path, system_jobs=sys_jobs, agent_jobs=agent_jobs)
        raw = await tool.execute(action="list_cron_jobs")
        data = json.loads(raw)

        assert data["total"] == 2
        types = {j["id"]: j["type"] for j in data["jobs"]}
        assert types["sys:refresh"] == "system"
        assert types["my-task"] == "agent"

    @pytest.mark.asyncio
    async def test_list_without_scheduler(self):
        """Should return error when cron_scheduler is None."""
        ctx = ToolContext(config=_FakeConfig(), cron_scheduler=None, start_time=0.0)
        tool = CronTool(context=ctx)
        result = await tool.execute(action="list_cron_jobs")
        assert "Error" in result
        assert "not available" in result


# ---------------------------------------------------------------------------
# add_cron_job
# ---------------------------------------------------------------------------


class TestAddCronJob:
    """E2E tests for the add_cron_job action."""

    @pytest.mark.asyncio
    async def test_add_basic_job(self, tmp_path: Path):
        """Creating a new job should persist to disk and be listable."""
        tool = _make_tool(tmp_path)

        result = await tool.execute(
            action="add_cron_job",
            cron_job={
                "id": "test:greet",
                "name": "Greeting Job",
                "schedule": {"kind": "every", "everyMs": 60000},
                "payload": {"kind": "agentTurn", "prompt": "Say hello"},
            },
        )
        assert "✅" in result
        assert "created" in result

        # Verify it's listable
        raw = await tool.execute(action="list_cron_jobs")
        data = json.loads(raw)
        assert data["total"] == 1
        assert data["jobs"][0]["id"] == "test:greet"
        assert data["jobs"][0]["name"] == "Greeting Job"

    @pytest.mark.asyncio
    async def test_add_persists_to_disk(self, tmp_path: Path):
        """Added job should be written to the JSON5 file on disk."""
        tool = _make_tool(tmp_path)
        await tool.execute(
            action="add_cron_job",
            cron_job={
                "id": "persist:check",
                "schedule": {"kind": "every", "everyMs": 30000},
                "payload": {"kind": "agentTurn", "prompt": "disk check"},
            },
        )

        # Read the file directly
        jobs_file = tmp_path / "cron" / "jobs.json5"
        assert jobs_file.exists()
        content = jobs_file.read_text()
        assert "persist:check" in content

    @pytest.mark.asyncio
    async def test_add_computes_next_run(self, tmp_path: Path):
        """Adding a job should auto-compute nextRunAtMs in state."""
        tool = _make_tool(tmp_path)
        await tool.execute(
            action="add_cron_job",
            cron_job={
                "id": "next-run:test",
                "schedule": {"kind": "every", "everyMs": 60000},
                "payload": {"kind": "agentTurn", "prompt": "test"},
            },
        )

        raw = await tool.execute(action="list_cron_jobs")
        data = json.loads(raw)
        job = data["jobs"][0]
        assert "nextRunAtMs" in job["state"]
        assert isinstance(job["state"]["nextRunAtMs"], int)

    @pytest.mark.asyncio
    async def test_add_sets_defaults(self, tmp_path: Path):
        """Job without name/enabled should get defaults."""
        tool = _make_tool(tmp_path)
        await tool.execute(
            action="add_cron_job",
            cron_job={
                "id": "defaults:test",
                "schedule": {"kind": "every", "everyMs": 5000},
                "payload": {"kind": "agentTurn", "prompt": "x"},
            },
        )

        raw = await tool.execute(action="list_cron_jobs")
        data = json.loads(raw)
        job = data["jobs"][0]
        assert job["enabled"] is True
        # name defaults to id when not provided
        assert job["name"] == "defaults:test"

    @pytest.mark.asyncio
    async def test_update_existing_job(self, tmp_path: Path):
        """Adding a job with an existing ID should update, not duplicate."""
        agent_jobs = [
            {
                "id": "update:me",
                "name": "Old Name",
                "enabled": True,
                "schedule": {"kind": "every", "everyMs": 60000},
                "payload": {"kind": "agentTurn", "prompt": "old prompt"},
            },
        ]
        tool = _make_tool(tmp_path, agent_jobs=agent_jobs)

        result = await tool.execute(
            action="add_cron_job",
            cron_job={
                "id": "update:me",
                "name": "New Name",
                "schedule": {"kind": "every", "everyMs": 30000},
                "payload": {"kind": "agentTurn", "prompt": "new prompt"},
            },
        )
        assert "updated" in result

        raw = await tool.execute(action="list_cron_jobs")
        data = json.loads(raw)
        assert data["total"] == 1  # no duplicate
        job = data["jobs"][0]
        assert job["name"] == "New Name"
        assert job["payload"]["prompt"] == "new prompt"

    @pytest.mark.asyncio
    async def test_add_system_job_rejected(self, tmp_path: Path):
        """Cannot add a job with sys: prefix."""
        tool = _make_tool(tmp_path)
        result = await tool.execute(
            action="add_cron_job",
            cron_job={
                "id": "sys:hacked",
                "schedule": {"kind": "every", "everyMs": 1000},
                "payload": {"kind": "agentTurn", "prompt": "evil"},
            },
        )
        assert "Error" in result
        assert "system" in result.lower()

    @pytest.mark.asyncio
    async def test_add_missing_id(self, tmp_path: Path):
        """Job without id should be rejected."""
        tool = _make_tool(tmp_path)
        result = await tool.execute(
            action="add_cron_job",
            cron_job={
                "schedule": {"kind": "every", "everyMs": 1000},
                "payload": {"kind": "agentTurn", "prompt": "no id"},
            },
        )
        assert "Error" in result
        assert "id" in result.lower()

    @pytest.mark.asyncio
    async def test_add_missing_schedule(self, tmp_path: Path):
        """Job without schedule should be rejected."""
        tool = _make_tool(tmp_path)
        result = await tool.execute(
            action="add_cron_job",
            cron_job={
                "id": "no-sched",
                "payload": {"kind": "agentTurn", "prompt": "missing schedule"},
            },
        )
        assert "Error" in result
        assert "schedule" in result.lower()

    @pytest.mark.asyncio
    async def test_add_missing_payload(self, tmp_path: Path):
        """Job without payload should be rejected."""
        tool = _make_tool(tmp_path)
        result = await tool.execute(
            action="add_cron_job",
            cron_job={
                "id": "no-payload",
                "schedule": {"kind": "every", "everyMs": 1000},
            },
        )
        assert "Error" in result
        assert "payload" in result.lower()

    @pytest.mark.asyncio
    async def test_add_empty_cron_job(self, tmp_path: Path):
        """Empty cron_job dict should be rejected."""
        tool = _make_tool(tmp_path)
        result = await tool.execute(action="add_cron_job", cron_job={})
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_add_without_scheduler(self):
        """Should return error when cron_scheduler is None."""
        ctx = ToolContext(config=_FakeConfig(), cron_scheduler=None, start_time=0.0)
        tool = CronTool(context=ctx)
        result = await tool.execute(
            action="add_cron_job",
            cron_job={
                "id": "x",
                "schedule": {"kind": "every", "everyMs": 1000},
                "payload": {"kind": "agentTurn", "prompt": "y"},
            },
        )
        assert "Error" in result
        assert "not available" in result

    @pytest.mark.asyncio
    async def test_add_multiple_jobs(self, tmp_path: Path):
        """Adding multiple distinct jobs should accumulate."""
        tool = _make_tool(tmp_path)

        for i in range(3):
            await tool.execute(
                action="add_cron_job",
                cron_job={
                    "id": f"job:{i}",
                    "schedule": {"kind": "every", "everyMs": 60000},
                    "payload": {"kind": "agentTurn", "prompt": f"task {i}"},
                },
            )

        raw = await tool.execute(action="list_cron_jobs")
        data = json.loads(raw)
        assert data["total"] == 3
        ids = sorted(j["id"] for j in data["jobs"])
        assert ids == ["job:0", "job:1", "job:2"]

    @pytest.mark.asyncio
    async def test_add_with_cron_schedule(self, tmp_path: Path):
        """Job with cron expression schedule should be accepted."""
        tool = _make_tool(tmp_path)
        result = await tool.execute(
            action="add_cron_job",
            cron_job={
                "id": "daily:report",
                "schedule": {"kind": "cron", "expr": "0 9 * * 1-5", "tz": "Asia/Shanghai"},
                "payload": {"kind": "agentTurn", "prompt": "Generate report"},
            },
        )
        assert "✅" in result
        assert "created" in result


# ---------------------------------------------------------------------------
# remove_cron_job
# ---------------------------------------------------------------------------


class TestRemoveCronJob:
    """E2E tests for the remove_cron_job action."""

    @pytest.mark.asyncio
    async def test_remove_existing_job(self, tmp_path: Path):
        """Removing an existing agent job should succeed."""
        agent_jobs = [
            {
                "id": "remove:me",
                "name": "To Remove",
                "enabled": True,
                "schedule": {"kind": "every", "everyMs": 60000},
                "payload": {"kind": "agentTurn", "prompt": "delete me"},
            },
        ]
        tool = _make_tool(tmp_path, agent_jobs=agent_jobs)

        # Verify it exists
        raw = await tool.execute(action="list_cron_jobs")
        assert json.loads(raw)["total"] == 1

        # Remove
        result = await tool.execute(action="remove_cron_job", job_id="remove:me")
        assert "✅" in result
        assert "removed" in result

        # Verify removal
        raw = await tool.execute(action="list_cron_jobs")
        assert json.loads(raw)["total"] == 0

    @pytest.mark.asyncio
    async def test_remove_persists_to_disk(self, tmp_path: Path):
        """Removal should be persisted to the JSON5 file."""
        agent_jobs = [
            {
                "id": "disk:remove",
                "schedule": {"kind": "every", "everyMs": 60000},
                "payload": {"kind": "agentTurn", "prompt": "x"},
            },
        ]
        tool = _make_tool(tmp_path, agent_jobs=agent_jobs)
        await tool.execute(action="remove_cron_job", job_id="disk:remove")

        content = (tmp_path / "cron" / "jobs.json5").read_text()
        assert "disk:remove" not in content

    @pytest.mark.asyncio
    async def test_remove_nonexistent(self, tmp_path: Path):
        """Removing a non-existent job should return error."""
        tool = _make_tool(tmp_path)
        result = await tool.execute(action="remove_cron_job", job_id="ghost:job")
        assert "Error" in result
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_remove_system_job_rejected(self, tmp_path: Path):
        """Cannot remove system jobs."""
        sys_jobs = [
            {
                "id": "sys:important",
                "name": "important",
                "enabled": True,
                "schedule": {"kind": "every", "everyMs": 3600000},
            },
        ]
        tool = _make_tool(tmp_path, system_jobs=sys_jobs)
        result = await tool.execute(action="remove_cron_job", job_id="sys:important")
        assert "Error" in result
        assert "system" in result.lower()

        # Verify it was NOT removed
        raw = await tool.execute(action="list_cron_jobs")
        assert json.loads(raw)["total"] == 1

    @pytest.mark.asyncio
    async def test_remove_without_job_id(self, tmp_path: Path):
        """Missing job_id should return error."""
        tool = _make_tool(tmp_path)
        result = await tool.execute(action="remove_cron_job")
        assert "Error" in result
        assert "required" in result.lower()

    @pytest.mark.asyncio
    async def test_remove_without_scheduler(self):
        """Should return error when cron_scheduler is None."""
        ctx = ToolContext(config=_FakeConfig(), cron_scheduler=None, start_time=0.0)
        tool = CronTool(context=ctx)
        result = await tool.execute(action="remove_cron_job", job_id="x")
        assert "Error" in result
        assert "not available" in result

    @pytest.mark.asyncio
    async def test_remove_only_target_others_remain(self, tmp_path: Path):
        """Removing one job should not affect others."""
        agent_jobs = [
            {
                "id": "keep:me",
                "schedule": {"kind": "every", "everyMs": 60000},
                "payload": {"kind": "agentTurn", "prompt": "keeper"},
            },
            {
                "id": "remove:me",
                "schedule": {"kind": "every", "everyMs": 60000},
                "payload": {"kind": "agentTurn", "prompt": "goner"},
            },
        ]
        tool = _make_tool(tmp_path, agent_jobs=agent_jobs)
        await tool.execute(action="remove_cron_job", job_id="remove:me")

        raw = await tool.execute(action="list_cron_jobs")
        data = json.loads(raw)
        assert data["total"] == 1
        assert data["jobs"][0]["id"] == "keep:me"


# ---------------------------------------------------------------------------
# Full workflow E2E
# ---------------------------------------------------------------------------


class TestCronE2EWorkflow:
    """Full add → list → update → remove workflow."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, tmp_path: Path):
        """Test the complete lifecycle: add → list → update → list → remove → list."""
        sys_jobs = [
            {"id": "sys:bg", "name": "background", "enabled": True, "schedule": {"kind": "every", "everyMs": 600000}}
        ]
        tool = _make_tool(tmp_path, system_jobs=sys_jobs)

        # 1. List: only system job
        raw = await tool.execute(action="list_cron_jobs")
        data = json.loads(raw)
        assert data["total"] == 1
        assert data["jobs"][0]["type"] == "system"

        # 2. Add an agent job
        result = await tool.execute(
            action="add_cron_job",
            cron_job={
                "id": "e2e:task",
                "name": "E2E Task",
                "schedule": {"kind": "every", "everyMs": 30000},
                "payload": {"kind": "agentTurn", "prompt": "E2E test prompt"},
            },
        )
        assert "created" in result

        # 3. List: system + agent
        raw = await tool.execute(action="list_cron_jobs")
        data = json.loads(raw)
        assert data["total"] == 2
        types = {j["id"]: j["type"] for j in data["jobs"]}
        assert types == {"sys:bg": "system", "e2e:task": "agent"}

        # 4. Update the agent job
        result = await tool.execute(
            action="add_cron_job",
            cron_job={
                "id": "e2e:task",
                "name": "E2E Task Updated",
                "schedule": {"kind": "every", "everyMs": 15000},
                "payload": {"kind": "agentTurn", "prompt": "Updated prompt"},
            },
        )
        assert "updated" in result

        # 5. List: verify update (no duplicates)
        raw = await tool.execute(action="list_cron_jobs")
        data = json.loads(raw)
        assert data["total"] == 2
        agent_job = next(j for j in data["jobs"] if j["id"] == "e2e:task")
        assert agent_job["name"] == "E2E Task Updated"
        assert agent_job["payload"]["prompt"] == "Updated prompt"

        # 6. Remove the agent job
        result = await tool.execute(action="remove_cron_job", job_id="e2e:task")
        assert "removed" in result

        # 7. List: only system job remains
        raw = await tool.execute(action="list_cron_jobs")
        data = json.loads(raw)
        assert data["total"] == 1
        assert data["jobs"][0]["id"] == "sys:bg"

    @pytest.mark.asyncio
    async def test_cannot_modify_system_via_add_or_remove(self, tmp_path: Path):
        """System jobs should be protected against both add and remove."""
        sys_jobs = [
            {
                "id": "sys:protected",
                "name": "protected",
                "enabled": True,
                "schedule": {"kind": "every", "everyMs": 60000},
            }
        ]
        tool = _make_tool(tmp_path, system_jobs=sys_jobs)

        # Try to overwrite system job via add
        result = await tool.execute(
            action="add_cron_job",
            cron_job={
                "id": "sys:protected",
                "schedule": {"kind": "every", "everyMs": 1000},
                "payload": {"kind": "agentTurn", "prompt": "overwrite attempt"},
            },
        )
        assert "Error" in result

        # Try to remove
        result = await tool.execute(action="remove_cron_job", job_id="sys:protected")
        assert "Error" in result

        # System job still intact
        raw = await tool.execute(action="list_cron_jobs")
        data = json.loads(raw)
        assert data["total"] == 1
        assert data["jobs"][0]["id"] == "sys:protected"

    @pytest.mark.asyncio
    async def test_definition_includes_cron_actions(self):
        """Tool definition should advertise all cron actions."""
        tool = CronTool()
        defn = tool.definition
        actions = defn["input_schema"]["properties"]["action"]["enum"]
        assert "list_cron_jobs" in actions
        assert "add_cron_job" in actions
        assert "remove_cron_job" in actions

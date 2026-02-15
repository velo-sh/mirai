"""Cron Tool — manage scheduled jobs via the CronScheduler.

Actions:
  - list_cron_jobs:   Show all scheduled jobs (system + agent).
  - add_cron_job:     Create or update an agent cron job.
  - remove_cron_job:  Delete an agent cron job by ID.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import orjson

from mirai.agent.tools.base import BaseTool
from mirai.logging import get_logger

if TYPE_CHECKING:
    from mirai.agent.tools.base import ToolContext

log = get_logger("mirai.tools.cron")


class CronTool(BaseTool):
    """Manage cron-scheduled jobs."""

    def __init__(self, context: ToolContext | None = None, **kwargs: Any) -> None:
        super().__init__(context)

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "name": "mirai_cron",
            "description": (
                "Manage scheduled jobs. "
                "Actions: 'list_cron_jobs' (show all scheduled jobs), "
                "'add_cron_job' (create or update a scheduled job), "
                "'remove_cron_job' (delete a scheduled job by ID)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list_cron_jobs", "add_cron_job", "remove_cron_job"],
                        "description": "The action to perform.",
                    },
                    "job_id": {
                        "type": "string",
                        "description": (
                            "For 'remove_cron_job' only. The ID of the job to remove. "
                            "Use 'list_cron_jobs' first to see existing jobs."
                        ),
                    },
                    "cron_job": {
                        "type": "object",
                        "description": (
                            "For 'add_cron_job' only. A job definition object. Example: "
                            '{"id": "agent:daily-report", "name": "daily-report", '
                            '"enabled": true, '
                            '"schedule": {"kind": "every", "everyMs": 60000}, '
                            '"payload": {"kind": "agentTurn", "prompt": "Do X"}}. '
                            "Schedule kinds: 'cron' (with 'expr' and optional 'tz'), "
                            "'every' (with 'everyMs' in milliseconds), "
                            "'at' (with ISO datetime for one-shot)."
                        ),
                    },
                },
                "required": ["action"],
            },
        }

    async def execute(  # type: ignore[override]
        self,
        action: str,
        job_id: str | None = None,
        cron_job: dict[str, Any] | None = None,
    ) -> str:
        if action == "list_cron_jobs":
            return await self._list_cron_jobs()
        if action == "add_cron_job":
            return await self._add_cron_job(cron_job or {})
        if action == "remove_cron_job":
            return await self._remove_cron_job(job_id)
        return f"Error: Unknown action '{action}'. Valid actions: list_cron_jobs, add_cron_job, remove_cron_job."

    # ------------------------------------------------------------------
    # Action: list_cron_jobs
    # ------------------------------------------------------------------
    async def _list_cron_jobs(self) -> str:
        """List all scheduled cron jobs (system + agent)."""
        cron = self.context.cron_scheduler
        if not cron:
            return "Error: Cron scheduler not available."

        jobs: list[dict[str, Any]] = []
        for job in cron._system_jobs:
            jobs.append(
                {
                    "id": job["id"],
                    "name": job.get("name"),
                    "type": "system",
                    "enabled": job.get("enabled", True),
                    "schedule": job.get("schedule"),
                    "state": job.get("state", {}),
                }
            )
        for job in cron._agent_jobs:
            jobs.append(
                {
                    "id": job["id"],
                    "name": job.get("name"),
                    "type": "agent",
                    "enabled": job.get("enabled", True),
                    "schedule": job.get("schedule"),
                    "payload": job.get("payload"),
                    "state": job.get("state", {}),
                }
            )

        result: dict[str, Any] = {
            "total": len(jobs),
            "running": list(cron._running.keys()),
            "jobs": jobs,
        }
        return orjson.dumps(result, option=orjson.OPT_INDENT_2).decode()

    # ------------------------------------------------------------------
    # Action: add_cron_job
    # ------------------------------------------------------------------
    async def _add_cron_job(self, cron_job: dict[str, Any]) -> str:
        """Add or update an agent cron job."""
        cron = self.context.cron_scheduler
        if not cron:
            return "Error: Cron scheduler not available."

        if not cron_job:
            return "Error: 'cron_job' parameter is required."

        job_id = cron_job.get("id")
        if not job_id:
            return "Error: Job must have an 'id' field."

        # Guard: cannot modify system jobs
        if str(job_id).startswith("sys:"):
            return "Error: Cannot add or modify system jobs. Only agent jobs (non-sys: prefix) are allowed."

        # Ensure required fields
        if "schedule" not in cron_job:
            return "Error: Job must have a 'schedule' field."
        if "payload" not in cron_job:
            return "Error: Job must have a 'payload' field with a 'prompt'."

        # Set defaults
        cron_job.setdefault("name", job_id)
        cron_job.setdefault("enabled", True)
        cron_job.setdefault("state", {})

        # Compute initial nextRunAtMs
        from mirai.cron import _now_ms, compute_next_run

        if "nextRunAtMs" not in cron_job["state"]:
            next_run = compute_next_run(cron_job["schedule"], after_ms=_now_ms())
            if next_run:
                cron_job["state"]["nextRunAtMs"] = next_run

        # Update existing or append
        updated = False
        for i, existing in enumerate(cron._agent_jobs):
            if existing["id"] == job_id:
                cron._agent_jobs[i] = cron_job
                updated = True
                break
        if not updated:
            cron._agent_jobs.append(cron_job)

        cron._save_agent_jobs()
        action_label = "updated" if updated else "created"
        log.info(f"cron_job_{action_label}_by_agent", job_id=job_id)
        return f"✅ Cron job `{job_id}` {action_label} successfully."

    # ------------------------------------------------------------------
    # Action: remove_cron_job
    # ------------------------------------------------------------------
    async def _remove_cron_job(self, job_id: str | None) -> str:
        """Remove an agent cron job by ID."""
        cron = self.context.cron_scheduler
        if not cron:
            return "Error: Cron scheduler not available."

        if not job_id:
            return "Error: 'job_id' parameter is required."

        # Guard: cannot remove system jobs
        if job_id.startswith("sys:"):
            return "Error: Cannot remove system jobs."

        original_count = len(cron._agent_jobs)
        cron._agent_jobs = [j for j in cron._agent_jobs if j["id"] != job_id]

        if len(cron._agent_jobs) == original_count:
            return f"Error: Job `{job_id}` not found."

        cron._save_agent_jobs()
        log.info("cron_job_removed_by_agent", job_id=job_id)
        return f"✅ Cron job `{job_id}` removed."

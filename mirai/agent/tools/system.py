"""System Tool — runtime introspection and lifecycle management.

Actions:
  - status:        Runtime health check (PID, uptime, memory, model, tools).
  - usage:         Per-model quota usage from the Antigravity API.
  - patch_runtime: Dynamically adjust temperature/max_tokens without restart.
  - restart:       Graceful self-restart via ``os.execv()``.
"""

from __future__ import annotations

import asyncio
import os
import resource
import sys
import time
from typing import TYPE_CHECKING, Any

import orjson

from mirai.agent.tools.base import BaseTool
from mirai.auth.antigravity_auth import fetch_usage
from mirai.logging import get_logger

if TYPE_CHECKING:
    from mirai.agent.agent_loop import AgentLoop
    from mirai.agent.providers.base import ProviderProtocol
    from mirai.agent.registry import ModelRegistry
    from mirai.agent.tools.base import ToolContext
    from mirai.config import MiraiConfig
    from mirai.db.duck import DuckDBStorage

log = get_logger("mirai.tools.system")


class SystemTool(BaseTool):
    """Runtime introspection and lifecycle management."""

    def __init__(
        self,
        context: ToolContext | None = None,
        config: MiraiConfig | None = None,
        registry: ModelRegistry | None = None,
        agent_loop: AgentLoop | None = None,
        provider: ProviderProtocol | None = None,
        storage: DuckDBStorage | None = None,
        start_time: float | None = None,
        **kwargs: Any,
    ) -> None:
        if context is None and any(
            v is not None for v in (config, registry, agent_loop, provider, storage, start_time)
        ):
            # Legacy support for tests that pass components individually
            from mirai.agent.tools.base import ToolContext as _TC

            context = _TC(
                config=config,
                storage=storage or kwargs.get("storage"),
                agent_loop=agent_loop or kwargs.get("agent") or kwargs.get("agent_loop"),
                registry=registry or kwargs.get("registry"),
                provider=provider or kwargs.get("provider"),
                start_time=start_time if start_time is not None else time.time(),
            )
        super().__init__(context)
        # Convenience aliases
        self._config = self.context.config
        self._start_time = self.context.start_time
        self._provider = self.context.provider
        self._registry = self.context.registry
        self._agent_loop = self.context.agent_loop

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "name": "mirai_system",
            "description": (
                "Introspect and control the Mirai runtime. "
                "Actions: 'status' (read-only health check), "
                "'usage' (per-model quota usage and reset times), "
                "'patch_runtime' (adjust temperature/max_tokens without restart), "
                "'restart' (graceful self-restart)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["status", "usage", "patch_runtime", "restart"],
                        "description": "The action to perform.",
                    },
                    "patch": {
                        "type": "object",
                        "description": "Key-value pairs to override (only for patch_runtime).",
                    },
                },
                "required": ["action"],
            },
        }

    async def execute(  # type: ignore[override]
        self,
        action: str,
        **kwargs: Any,
    ) -> str:
        if action == "status":
            return await self._status()
        if action == "usage":
            return await self._usage()
        if action == "patch_runtime":
            return await self._patch_runtime(kwargs.get("patch", {}))
        if action == "restart":
            return await self._restart()
        return f"Error: Unknown action '{action}'. Valid actions: status, usage, patch_runtime, restart."

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    async def _get_available_models(self) -> list[str]:
        """Query the provider's QuotaManager for actually available models."""
        qm = getattr(self._provider, "quota_manager", None)
        if qm is not None:
            await qm._maybe_refresh()
            return sorted(qm._quotas.keys())
        return []

    # ------------------------------------------------------------------
    # Action: status
    # ------------------------------------------------------------------
    async def _status(self) -> str:
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        memory_mb = round(rusage.ru_maxrss / (1024 * 1024), 1)  # macOS: bytes → MB

        model_name = None
        provider_name = None
        if self._config:
            model_name = getattr(self._config.llm, "default_model", None)
            provider_name = "MiraiConfig"

        available = await self._get_available_models()
        info = {
            "pid": os.getpid(),
            "uptime_s": round(time.monotonic() - self._start_time, 1),
            "memory_mb": memory_mb,
            "python": sys.version.split()[0],
            "model": model_name,
            "provider": provider_name,
            "available_models": available,
        }
        return orjson.dumps(info, option=orjson.OPT_INDENT_2).decode()

    # ------------------------------------------------------------------
    # Action: usage
    # ------------------------------------------------------------------
    async def _usage(self) -> str:
        """Return per-model quota usage from the Antigravity API."""
        from mirai.agent.providers.antigravity import AntigravityProvider

        if not isinstance(self._provider, AntigravityProvider) or not self._provider.credentials:
            return "Error: Provider not available for usage query."

        try:
            # Ensure token is fresh
            await self._provider._ensure_fresh_token()

            token = self._provider.credentials.get("access", "")
            project_id = self._provider.credentials.get("project_id", "")

            if not token:
                return "Error: No access token available."

            data = await fetch_usage(token, project_id)
            log.info("usage_fetched", models_count=len(data.get("models", [])))
        except Exception as exc:
            log.error("usage_fetch_error", error=str(exc))
            return f"Error fetching usage: {exc}"

        # Format a concise report
        models = data.get("models", [])
        report: dict[str, Any] = {
            "plan": data.get("plan"),
            "project": data.get("project"),
            "models": [],
        }
        for m in models:
            status = "🔴 exhausted" if m["used_pct"] >= 100 else ("🟡 high" if m["used_pct"] >= 80 else "🟢 ok")
            entry: dict[str, Any] = {
                "model": m["id"],
                "used_pct": round(m["used_pct"], 1),
                "status": status,
            }
            if m.get("reset_time"):
                entry["reset_time"] = m["reset_time"]
            report["models"].append(entry)

        return orjson.dumps(report, option=orjson.OPT_INDENT_2).decode()

    # ------------------------------------------------------------------
    # Action: patch_runtime
    # ------------------------------------------------------------------

    _ALLOWED_RUNTIME_KEYS = {"temperature", "max_tokens", "top_p", "top_k"}

    async def _patch_runtime(self, patch: dict[str, Any]) -> str:
        """Apply validated runtime overrides without restart."""
        if not self._agent_loop:
            return "Error: No agent loop available."
        if not patch:
            return "Error: No patch data provided."

        applied: dict[str, Any] = {}
        ignored: list[str] = []

        for key, value in patch.items():
            if key in self._ALLOWED_RUNTIME_KEYS:
                self._agent_loop.runtime_overrides[key] = value
                applied[key] = value
            else:
                ignored.append(key)

        parts: list[str] = []
        if applied:
            parts.append(f"✅ Successfully applied: {applied}")
        if ignored:
            parts.append(f"⚠️ Ignored unknown keys: {ignored}")
        if not parts:
            return "Error: No valid keys in patch."
        return " | ".join(parts)

    # ------------------------------------------------------------------
    # Action: restart
    # ------------------------------------------------------------------
    async def _restart(self) -> str:
        log.warning("restart_requested", pid=os.getpid())

        async def _do_restart() -> None:
            """Wait for reply to be sent, then cleanly restart."""
            import signal
            import subprocess

            await asyncio.sleep(8.0)

            # Save restart event to trace history before closing storage
            if self._agent_loop:
                try:
                    await self._agent_loop._archive_trace(
                        "restart initiated by system tool", "system", {"action": "restart"}
                    )
                except Exception:
                    pass  # storage may already be closing

            # Close DuckDB before spawning replacement to release file locks
            if self._agent_loop and self._agent_loop.l3_storage:
                try:
                    self._agent_loop.l3_storage.close()
                    log.info("duckdb_closed_for_restart")
                except Exception as exc:
                    log.warning("duckdb_close_error", error=str(exc))

            log.info("restart_executing", argv=sys.argv)

            # Spawn a new independent process (new session = new process group)
            # so it survives when we kill our own process group.
            subprocess.Popen(
                [sys.executable] + sys.argv,
                start_new_session=True,
                cwd=os.getcwd(),
            )

            # Kill our entire process group (main + granian workers).
            # This is the correct UNIX pattern: all children in the group
            # get the signal, releasing DuckDB file locks, etc.
            try:
                os.killpg(os.getpgid(0), signal.SIGTERM)
            except ProcessLookupError:
                pass
            # Give SIGTERM time to clean up (3s instead of 1s)
            await asyncio.sleep(3.0)
            try:
                os.killpg(os.getpgid(0), signal.SIGKILL)
            except ProcessLookupError:
                pass

        # Fire-and-forget: the restart will happen after we return
        asyncio.get_running_loop().create_task(_do_restart())

        return (
            "🔄 Restart scheduled. I'll send this reply first, then restart in ~8 seconds.\n"
            "You will lose this conversation context. I'll be back shortly!"
        )

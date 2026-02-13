"""
System Tool — self-evolution capabilities for the agent.

Provides three actions:
  - status:       Runtime introspection (PID, uptime, memory, model, tools).
  - patch_config: Merge a JSON patch into ~/.mirai/config.toml (whitelisted keys only).
  - restart:      Graceful self-restart via os.execv.
"""

from __future__ import annotations

import asyncio
import os
import resource
import sys
import time

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]  # Python <3.11 backport
from pathlib import Path
from typing import TYPE_CHECKING, Any

import orjson

from mirai.agent.providers.base import ProviderProtocol
from mirai.agent.tools.base import BaseTool
from mirai.logging import get_logger

if TYPE_CHECKING:
    from mirai.agent.loop import AgentLoop
    from mirai.agent.registry import ModelRegistry

log = get_logger("mirai.tools.system")

# Keys that the agent is allowed to modify via patch_config.
_MUTABLE_KEYS: set[tuple[str, str]] = {
    ("heartbeat", "interval"),
    ("heartbeat", "enabled"),
    ("llm", "default_model"),
    ("llm", "max_tokens"),
    ("server", "log_level"),
    ("registry", "refresh_interval"),
}

_CONFIG_PATH = Path.home() / ".mirai" / "config.toml"


def _serialize_toml(data: dict[str, Any]) -> str:
    """Minimal TOML serializer for flat section→key=value config files."""
    lines: list[str] = []
    for section, values in data.items():
        if not isinstance(values, dict):
            continue
        lines.append(f"[{section}]")
        for key, val in values.items():
            if isinstance(val, bool):
                lines.append(f"{key} = {'true' if val else 'false'}")
            elif isinstance(val, (int, float)):
                lines.append(f"{key} = {val}")
            elif isinstance(val, str):
                lines.append(f'{key} = "{val}"')
            else:
                # Fallback: quote as string
                lines.append(f'{key} = "{val}"')
        lines.append("")
    return "\n".join(lines) + "\n"


class SystemTool(BaseTool):
    """Self-evolution tool: status, patch_config, restart."""

    def __init__(
        self,
        config: Any | None = None,
        start_time: float | None = None,
        provider: ProviderProtocol | None = None,
        registry: ModelRegistry | None = None,
        agent_loop: AgentLoop | None = None,
    ):
        self._config = config
        self._start_time = start_time or time.monotonic()
        self._provider = provider
        self._registry = registry
        self._agent_loop = agent_loop

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "name": "mirai_system",
            "description": (
                "Introspect, configure, and control the Mirai runtime. "
                "Actions: 'status' (read-only health check), "
                "'usage' (per-model quota usage and reset times), "
                "'list_models' (discover all available models across providers), "
                "'set_active_model' (switch to a different model at runtime), "
                "'patch_config' (modify whitelisted config keys), "
                "'restart' (graceful self-restart)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["status", "usage", "list_models", "set_active_model", "patch_config", "restart"],
                        "description": "The action to perform.",
                    },
                    "model": {
                        "type": "string",
                        "description": (
                            "For 'set_active_model' only. The model ID to switch to "
                            "(e.g. 'claude-sonnet-4-20250514', 'MiniMax-M2.5'). "
                            "Use 'list_models' first to see available options."
                        ),
                    },
                    "patch": {
                        "type": "object",
                        "description": (
                            "For 'patch_config' only. JSON object with section→key→value "
                            'to merge. Example: {"heartbeat": {"interval": 1800}}. '
                            "Only whitelisted keys are accepted: "
                            "heartbeat.interval, heartbeat.enabled, "
                            "llm.default_model, llm.max_tokens, server.log_level."
                        ),
                    },
                },
                "required": ["action"],
            },
        }

    async def execute(self, action: str, patch: dict[str, Any] | None = None, model: str | None = None) -> str:  # type: ignore[override]
        if action == "status":
            return await self._status()
        elif action == "usage":
            return await self._usage()
        elif action == "list_models":
            return await self._list_models()
        elif action == "set_active_model":
            return await self._set_active_model(model)
        elif action == "patch_config":
            return await self._patch_config(patch or {})
        elif action == "restart":
            return await self._restart()
        else:
            return (
                f"Error: Unknown action '{action}'. "
                "Use 'status', 'usage', 'list_models', 'set_active_model', 'patch_config', or 'restart'."
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    async def _get_available_models(self) -> list[str]:
        """Query the provider's QuotaManager for actually available models."""
        if self._provider and hasattr(self._provider, "quota_manager"):
            qm = self._provider.quota_manager
            await qm._maybe_refresh()
            return sorted(qm._quotas.keys())
        return []

    # ------------------------------------------------------------------
    # Action: list_models
    # ------------------------------------------------------------------
    async def _list_models(self) -> str:
        """Return all available models across all configured providers."""
        if not self._registry:
            return "Error: Model registry not available."
        # Fetch quota data if provider has a QuotaManager
        quota_data: dict[str, float] | None = None
        if self._provider and hasattr(self._provider, "quota_manager"):
            qm = self._provider.quota_manager
            try:
                await qm._maybe_refresh()
            except Exception:
                log.warning("quota_refresh_failed_in_list_models")
            quota_data = dict(qm._quotas)  # use stale data on refresh failure
        return str(self._registry.get_catalog_text(quota_data=quota_data))

    # ------------------------------------------------------------------
    # Action: set_active_model
    # ------------------------------------------------------------------
    async def _set_active_model(self, model: str | None) -> str:
        """Switch to a different model at runtime via provider hot-swap."""
        if not model:
            return "Error: 'model' parameter is required. Use 'list_models' to see available options."

        if not self._registry:
            return "Error: Model registry not available."

        # 1. Validate model exists in registry
        provider_name = self._registry.find_provider_for_model(model)
        if not provider_name:
            return (
                f"Error: Model '{model}' not found in any available provider. "
                "Use action='list_models' to see available options."
            )

        # 2. Check if already active
        if model == self._registry.active_model and provider_name == self._registry.active_provider:
            return f"Already using model '{model}' on provider '{provider_name}'."

        # 3. Create new provider via factory
        try:
            from mirai.agent.providers.factory import create_provider

            new_provider = create_provider(provider=provider_name, model=model)
        except Exception as exc:
            log.error("model_switch_failed", model=model, provider=provider_name, error=str(exc))
            return f"Error: Failed to create provider for '{model}': {exc}"

        # 4. Hot-swap on AgentLoop
        if not self._agent_loop:
            return "Error: Agent loop not available for provider swap."

        self._agent_loop.swap_provider(new_provider)
        self._provider = new_provider  # update our own reference too

        # 5. Persist to registry
        await self._registry.set_active(provider_name, model)

        log.info("model_switched", model=model, provider=provider_name)
        return (
            f"✅ Switched to **{model}** (provider: {provider_name}).\n"
            "This takes effect starting with my next response."
        )

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
            "config_path": str(_CONFIG_PATH),
            "config_exists": _CONFIG_PATH.exists(),
            "heartbeat_interval": getattr(self._config.heartbeat, "interval", None) if self._config else None,
            "heartbeat_enabled": getattr(self._config.heartbeat, "enabled", None) if self._config else None,
        }
        return orjson.dumps(info, option=orjson.OPT_INDENT_2).decode()

    # ------------------------------------------------------------------
    # Action: usage
    # ------------------------------------------------------------------
    async def _usage(self) -> str:
        """Return per-model quota usage from the Antigravity API."""
        if not self._provider or not hasattr(self._provider, "credentials"):
            return "Error: Provider not available for usage query."

        try:
            # Ensure token is fresh
            if hasattr(self._provider, "_ensure_fresh_token"):
                await self._provider._ensure_fresh_token()

            token = self._provider.credentials.get("access", "")
            project_id = self._provider.credentials.get("project_id", "")

            if not token:
                return "Error: No access token available."

            from mirai.auth.antigravity_auth import fetch_usage

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
    # Action: patch_config
    # ------------------------------------------------------------------
    async def _patch_config(self, patch: dict[str, Any]) -> str:
        if not patch:
            return "Error: 'patch' parameter is required and must be a non-empty object."

        # Validate model name if being changed
        new_model = patch.get("llm", {}).get("default_model")
        if new_model:
            available = await self._get_available_models()
            if available and new_model not in available:
                return f"Error: '{new_model}' is not a valid model. Available models: {', '.join(available)}"

        # Validate all keys against the whitelist
        rejected: list[str] = []
        accepted: list[str] = []
        for section, values in patch.items():
            if not isinstance(values, dict):
                rejected.append(f"{section} (must be a section object)")
                continue
            for key in values:
                if (section, key) in _MUTABLE_KEYS:
                    accepted.append(f"{section}.{key}")
                else:
                    rejected.append(f"{section}.{key}")

        if rejected:
            return (
                f"Error: The following keys are not writable: {', '.join(rejected)}. "
                f"Allowed keys: {', '.join(f'{s}.{k}' for s, k in sorted(_MUTABLE_KEYS))}."
            )

        # Read existing config (or start fresh)
        existing: dict[str, Any] = {}
        if _CONFIG_PATH.exists():
            with open(_CONFIG_PATH, "rb") as f:
                existing = tomllib.load(f)

        # Deep merge patch into existing
        for section, values in patch.items():
            if section not in existing:
                existing[section] = {}
            existing[section].update(values)

        # Write back
        _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(_serialize_toml(existing))

        log.info("config_patched", keys=accepted, path=str(_CONFIG_PATH))
        return (
            f"✅ Config patched successfully: {', '.join(accepted)}.\n"
            f"File: {_CONFIG_PATH}\n"
            "Note: Some changes require a restart to take effect. "
            "Use action='restart' if needed."
        )

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
            if self._agent_loop and hasattr(self._agent_loop, "l3_storage"):
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

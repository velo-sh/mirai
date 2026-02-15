"""Config Tool — modify configuration on disk or volatile runtime settings.

Actions:
  - patch_config:  Merge a JSON patch into ~/.mirai/config.toml (whitelisted keys only).
  - patch_runtime: Apply volatile hyperparameter overrides (temperature, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]  # Python <3.11 backport

from mirai.agent.tools.base import BaseTool
from mirai.logging import get_logger

if TYPE_CHECKING:
    from mirai.agent.tools.base import ToolContext

log = get_logger("mirai.tools.config")

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


class ConfigTool(BaseTool):
    """Modify Mirai configuration on disk or volatile runtime overrides."""

    def __init__(self, context: ToolContext | None = None, **kwargs: Any) -> None:
        super().__init__(context)
        self._config = self.context.config
        self._agent_loop = self.context.agent_loop
        self._provider = self.context.provider

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "name": "mirai_config",
            "description": (
                "Modify Mirai configuration. "
                "Actions: 'patch_config' (modify whitelisted config keys on disk), "
                "'patch_runtime' (volatile hyperparameter overrides like temperature)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["patch_config", "patch_runtime"],
                        "description": "The action to perform.",
                    },
                    "patch": {
                        "type": "object",
                        "description": (
                            "JSON object with configuration to apply. "
                            "For 'patch_config': section→key→value to merge. "
                            'Example: {"heartbeat": {"interval": 1800}}. '
                            "Only whitelisted keys are accepted: "
                            "heartbeat.interval, heartbeat.enabled, "
                            "llm.default_model, llm.max_tokens, server.log_level. "
                            "For 'patch_runtime': key→value overrides. "
                            "Allowed: temperature, max_tokens, top_p, top_k, presence_penalty."
                        ),
                    },
                },
                "required": ["action"],
            },
        }

    async def execute(  # type: ignore[override]
        self,
        action: str,
        patch: dict[str, Any] | None = None,
    ) -> str:
        if action == "patch_config":
            return await self._patch_config(patch or {})
        if action == "patch_runtime":
            return await self._patch_runtime(patch or {})
        return f"Error: Unknown action '{action}'. Valid actions: patch_config, patch_runtime."

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
    # Action: patch_runtime
    # ------------------------------------------------------------------
    async def _patch_runtime(self, patch: dict[str, Any]) -> str:
        """Apply volatile hyperparameter overrides to the active AgentLoop."""
        if not patch:
            return "Error: 'patch' parameter is required for patch_runtime."
        if not self._agent_loop:
            return "Error: Agent loop not available for runtime patching."

        allowed = {"temperature", "max_tokens", "top_p", "top_k", "presence_penalty"}
        overrides = {k: v for k, v in patch.items() if k in allowed}
        if not overrides:
            return f"Error: No valid keys provided for patch_runtime. Allowed: {', '.join(sorted(allowed))}"

        # Update the volatile overrides in the loop
        self._agent_loop.runtime_overrides.update(overrides)

        log.info("runtime_config_patched", overrides=overrides)
        return (
            f"✅ Runtime configuration patched (volatile): {overrides}.\n"
            "These changes take effect immediately but will be lost if the agent restarts."
        )

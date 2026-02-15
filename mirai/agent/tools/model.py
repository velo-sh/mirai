"""Model Tool — discover and switch AI models at runtime.

Actions:
  - list_models:      Show all available models across providers.
  - set_active_model: Hot-swap to a different model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mirai.agent.tools.base import BaseTool
from mirai.logging import get_logger

if TYPE_CHECKING:
    from mirai.agent.tools.base import ToolContext

log = get_logger("mirai.tools.model")


class ModelTool(BaseTool):
    """Discover and switch AI models at runtime."""

    def __init__(self, context: ToolContext | None = None, **kwargs: Any) -> None:
        super().__init__(context)
        self._registry = self.context.registry
        self._provider = self.context.provider
        self._agent_loop = self.context.agent_loop

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "name": "mirai_model",
            "description": (
                "Discover and switch AI models at runtime. "
                "Actions: 'list_models' (discover all available models across providers), "
                "'set_active_model' (switch to a different model at runtime)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list_models", "set_active_model"],
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
                },
                "required": ["action"],
            },
        }

    async def execute(  # type: ignore[override]
        self,
        action: str,
        model: str | None = None,
    ) -> str:
        if action == "list_models":
            return await self._list_models()
        if action == "set_active_model":
            return await self._set_active_model(model)
        return f"Error: Unknown action '{action}'. Valid actions: list_models, set_active_model."

    # ------------------------------------------------------------------
    # Action: list_models
    # ------------------------------------------------------------------
    async def _list_models(self) -> str:
        """Return all available models across all configured providers."""
        if not self._registry:
            return "Error: Model registry not available."
        # Fetch quota data if provider has a QuotaManager
        quota_data: dict[str, float] | None = None
        qm = getattr(self._provider, "quota_manager", None) if self._provider else None
        if qm is not None:
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

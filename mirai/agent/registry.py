"""
Model Registry — persistent, async-refreshable cross-provider model catalog.

Provides a single source of truth for all available models across all
configured providers.  Discovery data is persisted to
``~/.mirai/model_registry.json`` and refreshed in the background via
each provider's ``list_models()`` method (which may call remote APIs).

Design:
  - ``config.toml`` owns the *startup default* (provider + model).
  - ``model_registry.json`` owns the *runtime state* (available models
    + active model override).
  - Priority: registry (runtime) > config.toml (default) > code default.
  - Reads are pure in-memory (``get_catalog_text()``).
  - Writes use copy-on-write + atomic swap for thread safety.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mirai.agent.registry_models import (
    RegistryData,
    RegistryModelEntry,
    RegistryProviderData,
)
from mirai.logging import get_logger

log = get_logger("mirai.registry")

# Provider name → (env var for API key, import path for provider class)
_PROVIDER_SPECS: list[tuple[str, str, str]] = [
    ("minimax", "MINIMAX_API_KEY", "mirai.agent.providers.minimax.MiniMaxProvider"),
    ("anthropic", "ANTHROPIC_API_KEY", "mirai.agent.providers.anthropic.AnthropicProvider"),
    ("openai", "OPENAI_API_KEY", "mirai.agent.providers.openai.OpenAIProvider"),
]


def _import_provider_class(import_path: str) -> type | None:
    """Dynamically import a provider class by dotted path."""
    try:
        module_path, class_name = import_path.rsplit(".", 1)
        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, class_name)  # type: ignore[no-any-return]
    except Exception as exc:
        log.warning("provider_import_failed", path=import_path, error=str(exc))
        return None


class ModelRegistry:
    """Persistent, async-refreshable cross-provider model catalog.

    Usage::

        registry = ModelRegistry()          # loads from disk
        await registry.refresh()            # discover via provider APIs
        text = registry.get_catalog_text()  # for tool response
    """

    PATH = Path.home() / ".mirai" / "model_registry.json"

    def __init__(self, config_provider: str | None = None, config_model: str | None = None):
        """Initialize registry.

        Args:
            config_provider: Default provider from config.toml (startup default).
            config_model: Default model from config.toml (startup default).
        """
        self._config_provider = config_provider
        self._config_model = config_model
        loaded = self._load()
        if isinstance(loaded, dict):
            # Compatibility with brittle mocks returning dict
            from mirai.agent.registry_models import RegistryData

            self._data: RegistryData = RegistryData.from_dict(loaded)
        else:
            self._data = loaded

    # ------------------------------------------------------------------
    # Load / Save
    # ------------------------------------------------------------------

    def _load(self) -> RegistryData:
        """Synchronous load from disk. Returns empty registry if missing/corrupt."""
        if not self.PATH.exists():
            log.info("registry_first_run", path=str(self.PATH))
            return RegistryData(
                active_provider=self._config_provider,
                active_model=self._config_model,
            )

        try:
            with open(self.PATH, encoding="utf-8") as f:
                raw = json.load(f)
            data = RegistryData.from_dict(raw)
            log.info(
                "registry_loaded",
                path=str(self.PATH),
                providers=len(data.providers),
            )
            return data
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("registry_load_failed", error=str(exc), path=str(self.PATH))
            return RegistryData(
                active_provider=self._config_provider,
                active_model=self._config_model,
            )

    def _save(self) -> None:
        """Write current state to disk. Fails gracefully."""
        try:
            self.PATH.parent.mkdir(parents=True, exist_ok=True)
            content = json.dumps(self._data.to_dict(), indent=2, ensure_ascii=False)
            # Atomic write: write to tmp then rename
            tmp_path = self.PATH.with_suffix(".json.tmp")
            tmp_path.write_text(content, encoding="utf-8")
            tmp_path.rename(self.PATH)
            log.debug("registry_saved", path=str(self.PATH))
        except OSError as exc:
            log.warning("registry_save_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Refresh (remote discovery)
    # ------------------------------------------------------------------

    async def refresh(self) -> None:
        """Re-scan all providers via their ``list_models()`` methods.

        For each provider with a configured API key, instantiate a
        lightweight client and call ``list_models()`` — which may hit
        a remote API (OpenAI) or return a static catalog (Anthropic,
        MiniMax).

        Uses copy-on-write: builds a new providers dict, then atomically
        swaps ``self._data``.
        """
        new_providers: dict[str, RegistryProviderData] = {}

        for pname, env_key, import_path in _PROVIDER_SPECS:
            api_key = os.getenv(env_key)
            if not api_key:
                # Keep existing model info but mark unavailable
                existing = self._data.providers.get(pname)
                new_providers[pname] = RegistryProviderData(
                    available=False,
                    env_key=env_key,
                    models=existing.models if existing else [],
                )
                continue

            # Try to instantiate provider and call list_models()
            try:
                cls = _import_provider_class(import_path)
                if cls is None:
                    continue

                provider = cls(api_key=api_key)
                models = await provider.list_models()
                new_providers[pname] = RegistryProviderData(
                    available=True,
                    env_key=env_key,
                    models=[
                        RegistryModelEntry(
                            id=m.id,
                            name=m.name,
                            description=m.description,
                            reasoning=m.reasoning,
                            vision=getattr(m, "supports_vision", False),
                        )
                        for m in models
                    ],
                )
                log.info("registry_provider_refreshed", provider=pname, model_count=len(models))
            except Exception as exc:
                log.warning("registry_refresh_failed", provider=pname, error=str(exc))
                # Keep last known state on failure
                existing = self._data.providers.get(pname)
                new_providers[pname] = (
                    existing
                    if existing
                    else RegistryProviderData(
                        available=False,
                        env_key=env_key,
                        models=[],
                    )
                )

        # Atomic swap (copy-on-write)
        new_data = RegistryData(
            version=getattr(self._data, "version", 1),
            last_refreshed=datetime.now(UTC).isoformat(),
            active_provider=self.active_provider,
            active_model=self.active_model,
            providers=new_providers,
        )
        self._data = new_data
        self._save()

    # ------------------------------------------------------------------
    # Read (non-blocking, in-memory)
    # ------------------------------------------------------------------

    @property
    def active_provider(self) -> str:
        """Resolve active provider: registry (runtime) > config.toml (default)."""
        data = self._data
        if isinstance(data, dict):
            return data.get("active_provider") or self._config_provider or "unknown"
        return data.active_provider or self._config_provider or "unknown"

    @property
    def active_model(self) -> str:
        """Resolve active model: registry (runtime) > config.toml (default)."""
        data = self._data
        if isinstance(data, dict):
            return data.get("active_model") or self._config_model or "unknown"
        return data.active_model or self._config_model or "unknown"

    def get_catalog_text(self, quota_data: dict[str, float] | None = None) -> str:
        """Format the full model catalog as human-readable text.

        Args:
            quota_data: Optional dict of model_id → used_pct from QuotaManager.
                        When provided, exhausted models are annotated.

        This is a pure in-memory read — no I/O, no blocking.
        """
        lines = [
            f"Current provider: {self.active_provider}",
            f"Current model: {self.active_model}",
        ]

        if isinstance(self._data, dict):
            last_refreshed = self._data.get("last_refreshed")
            providers = self._data.get("providers", {})
        else:
            last_refreshed = self._data.last_refreshed
            providers = self._data.providers

        if last_refreshed:
            lines.append(f"Last refreshed: {last_refreshed}")

        available_providers = {
            k: v
            for k, v in providers.items()
            if (v.get("available") if isinstance(v, dict) else getattr(v, "available", False))
        }

        if not available_providers:
            lines.append("\nNo providers with configured API keys found.")
            return "\n".join(lines)

        lines.append(f"\nAvailable models ({len(available_providers)} provider(s)):")

        for pname, pdata in available_providers.items():
            is_active = pname == self.active_provider
            lines.append(f"\n### {pname.upper()}{' (active)' if is_active else ''}:")

            models = pdata.get("models", []) if isinstance(pdata, dict) else pdata.models
            if not models:
                lines.append("  (no models discovered)")
                continue
            for m in models:
                mid = m.get("id") if isinstance(m, dict) else m.id
                mname = m.get("name") if isinstance(m, dict) else m.name
                mdesc = m.get("description") if isinstance(m, dict) else m.description
                mreasoning = m.get("reasoning") if isinstance(m, dict) else m.reasoning
                mvision = m.get("vision") if isinstance(m, dict) else m.vision

                marker = " ← current" if (is_active and mid == self.active_model) else ""
                desc = f"  - {mid}: {mdesc or mname}"
                if mreasoning:
                    desc += " [reasoning]"
                if mvision:
                    desc += " [vision]"
                # Annotate quota status if available
                if quota_data and mid in quota_data:
                    pct = quota_data[mid]
                    if pct >= 100.0:
                        desc += " ⚠️ exhausted"
                    elif pct >= 80.0:
                        desc += f" ({pct:.0f}% used)"
                lines.append(desc + marker)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Agent mutations
    # ------------------------------------------------------------------

    def find_provider_for_model(self, model_id: str) -> str | None:
        """Look up which provider owns the given model ID.

        Returns the provider name (e.g. 'minimax', 'anthropic') or None.
        """
        providers = self._data.get("providers", {}) if isinstance(self._data, dict) else self._data.providers
        for pname, pdata in providers.items():
            available = pdata.get("available") if isinstance(pdata, dict) else pdata.available
            if not available:
                continue
            models = pdata.get("models", []) if isinstance(pdata, dict) else pdata.models
            for m in models:
                mid = m.get("id") if isinstance(m, dict) else m.id
                if mid == model_id:
                    return pname
        return None

    async def set_active(self, provider: str, model: str) -> None:
        """Set the active provider + model (runtime override). Persists to disk."""
        new_data = RegistryData(
            version=getattr(self._data, "version", 1),
            last_refreshed=getattr(self._data, "last_refreshed", None),
            active_provider=provider,
            active_model=model,
            providers=getattr(self._data, "providers", {}),
        )
        self._data = new_data
        self._save()
        log.info("registry_active_changed", provider=provider, model=model)


# ---------------------------------------------------------------------------
# Background refresh task
# ---------------------------------------------------------------------------


async def registry_refresh_loop(registry: ModelRegistry, interval: int = 300, quota_manager: Any = None) -> None:
    """Periodically refresh the model registry in the background.

    Args:
        registry: The ModelRegistry instance to refresh.
        interval: Seconds between refreshes (default: 5 minutes).
        quota_manager: Optional QuotaManager to refresh alongside the registry.
    """
    # Initial refresh on startup
    try:
        await registry.refresh()
    except Exception as exc:
        log.warning("registry_initial_refresh_failed", error=str(exc))

    while True:
        await asyncio.sleep(interval)
        try:
            await registry.refresh()
            if quota_manager:
                try:
                    await quota_manager._maybe_refresh()
                except Exception as qe:
                    log.warning("quota_periodic_refresh_failed", error=str(qe))
        except Exception as exc:
            log.warning("registry_periodic_refresh_failed", error=str(exc))

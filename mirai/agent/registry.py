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
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from mirai.agent.registry_models import (
    RegistryData,
    RegistryModelEntry,
    RegistryProviderData,
)
from mirai.logging import get_logger

if TYPE_CHECKING:
    pass

log = get_logger("mirai.registry")


@runtime_checkable
class EnrichmentSource(Protocol):
    """Protocol for external model metadata enrichment sources."""

    async def fetch(self) -> dict[str, Any]: ...
    def enrich(self, entry: RegistryModelEntry, provider: str) -> RegistryModelEntry: ...


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


def _format_context(ctx: int | None) -> str:
    """Format context window as human-readable string (e.g. 200K, 1M)."""
    if ctx is None:
        return ""
    if ctx >= 1_000_000:
        return f"{ctx / 1_000_000:.0f}M ctx"
    if ctx >= 1_000:
        return f"{ctx / 1_000:.0f}K ctx"
    return f"{ctx} ctx"


def _format_pricing(inp: float | None, out: float | None) -> str:
    """Format pricing as '$in/$out per 1M tokens'."""
    if inp is None and out is None:
        return ""
    ip = f"${inp:.2f}" if inp is not None else "?"
    op = f"${out:.2f}" if out is not None else "?"
    return f"{ip}/{op}"


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
        self._data: RegistryData = self._load()
        self._enrichment_source: EnrichmentSource | None = None

    def set_enrichment_source(self, source: EnrichmentSource) -> None:
        """Register an external metadata enrichment source.

        Call this before the first ``refresh()`` to enable automatic
        enrichment of discovered models with external data (e.g. pricing,
        context limits).
        """
        self._enrichment_source = source

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

        If an enrichment source (e.g. ModelsDevSource) is configured,
        fills in any missing metadata fields (provider-native data wins).

        Uses copy-on-write: builds a new providers dict, then atomically
        swaps ``self._data``.

        The enrichment fetch runs **concurrently** with provider scanning
        so slow external APIs never block discovery.
        """

        # ------------------------------------------------------------------
        # Phase 1: Scan providers + fetch enrichment data concurrently
        # ------------------------------------------------------------------
        async def _scan_providers() -> dict[str, RegistryProviderData]:
            result: dict[str, RegistryProviderData] = {}
            for pname, env_key, import_path in _PROVIDER_SPECS:
                api_key = os.getenv(env_key)
                if not api_key:
                    existing = self._data.providers.get(pname)
                    result[pname] = RegistryProviderData(
                        available=False,
                        env_key=env_key,
                        models=existing.models if existing else [],
                    )
                    continue

                try:
                    cls = _import_provider_class(import_path)
                    if cls is None:
                        continue

                    provider = cls(api_key=api_key)
                    raw_models = await provider.list_models()
                    models = [RegistryModelEntry.from_model_info(m) for m in raw_models]
                    result[pname] = RegistryProviderData(
                        available=True,
                        env_key=env_key,
                        models=models,
                    )
                    log.info("registry_provider_refreshed", provider=pname, model_count=len(models))
                except Exception as exc:
                    log.warning("registry_refresh_failed", provider=pname, error=str(exc))
                    existing = self._data.providers.get(pname)
                    result[pname] = (
                        existing
                        if existing
                        else RegistryProviderData(
                            available=False,
                            env_key=env_key,
                            models=[],
                        )
                    )
            return result

        async def _fetch_enrichment() -> dict[str, Any]:
            if self._enrichment_source is None:
                return {}
            try:
                return await asyncio.wait_for(
                    self._enrichment_source.fetch(),
                    timeout=20,
                )
            except TimeoutError:
                log.warning("enrichment_fetch_timeout")
                return {}
            except Exception as exc:
                log.warning("enrichment_fetch_failed", error=str(exc))
                return {}

        # Run both concurrently — enrichment never blocks provider scanning
        new_providers, enrichment_data = await asyncio.gather(
            _scan_providers(),
            _fetch_enrichment(),
        )

        # ------------------------------------------------------------------
        # Phase 2: Enrich models with external data (fast, in-memory only)
        # ------------------------------------------------------------------
        enriched_count = 0
        total_count = 0
        if self._enrichment_source is not None and enrichment_data:
            for pname, pdata in new_providers.items():
                if not pdata.models:
                    continue
                enriched_models = []
                for m in pdata.models:
                    total_count += 1
                    before_price = m.input_price
                    enriched_m = self._enrichment_source.enrich(m, pname)
                    if enriched_m.input_price != before_price or enriched_m.context_window is not None:
                        enriched_count += 1
                    enriched_models.append(enriched_m)
                # Copy-on-write: create new RegistryProviderData
                new_providers[pname] = RegistryProviderData(
                    available=pdata.available,
                    env_key=pdata.env_key,
                    models=enriched_models,
                )
            log.info(
                "enrichment_applied",
                enriched=enriched_count,
                total=total_count,
            )

        # Atomic swap (copy-on-write)
        new_data = RegistryData(
            version=self._data.version,
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
        return self._data.active_provider or self._config_provider or "unknown"

    @property
    def active_model(self) -> str:
        """Resolve active model: registry (runtime) > config.toml (default)."""
        return self._data.active_model or self._config_model or "unknown"

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

        if self._data.last_refreshed:
            lines.append(f"Last refreshed: {self._data.last_refreshed}")

        available_providers = {k: v for k, v in self._data.providers.items() if v.available}

        if not available_providers:
            lines.append("\nNo providers with configured API keys found.")
            return "\n".join(lines)

        lines.append(f"\nAvailable models ({len(available_providers)} provider(s)):")

        for pname, pdata in available_providers.items():
            is_active = pname == self.active_provider
            lines.append(f"\n### {pname.upper()}{' (active)' if is_active else ''}:")

            if not pdata.models:
                lines.append("  (no models discovered)")
                continue

            for m in pdata.models:
                marker = " ← current" if (is_active and m.id == self.active_model) else ""

                # Main line: id + description + capability tags
                tags = []
                if m.reasoning:
                    tags.append("reasoning")
                if m.vision:
                    tags.append("vision")
                if m.supports_tool_use:
                    tags.append("tool_use")
                if m.supports_json_mode:
                    tags.append("json_mode")

                tag_str = " ".join(f"[{t}]" for t in tags)
                desc = f"  - {m.id}: {m.description or m.name}"
                if tag_str:
                    desc += f" {tag_str}"

                # Quota annotation
                if quota_data and m.id in quota_data:
                    pct = quota_data[m.id]
                    if pct >= 100.0:
                        desc += " ⚠️ exhausted"
                    elif pct >= 80.0:
                        desc += f" ({pct:.0f}% used)"

                lines.append(desc + marker)

                # Detail line: context + pricing + cutoff
                details = []
                ctx = _format_context(m.context_window)
                if ctx:
                    details.append(ctx)
                pricing = _format_pricing(m.input_price, m.output_price)
                if pricing:
                    details.append(pricing)
                if m.knowledge_cutoff:
                    details.append(f"cutoff {m.knowledge_cutoff}")
                if details:
                    lines.append(f"    {' · '.join(details)}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Agent mutations
    # ------------------------------------------------------------------

    def find_provider_for_model(self, model_id: str) -> str | None:
        """Look up which provider owns the given model ID.

        Returns the provider name (e.g. 'minimax', 'anthropic') or None.
        """
        for pname, pdata in self._data.providers.items():
            if not pdata.available:
                continue
            for m in pdata.models:
                if m.id == model_id:
                    return pname
        return None

    async def set_active(self, provider: str, model: str) -> None:
        """Set the active provider + model (runtime override). Persists to disk."""
        new_data = RegistryData(
            version=self._data.version,
            last_refreshed=self._data.last_refreshed,
            active_provider=provider,
            active_model=model,
            providers=self._data.providers,
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

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
from dataclasses import dataclass
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
    from mirai.agent.free_providers import FreeProviderData, FreeProviderSource

log = get_logger("mirai.registry")


@runtime_checkable
class EnrichmentSource(Protocol):
    """Protocol for external model metadata enrichment sources."""

    async def fetch(self) -> dict[str, Any]: ...
    def enrich(self, entry: RegistryModelEntry, provider: str) -> RegistryModelEntry: ...


@dataclass(frozen=True)
class ProviderSpec:
    """Specification for a provider that the registry can scan."""

    name: str
    env_key: str
    import_path: str
    base_url: str | None = None
    signup_url: str | None = None
    notes: str | None = None


# Built-in providers (dedicated classes)
_BUILTIN_SPECS: list[ProviderSpec] = [
    ProviderSpec("minimax", "MINIMAX_API_KEY", "mirai.agent.providers.minimax.MiniMaxProvider"),
    ProviderSpec("anthropic", "ANTHROPIC_API_KEY", "mirai.agent.providers.anthropic.AnthropicProvider"),
    ProviderSpec("openai", "OPENAI_API_KEY", "mirai.agent.providers.openai.OpenAIProvider"),
]


def _build_provider_specs() -> list[ProviderSpec]:
    """Combine built-in and free provider specs into a single scan list."""
    from mirai.agent.free_providers import FREE_PROVIDER_SPECS

    specs = list(_BUILTIN_SPECS)
    builtin_names = {s.name for s in specs}
    for fp in FREE_PROVIDER_SPECS:
        if fp.name in builtin_names:
            continue  # don't duplicate built-in providers
        specs.append(
            ProviderSpec(
                name=fp.name,
                env_key=fp.env_key,
                import_path="mirai.agent.providers.openai.OpenAIProvider",
                base_url=fp.base_url,
                signup_url=fp.signup_url,
                notes=fp.notes,
            )
        )
    return specs


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
        self._free_source: FreeProviderSource | None = None
        self._health_status: dict[str, Any] = {}  # name -> ProviderHealthStatus

    @classmethod
    def for_testing(
        cls,
        *,
        data: RegistryData | None = None,
        path: Path | None = None,
        config_provider: str | None = None,
        config_model: str | None = None,
        enrichment_source: EnrichmentSource | None = None,
        free_source: FreeProviderSource | None = None,
        health_status: dict[str, Any] | None = None,
    ) -> ModelRegistry:
        """Create a ModelRegistry for testing without touching disk.

        Bypasses ``__init__`` (which calls ``_load()`` from disk) and directly
        sets internal attributes.  Using this factory ensures tests stay in
        sync with any future attributes added to ``__init__``.
        """
        reg = cls.__new__(cls)
        reg._config_provider = config_provider
        reg._config_model = config_model
        reg._data = data or RegistryData()
        reg._enrichment_source = enrichment_source
        reg._free_source = free_source
        reg._health_status = health_status or {}
        if path is not None:
            reg.PATH = path  # type: ignore[misc]
        return reg

    def set_enrichment_source(self, source: EnrichmentSource) -> None:
        """Register an external metadata enrichment source.

        Call this before the first ``refresh()`` to enable automatic
        enrichment of discovered models with external data (e.g. pricing,
        context limits).
        """
        self._enrichment_source = source

    def set_free_source(self, source: FreeProviderSource) -> None:
        """Register a free-provider discovery source.

        When set, ``refresh()`` will concurrently fetch public model
        catalogs (OpenRouter, SambaNova, etc.) and merge them into the
        registry so unconfigured providers show available models.
        """
        self._free_source = source

    def update_health(self, health: dict[str, Any]) -> None:
        """Update cached health-check results for free providers."""
        self._health_status = health

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

        # Phase 1: Scan providers + fetch enrichment data concurrently
        new_providers, enrichment_data, free_data = await asyncio.gather(
            self._scan_providers(),
            self._fetch_enrichment(),
            self._fetch_free_models(),
        )

        # Phase 2: Enrich with external metadata (fast, in-memory)
        self._enrich_models(new_providers, enrichment_data)

        # Phase 3: Merge free-provider public model data
        self._merge_free_providers(new_providers, free_data)

        # Phase 4: Apply static capability tags as fallback
        self._apply_capability_tags(new_providers)

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

    async def _scan_providers(self) -> dict[str, RegistryProviderData]:
        """Phase 1: Scan all provider specs and collect available models."""
        result: dict[str, RegistryProviderData] = {}
        all_specs = _build_provider_specs()
        for spec in all_specs:
            api_key = os.getenv(spec.env_key)
            if not api_key:
                existing = self._data.providers.get(spec.name)
                result[spec.name] = RegistryProviderData(
                    available=False,
                    env_key=spec.env_key,
                    models=existing.models if existing else [],
                )
                continue

            try:
                cls = _import_provider_class(spec.import_path)
                if cls is None:
                    continue

                # Pass base_url and provider_name for free providers
                init_kwargs: dict[str, Any] = {"api_key": api_key}
                if spec.base_url:
                    init_kwargs["base_url"] = spec.base_url
                    init_kwargs["provider_name"] = spec.name
                provider = cls(**init_kwargs)
                raw_models = await provider.list_models()
                models = [RegistryModelEntry.from_model_info(m) for m in raw_models]
                result[spec.name] = RegistryProviderData(
                    available=True,
                    env_key=spec.env_key,
                    models=models,
                )
                log.info("registry_provider_refreshed", provider=spec.name, model_count=len(models))
            except Exception as exc:
                log.warning("registry_refresh_failed", provider=spec.name, error=str(exc))
                existing = self._data.providers.get(spec.name)
                result[spec.name] = (
                    existing
                    if existing
                    else RegistryProviderData(
                        available=False,
                        env_key=spec.env_key,
                        models=[],
                    )
                )
        return result

    async def _fetch_enrichment(self) -> dict[str, Any]:
        """Fetch external enrichment data (e.g. models.dev pricing)."""
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

    async def _fetch_free_models(self) -> dict[str, FreeProviderData]:
        """Fetch public model catalogs from free provider sources."""
        if self._free_source is None:
            return {}
        try:
            return await asyncio.wait_for(
                self._free_source.fetch(),
                timeout=20,
            )
        except TimeoutError:
            log.warning("free_provider_fetch_timeout")
            return {}
        except Exception as exc:
            log.warning("free_provider_fetch_failed", error=str(exc))
            return {}

    def _enrich_models(
        self,
        providers: dict[str, RegistryProviderData],
        enrichment_data: dict[str, Any],
    ) -> None:
        """Phase 2: Enrich discovered models with external metadata."""
        if self._enrichment_source is None or not enrichment_data:
            return

        enriched_count = 0
        total_count = 0
        for pname, pdata in providers.items():
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
            providers[pname] = RegistryProviderData(
                available=pdata.available,
                env_key=pdata.env_key,
                models=enriched_models,
            )
        log.info("enrichment_applied", enriched=enriched_count, total=total_count)

    def _merge_free_providers(
        self,
        providers: dict[str, RegistryProviderData],
        free_data: dict[str, FreeProviderData],
    ) -> None:
        """Phase 3: Merge free-provider public model data into unconfigured providers."""
        if not free_data:
            return

        for fp_name, fp_data in free_data.items():
            fp_pdata = providers.get(fp_name)
            if fp_pdata is not None and not fp_pdata.available and fp_data.models:
                free_models = [
                    RegistryModelEntry(
                        id=fm.id,
                        name=fm.name,
                        vision=fm.vision,
                        reasoning=fm.reasoning,
                        supports_tool_use=fm.supports_tool_use,
                        context_window=fm.context_length,
                    )
                    for fm in fp_data.models
                ]
                providers[fp_name] = RegistryProviderData(
                    available=False,
                    env_key=fp_pdata.env_key,
                    models=free_models,
                )
        log.info(
            "free_provider_data_merged",
            providers_with_models=sum(1 for fp in free_data.values() if fp.models),
        )

    def _apply_capability_tags(self, providers: dict[str, RegistryProviderData]) -> None:
        """Phase 4: Apply static capability tags as fallback for providers without API metadata."""
        from mirai.agent.free_model_capabilities import enrich_capabilities

        free_provider_names = {spec.name for spec in _build_provider_specs() if spec.signup_url}
        for pname in free_provider_names:
            cap_pdata = providers.get(pname)
            if cap_pdata and cap_pdata.models:
                for m in cap_pdata.models:
                    enrich_capabilities(m)

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

    def _format_model_entry(
        self,
        model: RegistryModelEntry,
        is_current: bool,
        quota_data: dict[str, float] | None,
    ) -> list[str]:
        """Format a single model entry as catalog lines."""
        lines: list[str] = []
        marker = " ← current" if is_current else ""

        tags = []
        if model.reasoning:
            tags.append("reasoning")
        if model.vision:
            tags.append("vision")
        if model.supports_tool_use:
            tags.append("tool_use")
        if model.supports_json_mode:
            tags.append("json_mode")

        tag_str = " ".join(f"[{t}]" for t in tags)
        desc = f"  - {model.id}: {model.description or model.name}"
        if tag_str:
            desc += f" {tag_str}"

        if quota_data and model.id in quota_data:
            pct = quota_data[model.id]
            if pct >= 100.0:
                desc += " ⚠️ exhausted"
            elif pct >= 80.0:
                desc += f" ({pct:.0f}% used)"

        lines.append(desc + marker)

        details = []
        ctx = _format_context(model.context_window)
        if ctx:
            details.append(ctx)
        pricing = _format_pricing(model.input_price, model.output_price)
        if pricing:
            details.append(pricing)
        if model.knowledge_cutoff:
            details.append(f"cutoff {model.knowledge_cutoff}")
        if details:
            lines.append(f"    {' · '.join(details)}")

        return lines

    def _format_unconfigured_providers(self) -> list[str]:
        """Format the unconfigured free providers suggestion section."""
        available_providers = {k for k, v in self._data.providers.items() if v.available}
        unconfigured_free: list[ProviderSpec] = []
        for spec in _build_provider_specs():
            if spec.signup_url and spec.name not in available_providers:
                unconfigured_free.append(spec)

        if not unconfigured_free:
            return []

        lines = ["\n### Free providers (not configured):"]
        for spec in unconfigured_free:
            note = f" — {spec.notes}" if spec.notes else ""
            cat_pdata = self._data.providers.get(spec.name)
            model_hint = ""
            if cat_pdata and cat_pdata.models:
                model_hint = f" ({len(cat_pdata.models)} models available)"
            lines.append(f"  ✗ {spec.name}{model_hint}: set {spec.env_key} (signup: {spec.signup_url}){note}")
        return lines

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
        else:
            lines.append(f"\nAvailable models ({len(available_providers)} provider(s)):")

            for pname, pdata in available_providers.items():
                is_active = pname == self.active_provider
                # Show health indicator for free providers
                health_tag = ""
                hs = self._health_status.get(pname)
                if hs is not None:
                    if hs.healthy:
                        health_tag = f" ✓ {hs.latency_ms:.0f}ms" if hs.latency_ms else " ✓"
                    else:
                        health_tag = " ✗ unhealthy"
                lines.append(f"\n### {pname.upper()}{' (active)' if is_active else ''}{health_tag}:")

                if not pdata.models:
                    lines.append("  (no models discovered)")
                    continue

                for m in pdata.models:
                    is_current = is_active and m.id == self.active_model
                    lines.extend(self._format_model_entry(m, is_current, quota_data))

        lines.extend(self._format_unconfigured_providers())

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

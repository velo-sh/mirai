"""
External model metadata source — models.dev API.

Fetches model metadata from https://models.dev/api.json, normalises it into
``ExternalModelData`` records, and persists a disk cache at
``~/.mirai/models_dev_cache.json``.

Design rules:
  - **Fail-open**: network errors surface as warnings; enrichment is skipped.
  - **Provider-wins**: ``enrich()`` only fills ``None`` fields — native
    provider data is never overwritten.
  - **Cache TTL**: defaults to 1 hour.  Cache is always read first; if
    fresh enough the network round-trip is skipped entirely.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from mirai.agent.registry_models import RegistryModelEntry
from mirai.logging import get_logger

log = get_logger("mirai.models_dev")

# ---------------------------------------------------------------------------
# Normalised external model data
# ---------------------------------------------------------------------------

_PROVIDER_NAME_MAP: dict[str, str] = {
    "minimax": "minimax",
    "anthropic": "anthropic",
    "openai": "openai",
    "google": "google",
    "deepseek": "deepseek",
    "x-ai": "xai",
    "qwen": "qwen",
    "z-ai": "zai",
    "zai": "zai",
    "moonshotai": "moonshot",
}


@dataclass
class ExternalModelData:
    """Normalised external metadata for one model."""

    id: str
    provider: str
    name: str | None = None
    input_cost: float | None = None
    output_cost: float | None = None
    context_limit: int | None = None
    output_limit: int | None = None
    tool_call: bool | None = None
    reasoning: bool | None = None
    vision: bool | None = None
    knowledge_cutoff: str | None = None
    input_modalities: list[str] = field(default_factory=list)
    output_modalities: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Models.dev source
# ---------------------------------------------------------------------------


class ModelsDevSource:
    """Fetches and caches metadata from the models.dev public API.

    Usage::

        source = ModelsDevSource()
        data = await source.fetch()           # {lookup_key: ExternalModelData}
        enriched = source.enrich(entry, "minimax")
    """

    CACHE_PATH = Path.home() / ".mirai" / "models_dev_cache.json"
    CACHE_TTL = 3600  # seconds (1 hour)
    MODELS_DEV_URL = "https://models.dev/api.json"
    REQUEST_TIMEOUT = 15  # seconds

    def __init__(self, *, cache_path: Path | None = None) -> None:
        if cache_path is not None:
            self.CACHE_PATH = cache_path
        self._data: dict[str, ExternalModelData] = {}
        # O(1) lookup index: maps multiple key variants -> ExternalModelData
        self._index: dict[str, ExternalModelData] = {}
        self._last_cache_timestamp: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch(self) -> dict[str, ExternalModelData]:
        """Load from cache or fetch fresh.  Returns ``{lookup_key: data}``.

        Lookup keys are ``provider/model_id`` (e.g. ``minimax/minimax-m2.5``)
        and also bare ``model_id`` as a fallback.
        """
        # Try disk cache first
        cached = self._load_cache()
        if cached is not None:
            self._data = cached
            self._build_index()
            age_s = round(time.time() - (self._last_cache_timestamp or 0), 1)
            log.info("models_dev_cache_hit", models=len(self._data), age_s=age_s)
            return self._data

        # Fetch from network
        raw = await self._fetch_api()
        if raw is not None:
            self._data = self._normalise(raw)
            self._build_index()
            self._save_cache()
            log.info("models_dev_fetched", models=len(self._data))
        else:
            log.warning("models_dev_fetch_failed_using_empty")

        return self._data

    def enrich(self, entry: RegistryModelEntry, provider: str) -> RegistryModelEntry:
        """Fill ``None`` fields on *entry* from external data.

        Provider-native data always wins (only ``None`` fields are filled).
        Uses the O(1) lookup index for fast matching.
        """
        ext = self._lookup(entry.id, provider)
        if ext is None:
            return entry

        # Only fill None / default fields — never overwrite provider data
        if entry.input_price is None and ext.input_cost is not None:
            entry.input_price = ext.input_cost
        if entry.output_price is None and ext.output_cost is not None:
            entry.output_price = ext.output_cost
        if entry.context_window is None and ext.context_limit is not None:
            entry.context_window = ext.context_limit
        if entry.max_output_tokens is None and ext.output_limit is not None:
            entry.max_output_tokens = ext.output_limit
        if entry.knowledge_cutoff is None and ext.knowledge_cutoff is not None:
            entry.knowledge_cutoff = ext.knowledge_cutoff

        return entry

    # ------------------------------------------------------------------
    # Lookup index (O(1))
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        """Pre-compute a fast lookup index from ``_data``.

        The index maps multiple key variants for each model to enable
        O(1) lookup regardless of how the registry names things.
        """
        idx: dict[str, ExternalModelData] = {}
        for key, ext in self._data.items():
            # Exact key from _data (provider_key/model_id or bare model_id)
            idx[key] = ext
            # Also index by lowercase variants for case-insensitive provider
            lk = key.lower()
            if lk not in idx:
                idx[lk] = ext
            # Index by mapped_provider/model_id
            mapped_key = f"{ext.provider}/{ext.id}"
            if mapped_key not in idx:
                idx[mapped_key] = ext
        self._index = idx

    def _lookup(self, model_id: str, provider: str) -> ExternalModelData | None:
        """O(1) lookup by ``provider/model_id`` or bare ``model_id``.

        Checks the pre-built index so no string construction loop is needed.
        """
        return (
            self._index.get(f"{provider}/{model_id}")
            or self._index.get(model_id)
            or self._index.get(f"{provider.lower()}/{model_id}")
        )

    # ------------------------------------------------------------------
    # Cache I/O
    # ------------------------------------------------------------------

    def _load_cache(self) -> dict[str, ExternalModelData] | None:
        """Load cache from disk if fresh enough.  Returns ``None`` on miss."""
        if not self.CACHE_PATH.exists():
            return None

        try:
            raw = json.loads(self.CACHE_PATH.read_text(encoding="utf-8"))
            ts = raw.get("timestamp", 0)
            if time.time() - ts > self.CACHE_TTL:
                log.debug("models_dev_cache_stale", age_s=round(time.time() - ts, 1))
                return None
            self._last_cache_timestamp = ts

            result: dict[str, ExternalModelData] = {}
            for key, item in raw.get("models", {}).items():
                result[key] = ExternalModelData(
                    id=item["id"],
                    provider=item["provider"],
                    name=item.get("name"),
                    input_cost=item.get("input_cost"),
                    output_cost=item.get("output_cost"),
                    context_limit=item.get("context_limit"),
                    output_limit=item.get("output_limit"),
                    tool_call=item.get("tool_call"),
                    reasoning=item.get("reasoning"),
                    vision=item.get("vision"),
                    knowledge_cutoff=item.get("knowledge_cutoff"),
                    input_modalities=item.get("input_modalities", []),
                    output_modalities=item.get("output_modalities", []),
                )
            return result
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            log.warning("models_dev_cache_corrupt", error=str(exc))
            return None

    def _save_cache(self) -> None:
        """Persist current data to disk cache."""
        try:
            self.CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            payload: dict[str, Any] = {
                "timestamp": time.time(),
                "models": {},
            }
            for key, item in self._data.items():
                payload["models"][key] = {
                    "id": item.id,
                    "provider": item.provider,
                    "name": item.name,
                    "input_cost": item.input_cost,
                    "output_cost": item.output_cost,
                    "context_limit": item.context_limit,
                    "output_limit": item.output_limit,
                    "tool_call": item.tool_call,
                    "reasoning": item.reasoning,
                    "vision": item.vision,
                    "knowledge_cutoff": item.knowledge_cutoff,
                    "input_modalities": item.input_modalities,
                    "output_modalities": item.output_modalities,
                }

            tmp = self.CACHE_PATH.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.rename(self.CACHE_PATH)
            log.debug("models_dev_cache_saved", path=str(self.CACHE_PATH))
        except OSError as exc:
            log.warning("models_dev_cache_save_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Network fetch + normalisation
    # ------------------------------------------------------------------

    async def _fetch_api(self) -> dict[str, Any] | None:
        """GET models.dev/api.json.  Returns raw JSON or ``None`` on error."""
        try:
            async with httpx.AsyncClient(timeout=self.REQUEST_TIMEOUT) as client:
                resp = await client.get(self.MODELS_DEV_URL)
                resp.raise_for_status()
                return resp.json()  # type: ignore[no-any-return]
        except (httpx.HTTPError, json.JSONDecodeError, OSError) as exc:
            log.warning("models_dev_api_error", error=str(exc))
            return None

    def _normalise(self, raw: dict[str, Any]) -> dict[str, ExternalModelData]:
        """Convert raw models.dev JSON into ``{lookup_key: ExternalModelData}``.

        The API structure is::

            { provider_id: { id, name, models: { model_id: { ... } } } }

        We index by both ``provider_id/model_id`` AND bare ``model_id``
        so lookup can succeed regardless of how the registry names things.
        """
        result: dict[str, ExternalModelData] = {}

        for provider_key, provider_blob in raw.items():
            if not isinstance(provider_blob, dict):
                continue
            models = provider_blob.get("models")
            if not isinstance(models, dict):
                continue

            # Map provider_key to our internal provider name
            mapped_provider = _PROVIDER_NAME_MAP.get(provider_key, provider_key)

            for model_id, model_data in models.items():
                if not isinstance(model_data, dict):
                    continue

                cost = model_data.get("cost", {})
                limit = model_data.get("limit", {})
                modalities = model_data.get("modalities", {})
                input_mods = modalities.get("input", [])

                ext = ExternalModelData(
                    id=model_id,
                    provider=mapped_provider,
                    name=model_data.get("name"),
                    input_cost=cost.get("input"),
                    output_cost=cost.get("output"),
                    context_limit=limit.get("context"),
                    output_limit=limit.get("output"),
                    tool_call=model_data.get("tool_call"),
                    reasoning=model_data.get("reasoning"),
                    vision="image" in input_mods or "video" in input_mods,
                    knowledge_cutoff=model_data.get("knowledge"),
                    input_modalities=input_mods,
                    output_modalities=modalities.get("output", []),
                )

                # Index by provider/model_id and bare model_id
                full_key = f"{provider_key}/{model_id}"
                result[full_key] = ext
                # Only set bare key if not already taken (first wins)
                if model_id not in result:
                    result[model_id] = ext

        return result

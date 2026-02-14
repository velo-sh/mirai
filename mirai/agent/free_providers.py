"""Free LLM provider catalog and runtime discovery.

Curated from the `free-llm-api-resources` project
(https://github.com/cheahjs/free-llm-api-resources).

This module provides:

1. ``FREE_PROVIDER_SPECS`` — static catalog of OpenAI-compatible free
   providers (name, base_url, env key, signup link, etc.).

2. ``FreeProviderSource`` — async runtime discovery that calls public
   APIs (OpenRouter ``/models``, SambaNova ``/api/pricing``) to fetch
   available free models, with local caching (TTL 1 h).

3. ``MODEL_NAME_MAP`` — model-id → human-readable name mapping, copied
   from ``free-llm-api-resources/src/data.py``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from mirai.logging import get_logger

log = get_logger("mirai.free_providers")


# ---------------------------------------------------------------------------
# Provider specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FreeProviderSpec:
    """An OpenAI-compatible provider that offers a free tier."""

    name: str
    """Internal identifier, e.g. ``"groq"``."""

    display_name: str
    """Human-facing label, e.g. ``"Groq"``."""

    env_key: str
    """Environment variable that holds the API key, e.g. ``"GROQ_API_KEY"``."""

    base_url: str
    """OpenAI-compatible base URL, e.g. ``"https://api.groq.com/openai/v1"``."""

    signup_url: str
    """Where users register for a free key."""

    public_api: bool
    """If True, model listing works without an API key."""

    notes: str | None = None
    """Brief human-readable description of limits / highlights."""


FREE_PROVIDER_SPECS: list[FreeProviderSpec] = [
    FreeProviderSpec(
        name="groq",
        display_name="Groq",
        env_key="GROQ_API_KEY",
        base_url="https://api.groq.com/openai/v1",
        signup_url="https://console.groq.com",
        public_api=False,
        notes="14,400 req/day, fastest inference",
    ),
    FreeProviderSpec(
        name="openrouter",
        display_name="OpenRouter",
        env_key="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        signup_url="https://openrouter.ai",
        public_api=True,
        notes="50 req/day (1000 with $10 topup), 30+ free models",
    ),
    FreeProviderSpec(
        name="cerebras",
        display_name="Cerebras",
        env_key="CEREBRAS_API_KEY",
        base_url="https://api.cerebras.ai/v1",
        signup_url="https://cloud.cerebras.ai",
        public_api=False,
        notes="14,400 req/day, ultra-fast inference",
    ),
    FreeProviderSpec(
        name="mistral",
        display_name="Mistral",
        env_key="MISTRAL_API_KEY",
        base_url="https://api.mistral.ai/v1",
        signup_url="https://console.mistral.ai",
        public_api=False,
        notes="1M tokens/min (Experiment plan, data training opt-in)",
    ),
    FreeProviderSpec(
        name="sambanova",
        display_name="SambaNova",
        env_key="SAMBANOVA_API_KEY",
        base_url="https://api.sambanova.ai/v1",
        signup_url="https://cloud.sambanova.ai",
        public_api=True,
        notes="Trial $5/3 months — DeepSeek, Llama 4, Qwen3",
    ),
    FreeProviderSpec(
        name="chutes",
        display_name="Chutes",
        env_key="CHUTES_API_KEY",
        base_url="https://api.chutes.ai/v1",
        signup_url="https://chutes.ai",
        public_api=True,
        notes="Free models with $0 per-token pricing",
    ),
]

_SPECS_BY_NAME: dict[str, FreeProviderSpec] = {s.name: s for s in FREE_PROVIDER_SPECS}


def get_free_provider_spec(name: str) -> FreeProviderSpec | None:
    """Look up a free provider spec by name."""
    return _SPECS_BY_NAME.get(name)


# ---------------------------------------------------------------------------
# Free model entry
# ---------------------------------------------------------------------------


@dataclass
class FreeModelEntry:
    """A model discovered from a free provider's API."""

    id: str
    name: str
    provider: str
    limits: dict[str, int | float] | None = None

    # Capabilities (populated from API metadata when available)
    context_length: int | None = None
    vision: bool = False
    reasoning: bool = False
    supports_tool_use: bool = False


# ---------------------------------------------------------------------------
# Runtime discovery source
# ---------------------------------------------------------------------------


@dataclass
class FreeProviderData:
    """Discovered data for one free provider."""

    spec: FreeProviderSpec
    models: list[FreeModelEntry]
    fetched_at: float = 0.0


class FreeProviderSource:
    """Discover free LLM providers and their models via public APIs.

    This mirrors the pattern of ``ModelsDevSource``: async fetch, local
    cache with TTL, fail-open on errors.
    """

    CACHE_PATH = Path.home() / ".mirai" / "free_providers_cache.json"
    CACHE_TTL = 3600  # 1 hour

    def __init__(self) -> None:
        self._data: dict[str, FreeProviderData] = {}

    async def fetch(self) -> dict[str, FreeProviderData]:
        """Fetch free provider catalogs, using cache when fresh."""
        cached = self._load_cache()
        if cached is not None:
            self._data = cached
            log.info("free_providers_cache_hit", providers=len(cached))
            return self._data

        data: dict[str, FreeProviderData] = {}

        # Fetch from public APIs concurrently
        fetchers: list[tuple[str, Any]] = [
            ("openrouter", self._fetch_openrouter()),
            ("sambanova", self._fetch_sambanova()),
        ]

        import asyncio

        results = await asyncio.gather(
            *[f for _, f in fetchers],
            return_exceptions=True,
        )

        for (provider_name, _), result in zip(fetchers, results, strict=True):
            spec = _SPECS_BY_NAME.get(provider_name)
            if spec is None:
                continue
            if isinstance(result, BaseException):
                log.warning(
                    "free_provider_fetch_failed",
                    provider=provider_name,
                    error=str(result),
                )
                data[provider_name] = FreeProviderData(spec=spec, models=[])
                continue
            models: list[FreeModelEntry] = result
            data[provider_name] = FreeProviderData(
                spec=spec,
                models=models,
                fetched_at=time.time(),
            )
            log.info(
                "free_provider_fetched",
                provider=provider_name,
                model_count=len(models),
            )

        # For providers without public APIs, add empty entries so the
        # registry knows they exist and can show them as "available to
        # configure".
        for spec in FREE_PROVIDER_SPECS:
            if spec.name not in data:
                data[spec.name] = FreeProviderData(spec=spec, models=[])

        self._data = data
        self._save_cache(data)
        return data

    # ------------------------------------------------------------------
    # Provider-specific fetchers (public APIs, no key needed)
    # ------------------------------------------------------------------

    async def _fetch_openrouter(self) -> list[FreeModelEntry]:
        """Fetch free models from OpenRouter's public /models endpoint.

        Reference: ``fetch_openrouter_models()`` in free-llm-api-resources.
        Filters for models with ``pricing.prompt == "0"`` and ``:free`` suffix.

        Extracts capability metadata from ``architecture``,
        ``supported_parameters``, and ``context_length`` fields.
        """
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()

        models_raw = resp.json().get("data", [])
        entries: list[FreeModelEntry] = []
        for m in models_raw:
            pricing = m.get("pricing", {})
            prompt_price = float(pricing.get("prompt", "1"))
            completion_price = float(pricing.get("completion", "1"))
            if prompt_price + completion_price != 0:
                continue
            if ":free" not in m.get("id", ""):
                continue

            model_id = m["id"]

            # Extract capabilities from API metadata
            arch = m.get("architecture", {})
            input_mods = arch.get("input_modalities", [])
            supported_params = m.get("supported_parameters", [])

            entries.append(
                FreeModelEntry(
                    id=model_id,
                    name=_get_model_name(model_id),
                    provider="openrouter",
                    limits={"requests/minute": 20, "requests/day": 50},
                    context_length=m.get("context_length"),
                    vision="image" in input_mods,
                    reasoning="reasoning" in supported_params,
                    supports_tool_use="tools" in supported_params,
                )
            )

        log.debug("openrouter_free_models", count=len(entries))
        return sorted(entries, key=lambda e: e.name)

    async def _fetch_sambanova(self) -> list[FreeModelEntry]:
        """Fetch models from SambaNova's public /api/pricing endpoint.

        Reference: ``fetch_samba_models()`` in free-llm-api-resources.
        """
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get("https://cloud.sambanova.ai/api/pricing")
            resp.raise_for_status()

        prices = resp.json().get("prices", [])
        entries: list[FreeModelEntry] = []
        for p in prices:
            model_id = p.get("model_id", "")
            model_name = p.get("model_name") or model_id
            if not model_id:
                continue
            entries.append(
                FreeModelEntry(
                    id=model_id,
                    name=model_name,
                    provider="sambanova",
                )
            )

        log.debug("sambanova_models", count=len(entries))
        return sorted(entries, key=lambda e: e.name)

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _load_cache(self) -> dict[str, FreeProviderData] | None:
        """Load cached data if fresh enough."""
        if not self.CACHE_PATH.exists():
            return None
        try:
            raw = json.loads(self.CACHE_PATH.read_text(encoding="utf-8"))
            ts = raw.get("timestamp", 0)
            if time.time() - ts > self.CACHE_TTL:
                log.debug(
                    "free_providers_cache_stale",
                    age_s=round(time.time() - ts, 1),
                )
                return None

            result: dict[str, FreeProviderData] = {}
            for pname, praw in raw.get("providers", {}).items():
                spec = _SPECS_BY_NAME.get(pname)
                if spec is None:
                    continue
                models = [
                    FreeModelEntry(
                        id=m["id"],
                        name=m["name"],
                        provider=m["provider"],
                        limits=m.get("limits"),
                        context_length=m.get("context_length"),
                        vision=m.get("vision", False),
                        reasoning=m.get("reasoning", False),
                        supports_tool_use=m.get("supports_tool_use", False),
                    )
                    for m in praw.get("models", [])
                ]
                result[pname] = FreeProviderData(
                    spec=spec,
                    models=models,
                    fetched_at=praw.get("fetched_at", ts),
                )
            return result

        except (json.JSONDecodeError, OSError, KeyError) as exc:
            log.warning("free_providers_cache_load_failed", error=str(exc))
            return None

    def _save_cache(self, data: dict[str, FreeProviderData]) -> None:
        """Persist data to local cache."""
        try:
            self.CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            payload: dict[str, Any] = {
                "timestamp": time.time(),
                "providers": {},
            }
            for pname, pdata in data.items():
                payload["providers"][pname] = {
                    "fetched_at": pdata.fetched_at,
                    "models": [
                        {
                            "id": m.id,
                            "name": m.name,
                            "provider": m.provider,
                            "limits": m.limits,
                            "context_length": m.context_length,
                            "vision": m.vision,
                            "reasoning": m.reasoning,
                            "supports_tool_use": m.supports_tool_use,
                        }
                        for m in pdata.models
                    ],
                }
            tmp = self.CACHE_PATH.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.rename(self.CACHE_PATH)
            log.debug("free_providers_cache_saved")
        except OSError as exc:
            log.warning("free_providers_cache_save_failed", error=str(exc))


# ---------------------------------------------------------------------------
# Provider health checks
# ---------------------------------------------------------------------------


@dataclass
class ProviderHealthStatus:
    """Result of a lightweight provider health probe."""

    name: str
    healthy: bool
    latency_ms: float | None = None
    last_checked: float = 0.0
    error: str | None = None


async def check_provider_health(
    specs: list[FreeProviderSpec] | None = None,
) -> dict[str, ProviderHealthStatus]:
    """Probe configured free providers with a lightweight GET /models call.

    Only checks providers whose API key is set in the environment.
    Returns a map of provider name → health status.
    """
    import os

    if specs is None:
        specs = FREE_PROVIDER_SPECS

    results: dict[str, ProviderHealthStatus] = {}
    tasks: list[tuple[FreeProviderSpec, Any]] = []

    for spec in specs:
        key = os.environ.get(spec.env_key)
        if not key:
            continue
        tasks.append((spec, _probe_provider(spec, key)))

    if not tasks:
        return results

    import asyncio

    probes = await asyncio.gather(
        *[t for _, t in tasks],
        return_exceptions=True,
    )

    for (spec, _), result in zip(tasks, probes, strict=True):
        if isinstance(result, BaseException):
            results[spec.name] = ProviderHealthStatus(
                name=spec.name,
                healthy=False,
                last_checked=time.time(),
                error=str(result),
            )
        else:
            results[spec.name] = result

    return results


async def _probe_provider(spec: FreeProviderSpec, api_key: str) -> ProviderHealthStatus:
    """Send GET /models to a single provider and measure latency."""
    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{spec.base_url}/models",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
        elapsed_ms = (time.monotonic() - start) * 1000
        return ProviderHealthStatus(
            name=spec.name,
            healthy=True,
            latency_ms=round(elapsed_ms, 1),
            last_checked=time.time(),
        )
    except Exception as exc:
        elapsed_ms = (time.monotonic() - start) * 1000
        return ProviderHealthStatus(
            name=spec.name,
            healthy=False,
            latency_ms=round(elapsed_ms, 1),
            last_checked=time.time(),
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Model name helper (from free-llm-api-resources/src/data.py)
# ---------------------------------------------------------------------------

# Subset of the 295-entry mapping from the upstream project.  We keep
# the most commonly used entries here; the full list can be synced from
# https://github.com/cheahjs/free-llm-api-resources/blob/main/src/data.py
MODEL_NAME_MAP: dict[str, str] = {
    "deepseek/deepseek-r1:free": "DeepSeek R1",
    "deepseek/deepseek-chat:free": "DeepSeek V3",
    "deepseek/deepseek-r1-distill-llama-70b:free": "DeepSeek R1 Distill Llama 70B",
    "deepseek/deepseek-r1-distill-qwen-32b:free": "DeepSeek R1 Distill Qwen 32B",
    "deepseek/deepseek-r1-distill-qwen-14b:free": "DeepSeek R1 Distill Qwen 14B",
    "deepseek/deepseek-r1-zero:free": "DeepSeek R1 Zero",
    "deepseek/deepseek-chat-v3-0324:free": "DeepSeek V3 0324",
    "deepseek/deepseek-v3-base:free": "DeepSeek V3 Base",
    "google/gemma-3-27b-it:free": "Gemma 3 27B Instruct",
    "google/gemma-3-12b-it:free": "Gemma 3 12B Instruct",
    "google/gemma-3-4b-it:free": "Gemma 3 4B Instruct",
    "google/gemma-3-1b-it:free": "Gemma 3 1B Instruct",
    "google/gemma-2-9b-it:free": "Gemma 2 9B Instruct",
    "meta-llama/llama-4-scout:free": "Llama 4 Scout",
    "meta-llama/llama-4-maverick:free": "Llama 4 Maverick",
    "meta-llama/llama-3.3-70b-instruct:free": "Llama 3.3 70B Instruct",
    "meta-llama/llama-3.2-90b-vision-instruct:free": "Llama 3.2 90B Vision Instruct",
    "meta-llama/llama-3.2-11b-vision-instruct:free": "Llama 3.2 11B Vision Instruct",
    "meta-llama/llama-3.2-3b-instruct:free": "Llama 3.2 3B Instruct",
    "meta-llama/llama-3.2-1b-instruct:free": "Llama 3.2 1B Instruct",
    "meta-llama/llama-3.1-70b-instruct:free": "Llama 3.1 70B Instruct",
    "meta-llama/llama-3.1-8b-instruct:free": "Llama 3.1 8B Instruct",
    "meta-llama/llama-3.1-405b-instruct:free": "Llama 3.1 405B Instruct",
    "mistralai/mistral-nemo:free": "Mistral Nemo",
    "mistralai/mistral-small-24b-instruct-2501:free": "Mistral Small 24B Instruct",
    "mistralai/mistral-small-3.1-24b-instruct:free": "Mistral Small 3.1 24B Instruct",
    "mistralai/mistral-7b-instruct:free": "Mistral 7B Instruct",
    "mistralai/pixtral-12b:free": "Pixtral 12B",
    "nvidia/llama-3.1-nemotron-ultra-253b-v1:free": "Nemotron Ultra 253B",
    "nvidia/llama-3.3-nemotron-super-49b-v1:free": "Nemotron Super 49B",
    "nvidia/llama-3.1-nemotron-nano-8b-v1:free": "Nemotron Nano 8B",
    "nvidia/llama-3.1-nemotron-70b-instruct:free": "Nemotron 70B Instruct",
    "qwen/qwq-32b:free": "Qwen QwQ 32B",
    "qwen/qwen-2.5-72b-instruct:free": "Qwen 2.5 72B Instruct",
    "qwen/qwen-2.5-7b-instruct:free": "Qwen 2.5 7B Instruct",
    "qwen/qwen-2.5-coder-32b-instruct:free": "Qwen 2.5 Coder 32B Instruct",
    "qwen/qwen2.5-vl-72b-instruct:free": "Qwen 2.5 VL 72B Instruct",
    "qwen/qwen2.5-vl-32b-instruct:free": "Qwen 2.5 VL 32B Instruct",
    "qwen/qwen2.5-vl-3b-instruct:free": "Qwen 2.5 VL 3B Instruct",
    "qwen/qwen-2.5-vl-7b-instruct:free": "Qwen 2.5 VL 7B Instruct",
    "qwen/qwq-32b-preview:free": "Qwen QwQ 32B Preview",
    "moonshotai/kimi-vl-a3b-thinking:free": "Kimi VL A3B Thinking",
    "moonshotai/moonlight-16b-a3b-instruct:free": "Moonlight 16B",
    "rekaai/reka-flash-3:free": "Reka Flash 3",
    "open-r1/olympiccoder-32b:free": "OlympicCoder 32B",
    "open-r1/olympiccoder-7b:free": "OlympicCoder 7B",
    "agentica-org/deepcoder-14b-preview:free": "DeepCoder 14B Preview",
    "arliai/qwq-32b-arliai-rpr-v1:free": "QwQ 32B ArliAI RpR v1",
    "shisa-ai/shisa-v2-llama3.3-70b:free": "Shisa V2 Llama 3.3 70B",
    "featherless/qwerky-72b:free": "Featherless Qwerky 72B",
    "cognitivecomputations/dolphin3.0-r1-mistral-24b:free": "Dolphin 3.0 R1 Mistral 24B",
    "cognitivecomputations/dolphin3.0-mistral-24b:free": "Dolphin 3.0 Mistral 24B",
    "allenai/molmo-7b-d:free": "Molmo 7B D",
    "nousresearch/deephermes-3-llama-3-8b-preview:free": "DeepHermes 3 Llama 3 8B",
    "bytedance-research/ui-tars-72b:free": "UI-TARS 72B",
    "sophosympatheia/rogue-rose-103b-v0.2:free": "Rogue Rose 103B v0.2",
    "gryphe/mythomax-l2-13b:free": "Mythomax L2 13B",
    "gryphe/mythomist-7b:free": "Mythomist 7B",
    "huggingfaceh4/zephyr-7b-beta:free": "Zephyr 7B Beta",
    "liquid/lfm-40b:free": "Liquid LFM 40B",
    "nousresearch/nous-capybara-7b:free": "Nous Capybara 7B",
    "openchat/openchat-7b:free": "OpenChat 7B",
    "thedrummer/unslopnemo-12b:free": "UnslopNemo 12B",
    "undi95/toppy-m-7b:free": "Toppy M 7B",
    "microsoft/phi-3-medium-128k-instruct:free": "Phi-3 Medium 128k Instruct",
    "microsoft/phi-3-mini-128k-instruct:free": "Phi-3 Mini 128k Instruct",
    # Groq models (common IDs)
    "llama-3.3-70b-versatile": "Llama 3.3 70B",
    "llama-3.1-8b-instant": "Llama 3.1 8B",
    "llama-3.3-70b-specdec": "Llama 3.3 70B (Speculative Decoding)",
    "llama3-70b-8192": "Llama 3 70B",
    "llama3-8b-8192": "Llama 3 8B",
    "gemma2-9b-it": "Gemma 2 9B Instruct",
    "mixtral-8x7b-32768": "Mixtral 8x7B",
    "qwen-qwq-32b": "Qwen QwQ 32B",
    "mistral-saba-24b": "Mistral Saba 24B",
    "deepseek-r1-distill-llama-70b": "DeepSeek R1 Distill Llama 70B",
    "meta-llama/llama-4-scout-17b-16e-instruct": "Llama 4 Scout Instruct",
    "compound-beta": "Groq Compound Beta",
    "compound-beta-mini": "Groq Compound Beta Mini",
    # Cerebras models
    "llama-3.3-70b": "Llama 3.3 70B",
    "llama3.1-8b": "Llama 3.1 8B",
    "qwen-3-32b": "Qwen 3 32B",
}


def _get_model_name(model_id: str) -> str:
    """Return a human-readable name for *model_id*, falling back to the raw id."""
    return MODEL_NAME_MAP.get(model_id, model_id)

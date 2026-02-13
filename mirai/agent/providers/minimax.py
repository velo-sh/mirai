"""MiniMax LLM provider — OpenAI-compatible.

MiniMax exposes a standard Chat Completions API. This provider adds:
  - Static MODEL_CATALOG (MiniMax has no /models endpoint)
  - Usage API via /v1/api/openplatform/coding_plan/remains

Usage in config.toml::

    [llm]
    provider = "minimax"
    api_key = "your-minimax-api-key"
    # base_url defaults to the China endpoint; override for international:
    # base_url = "https://api.minimax.io/v1"
"""

from __future__ import annotations

import httpx

from mirai.agent.providers.base import ModelInfo, UsageSnapshot
from mirai.agent.providers.openai import OpenAIProvider
from mirai.logging import get_logger

log = get_logger("mirai.providers.minimax")


class MiniMaxProvider(OpenAIProvider):
    """MiniMax provider via OpenAI-compatible Chat Completions API."""

    DEFAULT_BASE_URL = "https://api.minimaxi.com/v1"
    DEFAULT_MODEL = "MiniMax-M2.5"

    MODEL_CATALOG = [
        ModelInfo(
            id="MiniMax-M2.1",
            name="MiniMax M2.1",
            context_window=200_000,
            max_tokens=8192,
        ),
        ModelInfo(
            id="MiniMax-M2.1-lightning",
            name="MiniMax M2.1 Lightning",
            context_window=200_000,
            max_tokens=8192,
        ),
        ModelInfo(
            id="MiniMax-M2.5",
            name="MiniMax M2.5",
            context_window=200_000,
            max_tokens=8192,
            reasoning=True,
        ),
        ModelInfo(
            id="MiniMax-M2.5-Lightning",
            name="MiniMax M2.5 Lightning",
            context_window=200_000,
            max_tokens=8192,
            reasoning=True,
        ),
        ModelInfo(
            id="MiniMax-VL-01",
            name="MiniMax VL 01",
            context_window=200_000,
            max_tokens=8192,
            input_modalities=["text", "image"],
        ),
    ]

    @property
    def provider_name(self) -> str:
        return "minimax"

    async def get_usage(self) -> UsageSnapshot:
        """Query MiniMax usage API (coding_plan/remains).

        Endpoint learned from OpenClaw's provider-usage.fetch.minimax.ts.
        Returns used_percent and optional plan/reset info.
        """
        api_key = self.client.api_key
        base = str(self.client.base_url).rstrip("/")
        url = f"{base}/api/openplatform/coding_plan/remains"

        try:
            async with httpx.AsyncClient(timeout=10) as http:
                resp = await http.get(
                    url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                )
            if resp.status_code != 200:
                return UsageSnapshot(
                    provider="minimax",
                    error=f"HTTP {resp.status_code}",
                )

            data = resp.json()

            # Handle MiniMax base_resp error codes
            base_resp = data.get("base_resp", {})
            if base_resp.get("status_code", 0) != 0:
                return UsageSnapshot(
                    provider="minimax",
                    error=base_resp.get("status_msg", "API error"),
                )

            # Extract usage from data payload (auto-detect fields)
            payload = data.get("data", data)
            used = _pick_number(payload, ["used", "usage", "used_amount", "consumed"])
            total = _pick_number(payload, ["total", "total_amount", "limit", "quota"])
            remaining = _pick_number(payload, ["remain", "remaining", "left"])
            plan = _pick_string(payload, ["plan", "plan_name", "product", "tier"])

            used_pct = None
            if total and total > 0 and used is not None:
                used_pct = round((used / total) * 100, 1)
            elif total and total > 0 and remaining is not None:
                used_pct = round(((total - remaining) / total) * 100, 1)

            return UsageSnapshot(
                provider="minimax",
                used_percent=used_pct,
                plan=plan,
            )

        except Exception as exc:
            log.warning("minimax_usage_failed", error=str(exc))
            return UsageSnapshot(provider="minimax", error=str(exc))


# ---------------------------------------------------------------------------
# Helpers (adapted from OpenClaw's provider-usage.fetch.minimax.ts)
# ---------------------------------------------------------------------------
def _pick_number(record: dict, keys: list[str]) -> float | None:
    """Return the first numeric value found for any of the given keys."""
    for key in keys:
        val = record.get(key)
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            try:
                return float(val)
            except ValueError:
                continue
    return None


def _pick_string(record: dict, keys: list[str]) -> str | None:
    """Return the first non-empty string value found for any of the given keys."""
    for key in keys:
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None

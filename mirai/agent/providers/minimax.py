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

import re

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
            description="General-purpose chat model with 200K context",
            context_window=200_000,
            max_output_tokens=8192,
            supports_tool_use=True,
            supports_json_mode=True,
            input_price=1.1,
            output_price=4.4,
            knowledge_cutoff="2025-01",
        ),
        ModelInfo(
            id="MiniMax-M2.1-lightning",
            name="MiniMax M2.1 Lightning",
            description="Fast, low-latency variant of M2.1",
            context_window=200_000,
            max_output_tokens=8192,
            supports_tool_use=True,
            supports_json_mode=True,
            input_price=0.14,
            output_price=0.56,
            knowledge_cutoff="2025-01",
        ),
        ModelInfo(
            id="MiniMax-M2.5",
            name="MiniMax M2.5",
            description="Advanced reasoning model with extended thinking",
            context_window=200_000,
            max_output_tokens=8192,
            reasoning=True,
            supports_tool_use=True,
            supports_json_mode=True,
            input_price=1.1,
            output_price=4.4,
            knowledge_cutoff="2025-06",
        ),
        ModelInfo(
            id="MiniMax-M2.5-Lightning",
            name="MiniMax M2.5 Lightning",
            description="Fast reasoning model — best cost-performance ratio",
            context_window=200_000,
            max_output_tokens=8192,
            reasoning=True,
            supports_tool_use=True,
            supports_json_mode=True,
            input_price=0.14,
            output_price=0.56,
            knowledge_cutoff="2025-06",
        ),
        ModelInfo(
            id="MiniMax-VL-01",
            name="MiniMax VL 01",
            description="Vision-language model supporting text and image inputs",
            context_window=200_000,
            max_output_tokens=8192,
            supports_vision=True,
            supports_tool_use=True,
            input_modalities=["text", "image"],
            input_price=1.1,
            output_price=4.4,
            knowledge_cutoff="2025-01",
        ),
    ]

    @property
    def provider_name(self) -> str:
        return "minimax"

    @staticmethod
    def _to_provider_response(
        response: "Any",
        model_id: str | None = None,
    ) -> "ProviderResponse":
        """Override to strip MiniMax <think> reasoning tags from content.

        MiniMax-M2.5 embeds chain-of-thought reasoning inside
        ``<think>...</think>`` tags in ``message.content``.  We strip
        them so the reasoning never leaks into user-facing text.
        """
        from mirai.agent.models import ProviderResponse, TextBlock, ToolUseBlock
        import json
        from typing import Any

        choice = response.choices[0]
        message = choice.message
        content_blocks: list[TextBlock | ToolUseBlock] = []
        stop_reason = "end_turn"

        # Strip <think>...</think> from content
        if message.content:
            cleaned = re.sub(
                r"<think>.*?</think>", "", message.content, flags=re.DOTALL
            ).strip()
            if cleaned:
                content_blocks.append(TextBlock(text=cleaned))

        # Tool calls
        if message.tool_calls:
            stop_reason = "tool_use"
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                content_blocks.append(
                    ToolUseBlock(id=tc.id, name=tc.function.name, input=args)
                )

        if not content_blocks:
            content_blocks.append(TextBlock())

        if choice.finish_reason == "stop":
            stop_reason = "end_turn"
        elif choice.finish_reason == "length":
            stop_reason = "max_tokens"
        elif choice.finish_reason == "tool_calls":
            stop_reason = "tool_use"

        return ProviderResponse(
            content=content_blocks,
            stop_reason=stop_reason,
            model_id=model_id or response.model,
        )

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

            # Extract usage — MiniMax returns model_remains array
            remains = data.get("model_remains", [])
            used_pct = None
            plan = None
            reset_at = None

            if remains:
                # Aggregate across all model entries.
                # NB: current_interval_usage_count is the REMAINING count
                #     (the endpoint is "remains"), not the consumed count.
                total_remaining = 0
                total_limit = 0
                for entry in remains:
                    total_remaining += entry.get("current_interval_usage_count", 0)
                    total_limit += entry.get("current_interval_total_count", 0)
                    # Use end_time from the first entry as reset_at
                    if reset_at is None and entry.get("end_time"):
                        from datetime import datetime, timezone
                        ts = entry["end_time"] / 1000  # ms → s
                        reset_at = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                    plan = plan or entry.get("model_name")

                if total_limit > 0:
                    consumed = total_limit - total_remaining
                    used_pct = round((consumed / total_limit) * 100, 1)
            else:
                # Fallback: generic field detection for forward compatibility
                payload = data.get("data", data)
                used = _pick_number(payload, ["used", "usage", "used_amount", "consumed"])
                total = _pick_number(payload, ["total", "total_amount", "limit", "quota"])
                remaining = _pick_number(payload, ["remain", "remaining", "left"])
                plan = _pick_string(payload, ["plan", "plan_name", "product", "tier"])

                if total and total > 0 and used is not None:
                    used_pct = round((used / total) * 100, 1)
                elif total and total > 0 and remaining is not None:
                    used_pct = round(((total - remaining) / total) * 100, 1)

            return UsageSnapshot(
                provider="minimax",
                used_percent=used_pct,
                plan=plan,
                reset_at=reset_at,
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

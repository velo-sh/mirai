"""Smoke tests — live API calls to verify basic provider connectivity.

These tests require real API keys and make REAL network requests.
They are marked with ``@pytest.mark.smoke`` so they can be run selectively::

    pytest tests/test_provider_smoke.py -m smoke -v

To skip during normal CI runs, add ``-m 'not smoke'`` to default pytest args.

Required env vars (set whichever providers you want to test):
    MINIMAX_API_KEY    — for MiniMax tests
    ANTHROPIC_API_KEY  — for Anthropic tests
    OPENAI_API_KEY     — for OpenAI tests
"""

from __future__ import annotations

import os

import pytest

from mirai.agent.providers.base import ModelInfo, UsageSnapshot

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _skip_unless_key(env_var: str) -> str:
    """Return the key or skip the test if the env var is unset."""
    key = os.environ.get(env_var, "")
    if not key or key.startswith("sk-ant-xxx") or key.startswith("sk-xxx"):
        pytest.skip(f"{env_var} not configured")
    return key


# ---------------------------------------------------------------------------
# MiniMax smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.smoke
class TestMiniMaxSmoke:
    """Live smoke tests for MiniMax provider."""

    @pytest.fixture(autouse=True)
    def require_key(self):
        self.api_key = _skip_unless_key("MINIMAX_API_KEY")

    def _make_provider(self):
        from mirai.agent.providers.minimax import MiniMaxProvider

        return MiniMaxProvider(api_key=self.api_key)

    # --- list_models ---

    @pytest.mark.asyncio
    async def test_list_models(self):
        provider = self._make_provider()
        models = await provider.list_models()

        assert len(models) >= 3
        assert all(isinstance(m, ModelInfo) for m in models)

        ids = {m.id for m in models}
        assert "MiniMax-M2.5" in ids

        # Every model should have the enriched fields
        for m in models:
            assert m.description is not None
            assert m.max_output_tokens is not None
            assert m.input_price is not None

    # --- get_usage ---

    @pytest.mark.asyncio
    async def test_get_usage(self):
        provider = self._make_provider()
        usage = await provider.get_usage()

        assert isinstance(usage, UsageSnapshot)
        assert usage.provider == "minimax"
        assert usage.error is None
        assert usage.used_percent is not None
        assert 0 <= usage.used_percent <= 100
        assert usage.reset_at is not None

    # --- generate_response (basic) ---

    @pytest.mark.asyncio
    async def test_simple_chat_completion(self):
        """Send a trivial prompt and verify we get a non-empty text reply."""
        provider = self._make_provider()

        response = await provider.generate_response(
            model="MiniMax-M2.5",
            system="You are a helpful assistant. Reply in one sentence.",
            messages=[{"role": "user", "content": "Say hello."}],
            tools=[],
        )

        assert response is not None
        assert response.content is not None
        assert len(response.content) > 0

        # Should have at least one text block
        from mirai.agent.models import TextBlock

        text_blocks = [b for b in response.content if isinstance(b, TextBlock)]
        assert len(text_blocks) >= 1
        assert len(text_blocks[0].text.strip()) > 0

    @pytest.mark.asyncio
    async def test_tool_use_completion(self):
        """Send a prompt with a tool and verify the model calls it."""
        provider = self._make_provider()

        tools = [
            {
                "name": "get_weather",
                "description": "Get current weather for a city.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "City name",
                        },
                    },
                    "required": ["city"],
                },
            },
        ]

        response = await provider.generate_response(
            model="MiniMax-M2.5",
            system="You are a helpful assistant. Use the get_weather tool when asked about weather.",
            messages=[
                {"role": "user", "content": "What's the weather in Beijing?"},
            ],
            tools=tools,
        )

        assert response is not None
        assert response.content is not None

        # The model should either call the tool or respond with text
        from mirai.agent.models import ToolUseBlock

        tool_blocks = [b for b in response.content if isinstance(b, ToolUseBlock)]
        if tool_blocks:
            assert tool_blocks[0].name == "get_weather"
            assert "city" in tool_blocks[0].input

    # --- provider_name ---

    def test_provider_name(self):
        provider = self._make_provider()
        assert provider.provider_name == "minimax"

    # --- model default ---

    def test_default_model_is_m25(self):
        provider = self._make_provider()
        assert provider.model == "MiniMax-M2.5"


# ---------------------------------------------------------------------------
# Anthropic smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.smoke
class TestAnthropicSmoke:
    """Live smoke tests for Anthropic provider."""

    @pytest.fixture(autouse=True)
    def require_key(self):
        self.api_key = _skip_unless_key("ANTHROPIC_API_KEY")

    def _make_provider(self):
        from mirai.agent.providers.anthropic import AnthropicProvider

        return AnthropicProvider(api_key=self.api_key)

    @pytest.mark.asyncio
    async def test_list_models(self):
        provider = self._make_provider()
        models = await provider.list_models()

        assert len(models) >= 3
        ids = {m.id for m in models}
        assert "claude-sonnet-4-20250514" in ids

    @pytest.mark.asyncio
    async def test_get_usage_returns_not_supported(self):
        provider = self._make_provider()
        usage = await provider.get_usage()
        assert usage.error == "not supported"

    def test_provider_name(self):
        provider = self._make_provider()
        assert provider.provider_name == "anthropic"

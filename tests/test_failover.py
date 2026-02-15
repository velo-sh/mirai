from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirai.agent.providers import AntigravityProvider


@pytest.mark.asyncio
async def test_antigravity_provider_failover():
    """Test that AntigravityProvider fails over to Gemini when Claude is exhausted."""

    # Mock credentials
    creds = {"access": "fake_access", "refresh": "fake_refresh", "expires": 9999999999, "project_id": "test-project"}

    provider = AntigravityProvider(credentials=creds, model="claude-sonnet-4-20250514")

    # Mock fetch_usage to return Claude as 100% used and Gemini as 0%
    mock_usage = {
        "models": [
            {"id": "claude-sonnet-4-5", "used_pct": 100.0, "reset_time": "1h"},
            {"id": "claude-opus-4-5-thinking", "used_pct": 100.0, "reset_time": "1h"},
            {"id": "claude-opus-4-6-thinking", "used_pct": 100.0, "reset_time": "1h"},
            {"id": "claude-sonnet-4-5-thinking", "used_pct": 100.0, "reset_time": "1h"},
            {"id": "gemini-3-pro-high", "used_pct": 0.0, "reset_time": None},
            {"id": "gemini-3-flash", "used_pct": 0.0, "reset_time": None},
        ]
    }

    with patch("mirai.agent.providers.quota.fetch_usage", AsyncMock(return_value=mock_usage)):
        # Mock the HTTP post call
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = 'data: {"response": {"candidates": [{"content": {"parts": [{"text": "Hello from fallback"}]}, "finishReason": "STOP"}]}}\n'

        provider._http.post = AsyncMock(return_value=mock_response)

        # This should trigger failover from claude-sonnet-4-20250514 -> claude-sonnet-4-5 (100%)
        # -> all Claude models exhausted -> gemini-3-pro-high (0%)
        # Note: MODEL_MAP maps "claude-sonnet-4-20250514" to "claude-sonnet-4-5"
        resp = await provider.generate_response(
            model="claude-sonnet-4-20250514", system="sys", messages=[{"role": "user", "content": "hi"}], tools=[]
        )

        # Verify result
        assert resp.text() == "Hello from fallback"
        assert resp.model_id == "gemini-3-pro-high"

        # Verify the correct model was sent in the request
        call_args = provider._http.post.call_args
        import orjson

        body = orjson.loads(call_args[1]["content"])
        assert body["model"] == "gemini-3-pro-high"


@pytest.mark.asyncio
async def test_antigravity_provider_no_failover_when_available():
    """Test that AntigravityProvider does NOT fail over when primary model is available."""

    creds = {"access": "fake_access", "refresh": "fake_refresh", "expires": 9999999999, "project_id": "test-project"}

    provider = AntigravityProvider(credentials=creds, model="claude-sonnet-4-20250514")

    mock_usage = {
        "models": [
            {"id": "claude-sonnet-4-5", "used_pct": 50.0, "reset_time": "1h"},
            {"id": "gemini-3-pro-high", "used_pct": 0.0, "reset_time": None},
        ]
    }

    with patch("mirai.agent.providers.quota.fetch_usage", AsyncMock(return_value=mock_usage)):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = 'data: {"response": {"candidates": [{"content": {"parts": [{"text": "Hello"}]}, "finishReason": "STOP"}]}}\n'
        provider._http.post = AsyncMock(return_value=mock_response)

        await provider.generate_response(
            model="claude-sonnet-4-20250514", system="sys", messages=[{"role": "user", "content": "hi"}], tools=[]
        )

        call_args = provider._http.post.call_args
        import orjson

        body = orjson.loads(call_args[1]["content"])
        assert body["model"] == "claude-sonnet-4-5"

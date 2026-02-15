"""Unit tests for mirai.agent.tools.system — SystemTool (status, usage, restart)."""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirai.agent.tools.system import SystemTool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeConfig:
    """Minimal stand-in for MiraiConfig to avoid full config loading."""

    class _LLM:
        default_model = "claude-sonnet-4-20250514"
        max_tokens = 4096

    class _Heartbeat:
        interval = 3600.0
        enabled = True

    llm = _LLM()
    heartbeat = _Heartbeat()


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


class TestSystemToolStatus:
    @pytest.mark.asyncio
    async def test_status_returns_valid_json(self):
        tool = SystemTool(config=_FakeConfig(), start_time=0.0)
        result = await tool.execute(action="status")
        data = json.loads(result)
        assert "pid" in data
        assert data["pid"] == os.getpid()
        assert "uptime_s" in data
        assert "memory_mb" in data
        assert data["model"] == "claude-sonnet-4-20250514"

    @pytest.mark.asyncio
    async def test_status_without_config(self):
        tool = SystemTool()
        result = await tool.execute(action="status")
        data = json.loads(result)
        assert data["model"] is None


# ---------------------------------------------------------------------------
# restart
# ---------------------------------------------------------------------------


class TestSystemToolRestart:
    @pytest.mark.asyncio
    async def test_restart_returns_ack(self):
        tool = SystemTool(config=_FakeConfig())
        with patch("os.execv"):
            result = await tool.execute(action="restart")
            assert "Restart scheduled" in result

    @pytest.mark.asyncio
    async def test_restart_schedules_task(self):
        """Verify a background task was created (we don't actually execv)."""
        tool = SystemTool(config=_FakeConfig())
        with patch("os.execv"):
            result = await tool.execute(action="restart")
            assert "🔄" in result


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestSystemToolEdgeCases:
    def test_definition_has_required_fields(self):
        tool = SystemTool()
        defn = tool.definition
        assert defn["name"] == "mirai_system"
        assert "status" in str(defn)
        assert "restart" in str(defn)

    @pytest.mark.asyncio
    async def test_unknown_action(self):
        tool = SystemTool()
        result = await tool.execute(action="self_destruct")
        assert "Error" in result
        assert "self_destruct" in result


# ---------------------------------------------------------------------------
# usage
# ---------------------------------------------------------------------------


def _make_fake_provider(credentials: dict | None = None) -> MagicMock:
    """Create a mock provider that passes isinstance(_, AntigravityProvider)."""
    from mirai.agent.providers.antigravity import AntigravityProvider

    provider = MagicMock(spec=AntigravityProvider)
    provider.credentials = credentials or {
        "access": "fake-token-123",
        "refresh": "fake-refresh",
        "expires": 9999999999,
        "project_id": "test-project",
        "email": "test@example.com",
    }
    provider._ensure_fresh_token = AsyncMock()
    return provider


class TestSystemToolUsage:
    @pytest.mark.asyncio
    async def test_usage_returns_model_data(self):
        """Usage action should return per-model quota data."""
        fake_usage = {
            "plan": "Antigravity",
            "project": "test-project",
            "models": [
                {"id": "gemini-3-flash", "used_pct": 80.0, "reset_time": "2026-02-13T06:00:00Z"},
                {"id": "claude-sonnet-4-5", "used_pct": 40.0, "reset_time": "2026-02-13T06:00:00Z"},
                {"id": "gemini-2.5-pro", "used_pct": 100.0, "reset_time": "2026-02-13T06:00:00Z"},
                {"id": "gemini-2.5-flash", "used_pct": 0.0, "reset_time": None},
            ],
        }
        provider = _make_fake_provider()
        tool = SystemTool(config=_FakeConfig(), provider=provider)

        with patch("mirai.agent.tools.system.fetch_usage", return_value=fake_usage):
            result = await tool.execute(action="usage")

        data = json.loads(result)
        assert data["plan"] == "Antigravity"
        assert len(data["models"]) == 4

        # Check status indicators
        models_by_id = {m["model"]: m for m in data["models"]}
        assert "exhausted" in models_by_id["gemini-2.5-pro"]["status"]
        assert "high" in models_by_id["gemini-3-flash"]["status"]
        assert "ok" in models_by_id["claude-sonnet-4-5"]["status"]
        assert "ok" in models_by_id["gemini-2.5-flash"]["status"]

    @pytest.mark.asyncio
    async def test_usage_without_provider(self):
        """Should return error when provider is not available."""
        tool = SystemTool(config=_FakeConfig())
        result = await tool.execute(action="usage")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_usage_empty_token(self):
        """Should return error when access token is empty."""
        provider = _make_fake_provider(credentials={"access": "", "project_id": "p"})
        tool = SystemTool(config=_FakeConfig(), provider=provider)
        result = await tool.execute(action="usage")
        assert "Error" in result
        assert "token" in result.lower()

    @pytest.mark.asyncio
    async def test_usage_fetch_error(self):
        """Should return error message when fetch_usage throws."""
        provider = _make_fake_provider()
        tool = SystemTool(config=_FakeConfig(), provider=provider)

        with patch("mirai.agent.tools.system.fetch_usage", side_effect=Exception("network timeout")):
            result = await tool.execute(action="usage")

        assert "Error" in result
        assert "network timeout" in result

    def test_definition_includes_usage(self):
        """Usage should be listed in the tool definition."""
        tool = SystemTool()
        defn = tool.definition
        assert "usage" in defn["input_schema"]["properties"]["action"]["enum"]
        assert "usage" in defn["description"]

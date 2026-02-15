"""Unit tests for mirai.agent.tools.system — SystemTool self-evolution capabilities."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from mirai.agent.providers.antigravity import AntigravityProvider
from mirai.agent.tools.system import SystemTool, _serialize_toml

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
# patch_config
# ---------------------------------------------------------------------------


class TestSystemToolPatchConfig:
    @pytest.mark.asyncio
    async def test_patch_whitelisted_key(self, tmp_path: Path):
        config_file = tmp_path / "config.toml"
        with patch("mirai.agent.tools.system._CONFIG_PATH", config_file):
            tool = SystemTool(config=_FakeConfig())
            result = await tool.execute(
                action="patch_config",
                patch={"heartbeat": {"interval": 1800}},
            )
            assert "✅" in result
            assert "heartbeat.interval" in result
            # Verify the file was written
            content = config_file.read_text()
            assert "1800" in content

    @pytest.mark.asyncio
    async def test_patch_rejected_key(self):
        tool = SystemTool(config=_FakeConfig())
        result = await tool.execute(
            action="patch_config",
            patch={"database": {"sqlite_url": "sqlite:///hacked.db"}},
        )
        assert "Error" in result
        assert "database.sqlite_url" in result

    @pytest.mark.asyncio
    async def test_patch_empty_patch(self):
        tool = SystemTool(config=_FakeConfig())
        result = await tool.execute(action="patch_config", patch={})
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_patch_merges_with_existing(self, tmp_path: Path):
        config_file = tmp_path / "config.toml"
        config_file.write_text("[heartbeat]\ninterval = 3600.0\nenabled = true\n")
        with patch("mirai.agent.tools.system._CONFIG_PATH", config_file):
            tool = SystemTool(config=_FakeConfig())
            result = await tool.execute(
                action="patch_config",
                patch={"heartbeat": {"interval": 900}},
            )
            assert "✅" in result
            content = config_file.read_text()
            assert "900" in content
            # enabled should still be present
            assert "enabled" in content


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
        assert "patch_config" in str(defn)
        assert "restart" in str(defn)

    @pytest.mark.asyncio
    async def test_unknown_action(self):
        tool = SystemTool()
        result = await tool.execute(action="self_destruct")
        assert "Error" in result
        assert "self_destruct" in result


# ---------------------------------------------------------------------------
# _serialize_toml
# ---------------------------------------------------------------------------


class TestSerializeToml:
    def test_basic_sections(self):
        data = {
            "heartbeat": {"interval": 1800, "enabled": True},
            "llm": {"default_model": "gemini-3-flash"},
        }
        result = _serialize_toml(data)
        assert "[heartbeat]" in result
        assert "interval = 1800" in result
        assert "enabled = true" in result
        assert "[llm]" in result
        assert 'default_model = "gemini-3-flash"' in result

    def test_float_values(self):
        data = {"heartbeat": {"interval": 3600.0}}
        result = _serialize_toml(data)
        assert "interval = 3600.0" in result

    def test_bool_false(self):
        data = {"heartbeat": {"enabled": False}}
        result = _serialize_toml(data)
        assert "enabled = false" in result


# ---------------------------------------------------------------------------
# usage
# ---------------------------------------------------------------------------


def _make_fake_provider(credentials: dict | None = None) -> AntigravityProvider:
    """Create a mock AntigravityProvider with fake credentials.

    Uses MagicMock(spec=...) so isinstance() checks pass and the mock
    automatically tracks protocol changes in AntigravityProvider.
    """
    from unittest.mock import MagicMock

    provider = MagicMock(spec=AntigravityProvider)
    provider.credentials = credentials or {
        "access": "fake-token-123",
        "refresh": "fake-refresh",
        "expires": 9999999999,
        "project_id": "test-project",
        "email": "test@example.com",
    }
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

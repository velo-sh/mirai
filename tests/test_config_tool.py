"""Unit tests for mirai.agent.tools.config_tool — ConfigTool (patch_config, patch_runtime)."""

from pathlib import Path
from unittest.mock import patch

import pytest

from mirai.agent.tools.config_tool import ConfigTool, _serialize_toml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeConfig:
    """Minimal stand-in for MiraiConfig."""

    class _LLM:
        default_model = "claude-sonnet-4-20250514"
        max_tokens = 4096

    class _Heartbeat:
        interval = 3600.0
        enabled = True

    llm = _LLM()
    heartbeat = _Heartbeat()


# ---------------------------------------------------------------------------
# patch_config
# ---------------------------------------------------------------------------


class TestConfigToolPatchConfig:
    @pytest.mark.asyncio
    async def test_patch_whitelisted_key(self, tmp_path: Path):
        config_file = tmp_path / "config.toml"
        with patch("mirai.agent.tools.config_tool._CONFIG_PATH", config_file):
            tool = ConfigTool()
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
        tool = ConfigTool()
        result = await tool.execute(
            action="patch_config",
            patch={"database": {"sqlite_url": "sqlite:///hacked.db"}},
        )
        assert "Error" in result
        assert "database.sqlite_url" in result

    @pytest.mark.asyncio
    async def test_patch_empty_patch(self):
        tool = ConfigTool()
        result = await tool.execute(action="patch_config", patch={})
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_patch_merges_with_existing(self, tmp_path: Path):
        config_file = tmp_path / "config.toml"
        config_file.write_text("[heartbeat]\ninterval = 3600.0\nenabled = true\n")
        with patch("mirai.agent.tools.config_tool._CONFIG_PATH", config_file):
            tool = ConfigTool()
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

"""Tests for mirai.agent.prompt — dynamic tool awareness in system prompt."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from mirai.agent.prompt import _build_tools_section, build_system_prompt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool(name: str, description: str) -> MagicMock:
    """Create a mock tool with a definition property."""
    tool = MagicMock()
    tool.definition = {"name": name, "description": description}
    return tool


def _make_tool_no_desc(name: str) -> MagicMock:
    """Create a mock tool with an empty description."""
    tool = MagicMock()
    tool.definition = {"name": name}
    return tool




def _mock_deps() -> dict[str, Any]:
    """Return mock dependencies for build_system_prompt."""
    embedder = AsyncMock()
    embedder.get_embeddings = AsyncMock(return_value=[0.1, 0.2, 0.3])

    l2_storage = AsyncMock()
    l2_storage.search = AsyncMock(return_value=[])

    l3_storage = AsyncMock()
    l3_storage.get_traces_by_ids = AsyncMock(return_value=[])

    provider = MagicMock()
    provider.provider_name = "test-provider"
    provider.model = "test-model"

    return {
        "collaborator_id": "test-collab",
        "soul_content": "I am a test agent.",
        "base_system_prompt": "Base context here.",
        "provider": provider,
        "embedder": embedder,
        "l2_storage": l2_storage,
        "l3_storage": l3_storage,
        "msg_text": "hello",
    }


# ===========================================================================
# _build_tools_section tests
# ===========================================================================

class TestBuildToolsSection:
    """Unit tests for the _build_tools_section helper."""

    def test_empty_tools_returns_empty_string(self):
        assert _build_tools_section({}) == ""

    def test_single_tool_with_description(self):
        tools = {"im_tool": _make_tool("im_tool", "Send messages via Feishu")}
        result = _build_tools_section(tools)

        assert "## Available Tools" in result
        assert "- **im_tool**: Send messages via Feishu" in result

    def test_multiple_tools(self):
        tools = {
            "echo": _make_tool("echo", "Echo test"),
            "shell_tool": _make_tool("shell_tool", "Run shell commands"),
            "im_tool": _make_tool("im_tool", "Send messages"),
        }
        result = _build_tools_section(tools)

        assert "## Available Tools" in result
        assert "- **echo**: Echo test" in result
        assert "- **shell_tool**: Run shell commands" in result
        assert "- **im_tool**: Send messages" in result

    def test_tool_without_description_uses_name_only(self):
        tools = {"mystery_tool": _make_tool_no_desc("mystery_tool")}
        result = _build_tools_section(tools)

        assert "- **mystery_tool**" in result
        # Should NOT have a colon after the name (no description)
        assert "- **mystery_tool**:" not in result

    def test_preserves_tool_order(self):
        """Tools should appear in insertion order (dict order)."""
        tools = {
            "aaa": _make_tool("aaa", "First"),
            "zzz": _make_tool("zzz", "Second"),
            "mmm": _make_tool("mmm", "Third"),
        }
        result = _build_tools_section(tools)
        lines = result.split("\n")

        tool_lines = [l for l in lines if l.startswith("- **")]
        assert len(tool_lines) == 3
        assert "aaa" in tool_lines[0]
        assert "zzz" in tool_lines[1]
        assert "mmm" in tool_lines[2]

    def test_multiline_description(self):
        """Long descriptions should be included as-is."""
        long_desc = (
            "Send a message to the user via Feishu. "
            "Use this when you need to proactively communicate with the user, "
            "for example when woken by a cron job."
        )
        tools = {"im_tool": _make_tool("im_tool", long_desc)}
        result = _build_tools_section(tools)

        assert long_desc in result


# ===========================================================================
# build_system_prompt integration tests
# ===========================================================================

class TestBuildSystemPromptWithTools:
    """Integration tests for tool awareness in the full system prompt."""

    @pytest.mark.asyncio
    async def test_tools_section_appears_in_prompt(self):
        deps = _mock_deps()
        tools = {
            "echo": _make_tool("echo", "Echo test"),
            "im_tool": _make_tool("im_tool", "Send messages via Feishu"),
        }

        prompt = await build_system_prompt(**deps, tools=tools)

        assert "## Available Tools" in prompt
        assert "- **echo**: Echo test" in prompt
        assert "- **im_tool**: Send messages via Feishu" in prompt

    @pytest.mark.asyncio
    async def test_no_tools_means_no_tools_section(self):
        deps = _mock_deps()

        prompt = await build_system_prompt(**deps, tools=None)

        assert "## Available Tools" not in prompt

    @pytest.mark.asyncio
    async def test_empty_tools_means_no_tools_section(self):
        deps = _mock_deps()

        prompt = await build_system_prompt(**deps, tools={})

        assert "## Available Tools" not in prompt

    @pytest.mark.asyncio
    async def test_tools_section_between_context_and_runtime(self):
        """Verify sandwich order: IDENTITY → CONTEXT → tools → RUNTIME."""
        deps = _mock_deps()
        tools = {"echo": _make_tool("echo", "Test tool")}

        prompt = await build_system_prompt(**deps, tools=tools)

        idx_context = prompt.index("# CONTEXT")
        idx_tools = prompt.index("## Available Tools")
        idx_runtime = prompt.index("# RUNTIME INFO")

        assert idx_context < idx_tools < idx_runtime

    @pytest.mark.asyncio
    async def test_prompt_contains_all_standard_sections(self):
        """Even with tools, all standard sections should be present."""
        deps = _mock_deps()
        tools = {"echo": _make_tool("echo", "Test")}

        prompt = await build_system_prompt(**deps, tools=tools)

        assert "# IDENTITY" in prompt
        assert "# CONTEXT" in prompt
        assert "## Available Tools" in prompt
        assert "# RUNTIME INFO" in prompt
        assert "# IDENTITY REINFORCEMENT" in prompt
        assert "# TOOL USE RULES" in prompt

    @pytest.mark.asyncio
    async def test_backward_compatible_without_tools_param(self):
        """Calling without tools kwarg should still work (default None)."""
        deps = _mock_deps()

        prompt = await build_system_prompt(**deps)

        assert "# IDENTITY" in prompt
        assert "## Available Tools" not in prompt

    @pytest.mark.asyncio
    async def test_real_tool_definitions(self):
        """Smoke test with actual tool classes to verify definition reading."""
        from mirai.agent.tools.im import IMTool
        from mirai.agent.tools.echo import EchoTool

        im = IMTool()
        echo = EchoTool()
        tools = {
            im.definition["name"]: im,
            echo.definition["name"]: echo,
        }
        deps = _mock_deps()

        prompt = await build_system_prompt(**deps, tools=tools)

        assert "## Available Tools" in prompt
        # im_tool description should contain "Feishu"
        assert "Feishu" in prompt
        assert "im_tool" in prompt
        assert "echo" in prompt

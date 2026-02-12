"""Unit tests for mirai.agent.models — Pydantic v2 models and helpers."""

import pytest
from pydantic import ValidationError

from mirai.agent.models import (
    ContentBlock,
    LLMProvider,
    Message,
    ProviderResponse,
    TextBlock,
    ToolDefinition,
    ToolUseBlock,
    orjson_dumps,
    orjson_loads,
)

# ---------------------------------------------------------------------------
# TextBlock
# ---------------------------------------------------------------------------


class TestTextBlock:
    def test_default_values(self):
        block = TextBlock()
        assert block.type == "text"
        assert block.text == ""

    def test_custom_text(self):
        block = TextBlock(text="hello world")
        assert block.text == "hello world"
        assert block.type == "text"

    def test_frozen(self):
        block = TextBlock(text="immutable")
        with pytest.raises(ValidationError):
            block.text = "changed"  # type: ignore[misc]

    def test_serialization(self):
        block = TextBlock(text="test")
        data = block.model_dump()
        assert data == {"type": "text", "text": "test"}


# ---------------------------------------------------------------------------
# ToolUseBlock
# ---------------------------------------------------------------------------


class TestToolUseBlock:
    def test_required_fields(self):
        block = ToolUseBlock(id="call_1", name="echo", input={"message": "hi"})
        assert block.type == "tool_use"
        assert block.id == "call_1"
        assert block.name == "echo"
        assert block.input == {"message": "hi"}

    def test_missing_required_raises(self):
        with pytest.raises(ValidationError):
            ToolUseBlock(name="echo", input={})  # type: ignore[call-arg]  # missing id

    def test_frozen(self):
        block = ToolUseBlock(id="x", name="y", input={})
        with pytest.raises(ValidationError):
            block.name = "z"  # type: ignore[misc]

    def test_serialization_roundtrip(self):
        block = ToolUseBlock(id="c1", name="shell", input={"cmd": "ls"})
        data = block.model_dump()
        restored = ToolUseBlock(**data)
        assert restored == block


# ---------------------------------------------------------------------------
# ProviderResponse
# ---------------------------------------------------------------------------


class TestProviderResponse:
    def test_text_only(self):
        resp = ProviderResponse(content=[TextBlock(text="Hello")])
        assert resp.text() == "Hello"
        assert resp.stop_reason == "end_turn"
        assert resp.tool_use_blocks() == []

    def test_tool_use_only(self):
        tool = ToolUseBlock(id="t1", name="echo", input={"message": "x"})
        resp = ProviderResponse(content=[tool], stop_reason="tool_use")
        assert resp.text() == ""
        assert resp.tool_use_blocks() == [tool]

    def test_mixed_content(self):
        text = TextBlock(text="Let me help. ")
        tool = ToolUseBlock(id="t1", name="shell", input={"cmd": "ls"})
        resp = ProviderResponse(content=[text, tool], stop_reason="tool_use")
        assert resp.text() == "Let me help. "
        assert len(resp.tool_use_blocks()) == 1

    def test_multiple_text_blocks_concatenated(self):
        resp = ProviderResponse(content=[TextBlock(text="a"), TextBlock(text="b"), TextBlock(text="c")])
        assert resp.text() == "abc"

    def test_empty_content(self):
        resp = ProviderResponse(content=[])
        assert resp.text() == ""
        assert resp.tool_use_blocks() == []

    def test_frozen(self):
        resp = ProviderResponse(content=[TextBlock(text="x")])
        with pytest.raises(ValidationError):
            resp.stop_reason = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Message & ToolDefinition
# ---------------------------------------------------------------------------


class TestMessage:
    def test_string_content(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_list_content(self):
        msg = Message(role="assistant", content=[{"type": "text", "text": "Hi"}])
        assert isinstance(msg.content, list)


class TestToolDefinition:
    def test_valid(self):
        td = ToolDefinition(
            name="echo",
            description="Echo tool",
            input_schema={"type": "object", "properties": {}},
        )
        assert td.name == "echo"


# ---------------------------------------------------------------------------
# LLMProvider Protocol
# ---------------------------------------------------------------------------


class TestLLMProviderProtocol:
    def test_mock_provider_satisfies_protocol(self):
        from mirai.agent.providers import MockProvider

        provider = MockProvider()
        assert isinstance(provider, LLMProvider)

    def test_non_provider_fails_protocol(self):
        class NotAProvider:
            pass

        assert not isinstance(NotAProvider(), LLMProvider)


# ---------------------------------------------------------------------------
# orjson helpers
# ---------------------------------------------------------------------------


class TestOrjsonHelpers:
    def test_dumps_returns_str(self):
        result = orjson_dumps({"key": "value"})
        assert isinstance(result, str)
        assert '"key"' in result

    def test_loads_from_str(self):
        data = orjson_loads('{"a": 1}')
        assert data == {"a": 1}

    def test_loads_from_bytes(self):
        data = orjson_loads(b'{"a": 1}')
        assert data == {"a": 1}

    def test_roundtrip(self):
        original = {"nested": {"list": [1, 2, 3]}, "flag": True}
        json_str = orjson_dumps(original)
        restored = orjson_loads(json_str)
        assert restored == original


# ---------------------------------------------------------------------------
# ContentBlock union type
# ---------------------------------------------------------------------------


class TestContentBlockUnion:
    def test_text_is_contentblock(self):
        block: ContentBlock = TextBlock(text="x")
        assert isinstance(block, TextBlock)

    def test_tool_is_contentblock(self):
        block: ContentBlock = ToolUseBlock(id="1", name="t", input={})
        assert isinstance(block, ToolUseBlock)

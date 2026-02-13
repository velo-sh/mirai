"""Tests for message conversion from OpenAI format to Google Generative AI format.

The message_converter now converts from the internal canonical format (OpenAI
Chat Completions) to Google Generative AI format for Cloud Code Assist.

Key conversion rules:
  - OpenAI role="tool" with tool_call_id → Gemini functionResponse
  - OpenAI assistant.tool_calls → Gemini functionCall parts
  - OpenAI role="system" filtered out (handled separately)
  - Empty text content is dropped to satisfy Claude proxy requirements
"""

import json

import pytest

from mirai.agent.providers import AntigravityProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _convert(messages: list[dict]) -> list[dict]:
    """Shortcut to call the static converter."""
    return AntigravityProvider._convert_messages(messages)


# ---------------------------------------------------------------------------
# 1. Empty text filtering
# ---------------------------------------------------------------------------
class TestEmptyTextFiltering:
    """Claude rejects text blocks with empty or missing 'text' field."""

    def test_empty_string_content_skipped(self):
        """A message with content='' should produce no parts (and be dropped)."""
        result = _convert([{"role": "assistant", "content": ""}])
        # Empty string → no text part → message dropped entirely
        assert result == []

    def test_none_content_skipped(self):
        """A message with content=None should produce no parts (and be dropped)."""
        result = _convert([{"role": "assistant", "content": None}])
        assert result == []

    def test_nonempty_text_preserved(self):
        """Non-empty text blocks should pass through normally."""
        result = _convert([{"role": "user", "content": "Hello"}])
        assert len(result) == 1
        assert result[0]["parts"] == [{"text": "Hello"}]

    def test_system_messages_filtered(self):
        """System messages are filtered out (handled separately)."""
        result = _convert([{"role": "system", "content": "You are helpful."}])
        assert result == []


# ---------------------------------------------------------------------------
# 2. Tool call (assistant → functionCall)
# ---------------------------------------------------------------------------
class TestToolCallConversion:
    """OpenAI tool_calls on assistant messages → Gemini functionCall parts."""

    def test_tool_call_basic(self):
        """tool_calls should convert to functionCall parts."""
        result = _convert([
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "echo",
                            "arguments": json.dumps({"text": "hello"}),
                        },
                    }
                ],
            }
        ])
        assert len(result) == 1
        fc = result[0]["parts"][0]["functionCall"]
        assert fc["id"] == "call_abc123"
        assert fc["name"] == "echo"
        assert fc["args"] == {"text": "hello"}

    def test_tool_call_with_text(self):
        """Non-empty text + tool_calls should both be preserved."""
        result = _convert([
            {
                "role": "assistant",
                "content": "Let me check that.",
                "tool_calls": [
                    {
                        "id": "call_002",
                        "type": "function",
                        "function": {"name": "echo", "arguments": '{"text": "test"}'},
                    }
                ],
            }
        ])
        parts = result[0]["parts"]
        assert len(parts) == 2
        assert parts[0] == {"text": "Let me check that."}
        assert parts[1]["functionCall"]["id"] == "call_002"

    def test_tool_call_with_empty_text(self):
        """Empty text alongside tool_calls should be filtered."""
        result = _convert([
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_001",
                        "type": "function",
                        "function": {"name": "mirai_system", "arguments": '{"action": "status"}'},
                    }
                ],
            }
        ])
        parts = result[0]["parts"]
        # Empty text should be filtered — only tool_call remains
        assert len(parts) == 1
        assert "functionCall" in parts[0]
        assert parts[0]["functionCall"]["id"] == "call_001"

    def test_multiple_tool_calls(self):
        """Multiple tool calls in one message should each be converted."""
        result = _convert([
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "echo", "arguments": '{"text": "a"}'}},
                    {"id": "call_2", "type": "function", "function": {"name": "echo", "arguments": '{"text": "b"}'}},
                ],
            }
        ])
        parts = result[0]["parts"]
        assert len(parts) == 2
        assert parts[0]["functionCall"]["id"] == "call_1"
        assert parts[1]["functionCall"]["id"] == "call_2"

    def test_thought_signature_preserved(self):
        """thought_signature should be preserved on the functionCall part."""
        result = _convert([
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_xyz",
                        "type": "function",
                        "function": {"name": "status", "arguments": "{}"},
                        "thought_signature": "sig_abc",
                    }
                ],
            }
        ])
        part = result[0]["parts"][0]
        assert part["functionCall"]["id"] == "call_xyz"
        assert part["thoughtSignature"] == "sig_abc"


# ---------------------------------------------------------------------------
# 3. Tool result (role="tool" → functionResponse)
# ---------------------------------------------------------------------------
class TestToolResultConversion:
    """OpenAI role="tool" messages → Gemini functionResponse."""

    def test_tool_result_basic(self):
        """role=tool with tool_call_id → functionResponse."""
        result = _convert([
            {"role": "tool", "tool_call_id": "call_abc123", "content": "Tool output here"}
        ])
        fr = result[0]["parts"][0]["functionResponse"]
        assert fr["id"] == "call_abc123"
        assert fr["name"] == "call_abc123"
        assert fr["response"] == {"result": "Tool output here"}

    def test_tool_result_maps_to_user_role(self):
        """Tool results should map to 'user' role in Gemini format."""
        result = _convert([
            {"role": "tool", "tool_call_id": "call_1", "content": "result"}
        ])
        assert result[0]["role"] == "user"


# ---------------------------------------------------------------------------
# 4. Full conversation round trip
# ---------------------------------------------------------------------------
class TestFullConversation:
    """End-to-end conversation conversion."""

    def test_full_round_trip(self):
        """Simulate a full tool-use conversation round trip.

        user → assistant(tool_calls) → tool(result) → assistant(text)
        """
        messages = [
            # User message
            {"role": "user", "content": "Check system status"},
            # Assistant calls tool
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "toolu_abc",
                        "type": "function",
                        "function": {
                            "name": "mirai_system",
                            "arguments": json.dumps({"action": "status"}),
                        },
                    }
                ],
            },
            # Tool result
            {"role": "tool", "tool_call_id": "toolu_abc", "content": '{"pid": 1234, "model": "gemini-3-flash"}'},
            # Final assistant response
            {"role": "assistant", "content": "Your system is running fine."},
        ]

        result = _convert(messages)
        assert len(result) == 4

        # Message 0: user text
        assert result[0]["role"] == "user"
        assert result[0]["parts"] == [{"text": "Check system status"}]

        # Message 1: assistant tool call
        assert result[1]["role"] == "model"
        parts1 = result[1]["parts"]
        assert len(parts1) == 1
        assert parts1[0]["functionCall"]["id"] == "toolu_abc"
        assert parts1[0]["functionCall"]["name"] == "mirai_system"

        # Message 2: tool result
        assert result[2]["role"] == "user"
        fr = result[2]["parts"][0]["functionResponse"]
        assert fr["id"] == "toolu_abc"
        assert "pid" in fr["response"]["result"]

        # Message 3: final text
        assert result[3]["role"] == "model"
        assert result[3]["parts"] == [{"text": "Your system is running fine."}]


# ---------------------------------------------------------------------------
# 5. Failover and 429 marking
# ---------------------------------------------------------------------------
class TestFailoverMarking:
    """When a model returns 429, it should be marked as exhausted."""

    @pytest.mark.asyncio
    async def test_model_marked_exhausted_on_429(self):
        """QuotaManager should mark model at 100% after 429."""
        import asyncio
        import time

        from mirai.agent.providers import QuotaManager

        qm = QuotaManager.__new__(QuotaManager)
        qm._quotas = {"model-a": 50.0, "model-b": 0.0}
        qm._last_update = time.time()  # Prevent network refresh
        qm._lock = asyncio.Lock()
        qm.credentials = {}

        # Simulate marking (what happens on 429)
        qm._quotas["model-a"] = 100.0
        assert not await qm.is_available("model-a")
        assert await qm.is_available("model-b")

    @pytest.mark.asyncio
    async def test_is_available_threshold(self):
        """Models at 100% should not be available."""
        import asyncio
        import time

        from mirai.agent.providers import QuotaManager

        qm = QuotaManager.__new__(QuotaManager)
        qm._quotas = {
            "gemini-3-flash": 100.0,
            "claude-sonnet": 20.0,
            "gemini-2.5-flash": 0.0,
        }
        qm._last_update = time.time()
        qm._lock = asyncio.Lock()
        qm.credentials = {}

        assert not await qm.is_available("gemini-3-flash")
        assert await qm.is_available("claude-sonnet")
        assert await qm.is_available("gemini-2.5-flash")
        # Unknown model defaults to 0% usage → available
        assert await qm.is_available("nonexistent-model")

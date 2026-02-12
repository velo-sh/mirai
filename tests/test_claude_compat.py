"""Tests for Claude model compatibility through Cloud Code Assist.

When the Cloud Code Assist API proxies requests to Claude models, the
Gemini-format messages must carry enough metadata (ids, non-empty text)
for the proxy to reconstruct valid Claude API messages.

Key Claude requirements that Gemini doesn't enforce:
  - tool_use blocks MUST have an `id` field
  - text blocks MUST have a non-empty `text` field
  - tool_result blocks MUST reference the tool_use `id`
"""

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

    def test_empty_text_block_skipped(self):
        """A text block with text='' inside a list should be skipped."""
        result = _convert(
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": ""},
                    ],
                }
            ]
        )
        assert result == []

    def test_none_text_block_skipped(self):
        """A text block with text=None should be skipped."""
        result = _convert(
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": None},
                    ],
                }
            ]
        )
        assert result == []

    def test_nonempty_text_preserved(self):
        """Non-empty text blocks should pass through normally."""
        result = _convert([{"role": "user", "content": "Hello"}])
        assert len(result) == 1
        assert result[0]["parts"] == [{"text": "Hello"}]

    def test_mixed_empty_and_nonempty_text(self):
        """Empty text blocks should be filtered while non-empty ones remain."""
        result = _convert(
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": ""},
                        {"type": "text", "text": "I'll help you."},
                        {"type": "text", "text": ""},
                    ],
                }
            ]
        )
        assert len(result) == 1
        parts = result[0]["parts"]
        assert len(parts) == 1
        assert parts[0] == {"text": "I'll help you."}


# ---------------------------------------------------------------------------
# 2. Tool use ID preservation
# ---------------------------------------------------------------------------
class TestToolUseIdPreservation:
    """Claude requires tool_use.id; the Gemini proxy needs it in functionCall."""

    def test_tool_use_id_included(self):
        """functionCall should include 'id' from the tool_use block."""
        result = _convert(
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_abc123",
                            "name": "echo",
                            "input": {"text": "hello"},
                        }
                    ],
                }
            ]
        )
        assert len(result) == 1
        fc = result[0]["parts"][0]["functionCall"]
        assert fc["id"] == "call_abc123"
        assert fc["name"] == "echo"
        assert fc["args"] == {"text": "hello"}

    def test_tool_use_without_id_no_crash(self):
        """If id is missing, functionCall should still work (no crash)."""
        result = _convert(
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "echo",
                            "input": {"text": "hello"},
                        }
                    ],
                }
            ]
        )
        fc = result[0]["parts"][0]["functionCall"]
        assert fc["name"] == "echo"
        assert "id" not in fc  # No id to preserve

    def test_tool_use_empty_id_not_included(self):
        """An empty string id should not be included."""
        result = _convert(
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "",
                            "name": "echo",
                            "input": {},
                        }
                    ],
                }
            ]
        )
        fc = result[0]["parts"][0]["functionCall"]
        assert "id" not in fc

    def test_thought_signature_preserved_alongside_id(self):
        """Both id and thought_signature should be preserved."""
        result = _convert(
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_xyz",
                            "name": "status",
                            "input": {},
                            "thought_signature": "sig_abc",
                        }
                    ],
                }
            ]
        )
        part = result[0]["parts"][0]
        assert part["functionCall"]["id"] == "call_xyz"
        assert part["thoughtSignature"] == "sig_abc"


# ---------------------------------------------------------------------------
# 3. Tool result ID preservation
# ---------------------------------------------------------------------------
class TestToolResultIdPreservation:
    """Claude needs to match tool results to tool calls via id."""

    def test_tool_result_id_included(self):
        """functionResponse should include 'id' from tool_use_id."""
        result = _convert(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_abc123",
                            "content": "Tool output here",
                        }
                    ],
                }
            ]
        )
        fr = result[0]["parts"][0]["functionResponse"]
        assert fr["id"] == "call_abc123"
        assert fr["name"] == "call_abc123"
        assert fr["response"] == {"result": "Tool output here"}

    def test_tool_result_without_id(self):
        """If tool_use_id is missing, fallback to 'unknown'."""
        result = _convert(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "content": "result",
                        }
                    ],
                }
            ]
        )
        fr = result[0]["parts"][0]["functionResponse"]
        assert fr["name"] == "unknown"
        assert "id" not in fr  # No id to preserve

    def test_tool_result_with_list_content(self):
        """Tool results can be a list of text blocks — should be joined."""
        result = _convert(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_xyz",
                            "content": [
                                {"type": "text", "text": "Part 1."},
                                {"type": "text", "text": "Part 2."},
                            ],
                        }
                    ],
                }
            ]
        )
        fr = result[0]["parts"][0]["functionResponse"]
        assert fr["response"] == {"result": "Part 1. Part 2."}
        assert fr["id"] == "call_xyz"


# ---------------------------------------------------------------------------
# 4. Mixed content scenarios (the real-world case)
# ---------------------------------------------------------------------------
class TestMixedContent:
    """Real assistant responses often mix text + tool_use blocks."""

    def test_empty_text_with_tool_use(self):
        """Empty text block alongside tool_use should be filtered.

        This is the exact scenario that causes the Claude 400 error:
        Gemini returns TextBlock(text='') + ToolUseBlock in the same response.
        """
        result = _convert(
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": ""},
                        {
                            "type": "tool_use",
                            "id": "call_001",
                            "name": "mirai_system",
                            "input": {"action": "status"},
                        },
                    ],
                }
            ]
        )
        assert len(result) == 1
        parts = result[0]["parts"]
        # Empty text should be filtered — only tool_use remains
        assert len(parts) == 1
        assert "functionCall" in parts[0]
        assert parts[0]["functionCall"]["id"] == "call_001"

    def test_text_with_tool_use(self):
        """Non-empty text + tool_use should both be preserved."""
        result = _convert(
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me check that."},
                        {
                            "type": "tool_use",
                            "id": "call_002",
                            "name": "echo",
                            "input": {"text": "test"},
                        },
                    ],
                }
            ]
        )
        parts = result[0]["parts"]
        assert len(parts) == 2
        assert parts[0] == {"text": "Let me check that."}
        assert parts[1]["functionCall"]["id"] == "call_002"

    def test_full_conversation_round_trip(self):
        """Simulate a full tool-use conversation round trip.

        user → assistant(tool_use) → user(tool_result) → assistant(text)
        All ids must be preserved, all empty texts filtered.
        """
        messages = [
            # User message
            {"role": "user", "content": "Check system status"},
            # Assistant calls tool (with empty text from Gemini)
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": ""},
                    {
                        "type": "tool_use",
                        "id": "toolu_abc",
                        "name": "mirai_system",
                        "input": {"action": "status"},
                    },
                ],
            },
            # Tool result
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_abc",
                        "content": '{"pid": 1234, "model": "gemini-3-flash"}',
                    }
                ],
            },
            # Final assistant response
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Your system is running fine."},
                ],
            },
        ]

        result = _convert(messages)
        assert len(result) == 4

        # Message 0: user text
        assert result[0]["role"] == "user"
        assert result[0]["parts"] == [{"text": "Check system status"}]

        # Message 1: assistant tool call (empty text filtered)
        assert result[1]["role"] == "model"
        parts1 = result[1]["parts"]
        assert len(parts1) == 1  # Empty text filtered out
        assert parts1[0]["functionCall"]["id"] == "toolu_abc"
        assert parts1[0]["functionCall"]["name"] == "mirai_system"

        # Message 2: tool result with id
        assert result[2]["role"] == "user"
        fr = result[2]["parts"][0]["functionResponse"]
        assert fr["id"] == "toolu_abc"
        assert "pid" in fr["response"]["result"]

        # Message 3: final text
        assert result[3]["role"] == "model"
        assert result[3]["parts"] == [{"text": "Your system is running fine."}]

    def test_multiple_tool_calls_preserve_ids(self):
        """Multiple tool calls in one message should each preserve their id."""
        result = _convert(
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": ""},
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "echo",
                            "input": {"text": "a"},
                        },
                        {
                            "type": "tool_use",
                            "id": "call_2",
                            "name": "echo",
                            "input": {"text": "b"},
                        },
                    ],
                }
            ]
        )
        parts = result[0]["parts"]
        assert len(parts) == 2  # Empty text filtered
        assert parts[0]["functionCall"]["id"] == "call_1"
        assert parts[1]["functionCall"]["id"] == "call_2"


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

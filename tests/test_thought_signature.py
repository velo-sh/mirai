"""
Tests for Gemini 3 thought_signature round-trip.

Validates:
1. _parse_sse_response extracts thoughtSignature from functionCall parts
2. _convert_messages replays thought_signature in tool_use blocks
3. ToolUseBlock.model_dump() preserves the signature
4. End-to-end: parse → dump → convert → verify signature present
"""

from mirai.agent.models import TextBlock, ToolUseBlock
from mirai.agent.providers import AntigravityProvider

# ---------------------------------------------------------------------------
# Fixtures: Realistic SSE payloads
# ---------------------------------------------------------------------------

SSE_WITH_SIGNATURE = (
    'data: {"response":{"candidates":[{"content":{"parts":['
    '{"functionCall":{"name":"mirai_system","args":{"action":"status"}}}'
    ']},"finishReason":"STOP"}],'
    '"thoughtSignature":"abc123_test_sig"}}\n'
)

SSE_WITH_SIGNATURE_V2 = (
    'data: {"response":{"candidates":[{"content":{"parts":['
    '{"thoughtSignature":"sig_xyz_789","functionCall":{"name":"mirai_system","args":{"action":"status"}}}'
    ']},"finishReason":"STOP"}]}}\n'
)

SSE_WITHOUT_SIGNATURE = (
    'data: {"response":{"candidates":[{"content":{"parts":['
    '{"functionCall":{"name":"echo","args":{"text":"hello"}}}'
    ']},"finishReason":"STOP"}]}}\n'
)

SSE_PARALLEL_CALLS = (
    'data: {"response":{"candidates":[{"content":{"parts":['
    '{"thoughtSignature":"sig_first","functionCall":{"name":"mirai_system","args":{"action":"status"}}},'
    '{"functionCall":{"name":"workspace_tool","args":{"action":"list","path":"."}}}'
    ']},"finishReason":"STOP"}]}}\n'
)

SSE_TEXT_AND_TOOL = (
    'data: {"response":{"candidates":[{"content":{"parts":['
    '{"text":"Let me check..."},'
    '{"thoughtSignature":"sig_after_text","functionCall":{"name":"mirai_system","args":{"action":"status"}}}'
    ']},"finishReason":"STOP"}]}}\n'
)


# ---------------------------------------------------------------------------
# Test 1: _parse_sse_response extracts thought_signature
# ---------------------------------------------------------------------------


class TestParseSSEThoughtSignature:
    """Verify _parse_sse_response extracts thoughtSignature correctly."""

    def test_signature_in_part_level(self):
        """thoughtSignature at part level (inside the part dict)."""
        resp = AntigravityProvider._parse_sse_response(SSE_WITH_SIGNATURE_V2)
        tool_blocks = resp.tool_use_blocks()
        assert len(tool_blocks) == 1
        assert tool_blocks[0].name == "mirai_system"
        assert tool_blocks[0].thought_signature == "sig_xyz_789"

    def test_signature_at_response_level_not_extracted(self):
        """thoughtSignature at response level (not in part) — should NOT be extracted."""
        resp = AntigravityProvider._parse_sse_response(SSE_WITH_SIGNATURE)
        tool_blocks = resp.tool_use_blocks()
        assert len(tool_blocks) == 1
        assert tool_blocks[0].name == "mirai_system"
        # Response-level signature is not per-part, so it's None
        assert tool_blocks[0].thought_signature is None

    def test_no_signature(self):
        """No thoughtSignature at all — should be None."""
        resp = AntigravityProvider._parse_sse_response(SSE_WITHOUT_SIGNATURE)
        tool_blocks = resp.tool_use_blocks()
        assert len(tool_blocks) == 1
        assert tool_blocks[0].thought_signature is None

    def test_parallel_calls_first_has_signature(self):
        """Parallel calls: only first functionCall has signature."""
        resp = AntigravityProvider._parse_sse_response(SSE_PARALLEL_CALLS)
        tool_blocks = resp.tool_use_blocks()
        assert len(tool_blocks) == 2
        assert tool_blocks[0].thought_signature == "sig_first"
        assert tool_blocks[1].thought_signature is None

    def test_text_and_tool_with_signature(self):
        """Text block followed by functionCall with signature."""
        resp = AntigravityProvider._parse_sse_response(SSE_TEXT_AND_TOOL)
        assert len(resp.content) == 2
        assert isinstance(resp.content[0], TextBlock)
        assert resp.content[0].text == "Let me check..."
        tool_blocks = resp.tool_use_blocks()
        assert len(tool_blocks) == 1
        assert tool_blocks[0].thought_signature == "sig_after_text"


# ---------------------------------------------------------------------------
# Test 2: ToolUseBlock.model_dump() preserves thought_signature
# ---------------------------------------------------------------------------


class TestToolUseBlockDump:
    """Verify model_dump includes thought_signature when present."""

    def test_dump_with_signature(self):
        block = ToolUseBlock(id="call_1", name="mirai_system", input={"action": "status"}, thought_signature="sig_abc")
        dumped = block.model_dump()
        assert dumped["thought_signature"] == "sig_abc"
        assert dumped["name"] == "mirai_system"

    def test_dump_without_signature(self):
        block = ToolUseBlock(id="call_1", name="echo", input={"text": "hi"})
        dumped = block.model_dump()
        assert dumped["thought_signature"] is None


# ---------------------------------------------------------------------------
# Test 3: _convert_messages replays thought_signature
# ---------------------------------------------------------------------------


class TestConvertMessagesThoughtSignature:
    """Verify _convert_messages replays thought_signature as thoughtSignature."""

    def test_tool_use_with_signature_replayed(self):
        """tool_calls with thought_signature → functionCall + thoughtSignature."""
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "mirai_system",
                            "arguments": '{"action": "status"}',
                        },
                        "thought_signature": "sig_replay_test",
                    }
                ],
            }
        ]
        converted = AntigravityProvider._convert_messages(messages)
        assert len(converted) == 1
        parts = converted[0]["parts"]
        assert len(parts) == 1
        assert "functionCall" in parts[0]
        assert parts[0]["functionCall"]["name"] == "mirai_system"
        assert parts[0].get("thoughtSignature") == "sig_replay_test"

    def test_tool_use_without_signature_no_extra_field(self):
        """tool_calls without thought_signature → no thoughtSignature key."""
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "echo", "arguments": '{"text": "hi"}'},
                    }
                ],
            }
        ]
        converted = AntigravityProvider._convert_messages(messages)
        parts = converted[0]["parts"]
        assert "thoughtSignature" not in parts[0]

    def test_tool_use_with_none_signature_no_extra_field(self):
        """tool_calls with explicit None thought_signature → no thoughtSignature key."""
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "echo", "arguments": '{"text": "hi"}'},
                        "thought_signature": None,
                    }
                ],
            }
        ]
        converted = AntigravityProvider._convert_messages(messages)
        parts = converted[0]["parts"]
        assert "thoughtSignature" not in parts[0]


# ---------------------------------------------------------------------------
# Test 4: End-to-end round-trip
# ---------------------------------------------------------------------------


class TestThoughtSignatureRoundTrip:
    """End-to-end: parse SSE → model_dump → convert_messages → verify."""

    def test_full_round_trip(self):
        import json

        # Step 1: Parse SSE response (simulating Gemini API response)
        resp = AntigravityProvider._parse_sse_response(SSE_WITH_SIGNATURE_V2)
        tool_block = resp.tool_use_blocks()[0]
        assert tool_block.thought_signature == "sig_xyz_789"

        # Step 2: Build OpenAI-format assistant message (as done in loop.py)
        assistant_msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_block.id,
                    "type": "function",
                    "function": {
                        "name": tool_block.name,
                        "arguments": json.dumps(tool_block.input),
                    },
                    "thought_signature": tool_block.thought_signature,
                }
            ],
        }

        # Step 3: Convert back to Gemini format (as done in _convert_messages)
        converted = AntigravityProvider._convert_messages([assistant_msg])
        parts = converted[0]["parts"]

        # Verify thoughtSignature is present in the functionCall part
        assert "thoughtSignature" in parts[0], f"thoughtSignature missing from converted part: {parts[0]}"
        assert parts[0]["thoughtSignature"] == "sig_xyz_789"
        assert parts[0]["functionCall"]["name"] == "mirai_system"

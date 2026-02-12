"""Unit tests for mirai.agent.providers — SSE parsing, message conversion, model mapping, retry."""

import pytest

from mirai.agent.models import ProviderResponse, TextBlock
from mirai.agent.providers import AntigravityProvider, MockProvider, _RetryableAPIError

# ---------------------------------------------------------------------------
# SSE Response Parsing
# ---------------------------------------------------------------------------


class TestParseSSEResponse:
    """Tests for AntigravityProvider._parse_sse_response (static method)."""

    def _make_sse(self, *chunks: dict) -> str:
        """Build a realistic SSE stream from response chunks."""
        import orjson

        lines = []
        for chunk in chunks:
            lines.append(f"data: {orjson.dumps(chunk).decode()}")
            lines.append("")  # empty line separator
        return "\n".join(lines)

    # -- Text responses --

    def test_single_text_chunk(self):
        sse = self._make_sse(
            {"response": {"candidates": [{"content": {"parts": [{"text": "Hello, world!"}]}, "finishReason": "STOP"}]}}
        )
        resp = AntigravityProvider._parse_sse_response(sse)
        assert isinstance(resp, ProviderResponse)
        assert resp.text() == "Hello, world!"
        assert resp.stop_reason == "end_turn"

    def test_multiple_text_chunks_merged(self):
        """Consecutive text parts across SSE data lines should merge into a single TextBlock."""
        sse = self._make_sse(
            {"response": {"candidates": [{"content": {"parts": [{"text": "Hello, "}]}}]}},
            {"response": {"candidates": [{"content": {"parts": [{"text": "world!"}]}, "finishReason": "STOP"}]}},
        )
        resp = AntigravityProvider._parse_sse_response(sse)
        assert resp.text() == "Hello, world!"
        # Should be a single merged block, not two
        text_blocks = [b for b in resp.content if isinstance(b, TextBlock)]
        assert len(text_blocks) == 1

    def test_empty_response_fallback(self):
        """Empty SSE should produce a single empty TextBlock."""
        resp = AntigravityProvider._parse_sse_response("")
        assert len(resp.content) == 1
        assert isinstance(resp.content[0], TextBlock)
        assert resp.content[0].text == ""

    def test_invalid_json_lines_skipped(self):
        sse = "data: {invalid json}\ndata: also bad\n"
        resp = AntigravityProvider._parse_sse_response(sse)
        # Should fall back to empty TextBlock
        assert len(resp.content) == 1
        assert resp.content[0].text == ""

    def test_non_data_lines_skipped(self):
        sse = "event: message\nid: 1\n: comment\nretry: 5000\n"
        resp = AntigravityProvider._parse_sse_response(sse)
        assert len(resp.content) == 1  # empty fallback

    # -- Tool use responses --

    def test_function_call_parsed(self):
        sse = self._make_sse(
            {
                "response": {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "functionCall": {
                                            "name": "echo",
                                            "args": {"message": "test"},
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }
            }
        )
        resp = AntigravityProvider._parse_sse_response(sse)
        assert resp.stop_reason == "tool_use"
        tools = resp.tool_use_blocks()
        assert len(tools) == 1
        assert tools[0].name == "echo"
        assert tools[0].input == {"message": "test"}

    def test_text_then_tool_call(self):
        """Mixed content: text followed by a tool call."""
        sse = self._make_sse(
            {"response": {"candidates": [{"content": {"parts": [{"text": "Let me help. "}]}}]}},
            {
                "response": {
                    "candidates": [{"content": {"parts": [{"functionCall": {"name": "shell", "args": {"cmd": "ls"}}}]}}]
                }
            },
        )
        resp = AntigravityProvider._parse_sse_response(sse)
        assert resp.text() == "Let me help. "
        assert resp.stop_reason == "tool_use"
        assert len(resp.tool_use_blocks()) == 1

    # -- Finish reasons --

    def test_max_tokens_finish_reason(self):
        sse = self._make_sse(
            {
                "response": {
                    "candidates": [{"content": {"parts": [{"text": "truncated"}]}, "finishReason": "MAX_TOKENS"}]
                }
            }
        )
        resp = AntigravityProvider._parse_sse_response(sse)
        assert resp.stop_reason == "max_tokens"

    def test_no_candidates(self):
        sse = self._make_sse({"response": {}})
        resp = AntigravityProvider._parse_sse_response(sse)
        assert len(resp.content) == 1
        assert resp.content[0].text == ""


# ---------------------------------------------------------------------------
# Message Conversion (Anthropic → Google GenAI format)
# ---------------------------------------------------------------------------


class TestConvertMessages:
    def test_simple_user_message(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = AntigravityProvider._convert_messages(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["parts"] == [{"text": "Hello"}]

    def test_assistant_maps_to_model(self):
        messages = [{"role": "assistant", "content": "Hi there"}]
        result = AntigravityProvider._convert_messages(messages)
        assert result[0]["role"] == "model"

    def test_text_block_content(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "multi-block"}]}]
        result = AntigravityProvider._convert_messages(messages)
        assert result[0]["parts"] == [{"text": "multi-block"}]

    def test_tool_use_block_content(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "name": "echo", "input": {"message": "test"}},
                ],
            }
        ]
        result = AntigravityProvider._convert_messages(messages)
        fc = result[0]["parts"][0]["functionCall"]
        assert fc["name"] == "echo"
        assert fc["args"] == {"message": "test"}

    def test_tool_result_block_content(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call_1", "content": "result_text"},
                ],
            }
        ]
        result = AntigravityProvider._convert_messages(messages)
        fr = result[0]["parts"][0]["functionResponse"]
        assert fr["name"] == "call_1"
        assert fr["response"]["result"] == "result_text"

    def test_empty_messages(self):
        assert AntigravityProvider._convert_messages([]) == []

    def test_empty_content_produces_text_part(self):
        messages = [{"role": "user", "content": ""}]
        result = AntigravityProvider._convert_messages(messages)
        # Empty string content is now skipped for Claude compatibility
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Tool Conversion
# ---------------------------------------------------------------------------


class TestConvertTools:
    def test_single_tool(self):
        tools = [
            {
                "name": "echo",
                "description": "Echo tool",
                "input_schema": {"type": "object", "properties": {"message": {"type": "string"}}},
            }
        ]
        result = AntigravityProvider._convert_tools(tools)
        assert len(result) == 1
        decls = result[0]["functionDeclarations"]
        assert len(decls) == 1
        assert decls[0]["name"] == "echo"
        assert "parameters" in decls[0]

    def test_empty_tools(self):
        assert AntigravityProvider._convert_tools([]) == []

    def test_tool_without_schema(self):
        tools = [{"name": "simple", "description": "No schema"}]
        result = AntigravityProvider._convert_tools(tools)
        decls = result[0]["functionDeclarations"]
        assert decls[0]["name"] == "simple"
        assert "parameters" not in decls[0]


# ---------------------------------------------------------------------------
# Model Mapping
# ---------------------------------------------------------------------------


class TestModelMapping:
    def test_claude_sonnet_4_mapped(self):
        assert AntigravityProvider.MODEL_MAP["claude-sonnet-4-20250514"] == "claude-sonnet-4-5"

    def test_legacy_claude_35_mapped(self):
        assert AntigravityProvider.MODEL_MAP["claude-3-5-sonnet-20241022"] == "claude-sonnet-4-5"

    def test_gemini_passthrough(self):
        assert AntigravityProvider.MODEL_MAP["gemini-2.0-flash"] == "gemini-2.0-flash"

    def test_unknown_model_unmapped(self):
        # Unknown models should NOT be in MODEL_MAP
        assert "unknown-model-v99" not in AntigravityProvider.MODEL_MAP


# ---------------------------------------------------------------------------
# _RetryableAPIError
# ---------------------------------------------------------------------------


class TestRetryableAPIError:
    def test_stores_status(self):
        err = _RetryableAPIError(429, "rate limited")
        assert err.status == 429
        assert "429" in str(err)
        assert "rate limited" in str(err)

    def test_is_exception(self):
        assert issubclass(_RetryableAPIError, Exception)


# ---------------------------------------------------------------------------
# MockProvider
# ---------------------------------------------------------------------------


class TestMockProvider:
    def test_init(self):
        p = MockProvider()
        assert p.model == "mock-model"
        assert p.call_count == 0

    @pytest.mark.asyncio
    async def test_returns_provider_response(self):
        p = MockProvider()
        resp = await p.generate_response(
            model="mock", system="sys", messages=[{"role": "user", "content": "hi"}], tools=[]
        )
        assert isinstance(resp, ProviderResponse)
        assert p.call_count == 1

    @pytest.mark.asyncio
    async def test_multiple_calls_increment_count(self):
        p = MockProvider()
        for _ in range(3):
            await p.generate_response(model="m", system="s", messages=[{"role": "user", "content": "x"}], tools=[])
        assert p.call_count == 3

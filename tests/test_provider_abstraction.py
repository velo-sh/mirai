"""QA tests for the provider abstraction layer.

Covers all new and refactored components from the OpenAI-canonical-format
migration, focusing on boundary conditions, error paths, and cross-provider
round-trip correctness.

Test categories:
  1. LLMConfig — new provider/api_key/base_url fields
  2. Provider Factory — all dispatch paths and error handling
  3. OpenAIProvider — tool conversion, response parsing
  4. AnthropicProvider — OpenAI→Claude message conversion (all branches)
  5. Message Converter — OpenAI→Gemini edge cases
  6. MockProvider helpers — OpenAI-format message inspection
  7. ProviderProtocol compliance
  8. Cross-provider message round-trip
"""

from unittest.mock import MagicMock, patch

import pytest

from mirai.agent.models import ProviderResponse, TextBlock
from mirai.agent.providers.base import ProviderProtocol

# ---------------------------------------------------------------------------
# 1. LLMConfig — new fields
# ---------------------------------------------------------------------------


class TestLLMConfigFields:
    """Verify the expanded LLMConfig handles provider/api_key/base_url."""

    def test_default_provider(self):
        from mirai.config import LLMConfig

        cfg = LLMConfig()
        assert cfg.provider == "antigravity"

    def test_default_api_key_is_none(self):
        from mirai.config import LLMConfig

        cfg = LLMConfig()
        assert cfg.api_key is None

    def test_default_base_url_is_none(self):
        from mirai.config import LLMConfig

        cfg = LLMConfig()
        assert cfg.base_url is None

    def test_custom_provider(self):
        from mirai.config import LLMConfig

        cfg = LLMConfig(provider="openai")
        assert cfg.provider == "openai"

    def test_custom_api_key(self):
        from mirai.config import LLMConfig

        cfg = LLMConfig(api_key="sk-test-123")
        assert cfg.api_key == "sk-test-123"

    def test_custom_base_url(self):
        from mirai.config import LLMConfig

        cfg = LLMConfig(base_url="https://api.deepseek.com/v1")
        assert cfg.base_url == "https://api.deepseek.com/v1"

    def test_toml_provider_override(self, tmp_toml):
        from mirai.config import MiraiConfig

        toml_path = tmp_toml("""
[llm]
provider = "openai"
api_key = "sk-from-toml"
base_url = "https://custom.api/v1"
""")
        cfg = MiraiConfig.load(config_path=toml_path)
        assert cfg.llm.provider == "openai"
        assert cfg.llm.api_key == "sk-from-toml"
        assert cfg.llm.base_url == "https://custom.api/v1"

    def test_env_provider_override(self, monkeypatch):
        from mirai.config import MiraiConfig

        monkeypatch.setenv("MIRAI_LLM__PROVIDER", "anthropic")
        cfg = MiraiConfig.load()
        assert cfg.llm.provider == "anthropic"


# ---------------------------------------------------------------------------
# 2. Provider Factory — dispatch & error paths
# ---------------------------------------------------------------------------


class TestProviderFactory:
    """Test create_provider dispatches correctly and errors on missing creds."""

    def test_openai_provider_with_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from mirai.agent.providers.factory import create_provider
        from mirai.agent.providers.openai import OpenAIProvider

        p = create_provider(provider="openai", model="gpt-4o", api_key="sk-test")
        assert isinstance(p, OpenAIProvider)
        assert p.model == "gpt-4o"

    def test_openai_provider_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-test")
        from mirai.agent.providers.factory import create_provider
        from mirai.agent.providers.openai import OpenAIProvider

        p = create_provider(provider="openai", model="gpt-4o-mini")
        assert isinstance(p, OpenAIProvider)

    def test_openai_provider_raises_without_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from mirai.agent.providers.factory import create_provider
        from mirai.errors import ProviderError

        with pytest.raises((ValueError, ProviderError), match="requires an API key"):
            create_provider(provider="openai", model="gpt-4o")

    def test_unknown_provider_falls_to_openai(self, monkeypatch):
        """Unknown provider names should be treated as OpenAI-compatible."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from mirai.agent.providers.factory import create_provider
        from mirai.agent.providers.openai import OpenAIProvider

        p = create_provider(
            provider="deepseek",
            model="deepseek-chat",
            api_key="sk-ds",
            base_url="https://api.deepseek.com/v1",
        )
        assert isinstance(p, OpenAIProvider)
        assert p.model == "deepseek-chat"

    def test_anthropic_provider_with_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from mirai.agent.providers.anthropic import AnthropicProvider
        from mirai.agent.providers.factory import create_provider

        p = create_provider(provider="anthropic", model="claude-sonnet-4-20250514", api_key="sk-ant-test")
        assert isinstance(p, AnthropicProvider)
        assert p.model == "claude-sonnet-4-20250514"

    def test_anthropic_provider_from_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
        from mirai.agent.providers.anthropic import AnthropicProvider
        from mirai.agent.providers.factory import create_provider

        p = create_provider(provider="anthropic")
        assert isinstance(p, AnthropicProvider)

    def test_antigravity_fallback_to_anthropic(self, monkeypatch):
        """When antigravity has no creds, should fall through to anthropic if ANTHROPIC_API_KEY is set."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fallback")
        from mirai.agent.providers.anthropic import AnthropicProvider
        from mirai.agent.providers.factory import create_provider

        # Patch load_credentials at its source module
        with patch("mirai.auth.antigravity_auth.load_credentials", return_value=None):
            p = create_provider(provider="antigravity")
            assert isinstance(p, AnthropicProvider)

    def test_no_credentials_raises(self, monkeypatch):
        """When antigravity has no creds and no env keys, should raise."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from mirai.agent.providers.factory import create_provider
        from mirai.errors import ProviderError

        with patch("mirai.auth.antigravity_auth.load_credentials", return_value=None):
            with pytest.raises((ValueError, ProviderError), match="No API credentials"):
                create_provider(provider="antigravity")


# ---------------------------------------------------------------------------
# 3. OpenAIProvider — static helpers
# ---------------------------------------------------------------------------


class TestOpenAIProviderToolConversion:
    """Test OpenAIProvider._convert_tools (internal → OpenAI function format)."""

    def test_basic_tool(self):
        from mirai.agent.providers.openai import OpenAIProvider

        tools = [
            {
                "name": "echo",
                "description": "Echo input",
                "input_schema": {"type": "object", "properties": {"msg": {"type": "string"}}},
            }
        ]
        result = OpenAIProvider._convert_tools(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        fn = result[0]["function"]
        assert fn["name"] == "echo"
        assert fn["description"] == "Echo input"
        assert fn["parameters"]["type"] == "object"

    def test_tool_without_description(self):
        from mirai.agent.providers.openai import OpenAIProvider

        tools = [{"name": "noop", "input_schema": {"type": "object"}}]
        result = OpenAIProvider._convert_tools(tools)
        assert result[0]["function"]["description"] == ""

    def test_tool_without_schema(self):
        from mirai.agent.providers.openai import OpenAIProvider

        tools = [{"name": "simple", "description": "No params"}]
        result = OpenAIProvider._convert_tools(tools)
        assert result[0]["function"]["parameters"] == {}

    def test_multiple_tools(self):
        from mirai.agent.providers.openai import OpenAIProvider

        tools = [
            {"name": "tool_a", "description": "A", "input_schema": {}},
            {"name": "tool_b", "description": "B", "input_schema": {}},
            {"name": "tool_c", "description": "C", "input_schema": {}},
        ]
        result = OpenAIProvider._convert_tools(tools)
        assert len(result) == 3
        names = [r["function"]["name"] for r in result]
        assert names == ["tool_a", "tool_b", "tool_c"]


class TestOpenAIProviderResponseParsing:
    """Test OpenAIProvider._to_provider_response with various mock responses."""

    def _mock_response(self, content=None, tool_calls=None, finish_reason="stop"):
        """Build a mock ChatCompletion response object."""
        message = MagicMock()
        message.content = content
        message.tool_calls = tool_calls

        choice = MagicMock()
        choice.message = message
        choice.finish_reason = finish_reason

        response = MagicMock()
        response.choices = [choice]
        response.model = "gpt-4o"
        return response

    def _mock_tool_call(self, tc_id="call_1", name="echo", arguments='{"text":"hi"}'):
        tc = MagicMock()
        tc.id = tc_id
        tc.function = MagicMock()
        tc.function.name = name
        tc.function.arguments = arguments
        return tc

    def test_text_response(self):
        from mirai.agent.providers.openai import OpenAIProvider

        resp = self._mock_response(content="Hello!")
        pr = OpenAIProvider._to_provider_response(resp)
        assert isinstance(pr, ProviderResponse)
        assert pr.text() == "Hello!"
        assert pr.stop_reason == "end_turn"

    def test_tool_call_response(self):
        from mirai.agent.providers.openai import OpenAIProvider

        tc = self._mock_tool_call()
        resp = self._mock_response(tool_calls=[tc], finish_reason="tool_calls")
        pr = OpenAIProvider._to_provider_response(resp)
        assert pr.stop_reason == "tool_use"
        tools = pr.tool_use_blocks()
        assert len(tools) == 1
        assert tools[0].name == "echo"
        assert tools[0].input == {"text": "hi"}

    def test_text_and_tool_call(self):
        from mirai.agent.providers.openai import OpenAIProvider

        tc = self._mock_tool_call()
        resp = self._mock_response(content="Let me check...", tool_calls=[tc])
        pr = OpenAIProvider._to_provider_response(resp)
        assert pr.text() == "Let me check..."
        assert len(pr.tool_use_blocks()) == 1

    def test_empty_response_fallback(self):
        from mirai.agent.providers.openai import OpenAIProvider

        resp = self._mock_response(content=None, tool_calls=None)
        pr = OpenAIProvider._to_provider_response(resp)
        assert len(pr.content) == 1
        assert isinstance(pr.content[0], TextBlock)
        assert pr.content[0].text == ""

    def test_length_finish_reason(self):
        from mirai.agent.providers.openai import OpenAIProvider

        resp = self._mock_response(content="truncated...", finish_reason="length")
        pr = OpenAIProvider._to_provider_response(resp)
        assert pr.stop_reason == "max_tokens"

    def test_malformed_arguments_json(self):
        """Tool call with invalid JSON arguments should default to {}."""
        from mirai.agent.providers.openai import OpenAIProvider

        tc = self._mock_tool_call(arguments="not a json")
        resp = self._mock_response(tool_calls=[tc], finish_reason="tool_calls")
        pr = OpenAIProvider._to_provider_response(resp)
        tools = pr.tool_use_blocks()
        assert tools[0].input == {}

    def test_model_id_override(self):
        from mirai.agent.providers.openai import OpenAIProvider

        resp = self._mock_response(content="hi")
        pr = OpenAIProvider._to_provider_response(resp, model_id="custom-model")
        assert pr.model_id == "custom-model"

    def test_model_id_from_response(self):
        from mirai.agent.providers.openai import OpenAIProvider

        resp = self._mock_response(content="hi")
        pr = OpenAIProvider._to_provider_response(resp)
        assert pr.model_id == "gpt-4o"

    def test_multiple_tool_calls(self):
        from mirai.agent.providers.openai import OpenAIProvider

        tc1 = self._mock_tool_call(tc_id="call_1", name="shell", arguments='{"cmd":"ls"}')
        tc2 = self._mock_tool_call(tc_id="call_2", name="echo", arguments='{"text":"done"}')
        resp = self._mock_response(tool_calls=[tc1, tc2], finish_reason="tool_calls")
        pr = OpenAIProvider._to_provider_response(resp)
        assert len(pr.tool_use_blocks()) == 2
        assert pr.tool_use_blocks()[0].name == "shell"
        assert pr.tool_use_blocks()[1].name == "echo"


# ---------------------------------------------------------------------------
# 4. AnthropicProvider — OpenAI→Claude conversion (all branches)
# ---------------------------------------------------------------------------


class TestAnthropicConvertMessages:
    """Test AnthropicProvider._convert_messages with all input formats."""

    @staticmethod
    def _convert(messages):
        from mirai.agent.providers.anthropic import AnthropicProvider

        return AnthropicProvider._convert_messages(messages)

    def test_user_message_passthrough(self):
        result = self._convert([{"role": "user", "content": "Hello"}])
        assert result == [{"role": "user", "content": "Hello"}]

    def test_system_message_filtered(self):
        result = self._convert([{"role": "system", "content": "You are helpful."}])
        assert result == []

    def test_assistant_text_only(self):
        result = self._convert([{"role": "assistant", "content": "Hi there"}])
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == [{"type": "text", "text": "Hi there"}]

    def test_assistant_with_tool_calls(self):
        result = self._convert(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {"name": "echo", "arguments": '{"msg": "test"}'},
                        }
                    ],
                }
            ]
        )
        assert len(result) == 1
        blocks = result[0]["content"]
        assert len(blocks) == 1
        assert blocks[0]["type"] == "tool_use"
        assert blocks[0]["id"] == "call_abc"
        assert blocks[0]["name"] == "echo"
        assert blocks[0]["input"] == {"msg": "test"}

    def test_assistant_text_and_tool_calls(self):
        result = self._convert(
            [
                {
                    "role": "assistant",
                    "content": "Let me check.",
                    "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"name": "status", "arguments": "{}"}},
                    ],
                }
            ]
        )
        blocks = result[0]["content"]
        assert len(blocks) == 2
        assert blocks[0] == {"type": "text", "text": "Let me check."}
        assert blocks[1]["type"] == "tool_use"

    def test_assistant_empty_content_no_tool_calls(self):
        """Assistant with no text and no tool_calls produces an empty content."""
        result = self._convert([{"role": "assistant", "content": None}])
        assert result[0]["content"] == ""

    def test_tool_result_conversion(self):
        result = self._convert([{"role": "tool", "tool_call_id": "call_abc", "content": "Success!"}])
        assert len(result) == 1
        assert result[0]["role"] == "user"
        block = result[0]["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "call_abc"
        assert block["content"] == "Success!"

    def test_tool_result_missing_id(self):
        """Missing tool_call_id defaults to empty string."""
        result = self._convert([{"role": "tool", "content": "Failed"}])
        block = result[0]["content"][0]
        assert block["tool_use_id"] == ""

    def test_malformed_arguments(self):
        """Invalid JSON in function arguments should default to empty dict."""
        result = self._convert(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"name": "broken", "arguments": "not json"}},
                    ],
                }
            ]
        )
        blocks = result[0]["content"]
        assert blocks[0]["input"] == {}

    def test_full_conversation_sequence(self):
        """Verify correct conversion of a multi-turn tool-use conversation."""
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Check status"},
            {
                "role": "assistant",
                "content": "Checking...",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "status", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": '{"ok": true}'},
            {"role": "assistant", "content": "All good!"},
        ]
        result = self._convert(messages)
        # system filtered out → 4 messages
        assert len(result) == 4
        assert result[0]["role"] == "user"  # user
        assert result[1]["role"] == "assistant"  # assistant + tool_call
        assert result[2]["role"] == "user"  # tool result (mapped to user)
        assert result[3]["role"] == "assistant"  # final text


class TestAnthropicConvertTools:
    """Anthropic tool format is our internal format — should passthrough."""

    def test_passthrough(self):
        from mirai.agent.providers.anthropic import AnthropicProvider

        tools = [{"name": "echo", "description": "Echo", "input_schema": {"type": "object"}}]
        result = AnthropicProvider._convert_tools(tools)
        assert result is tools  # Same reference — no copy

    def test_empty_tools(self):
        from mirai.agent.providers.anthropic import AnthropicProvider

        assert AnthropicProvider._convert_tools([]) == []


# ---------------------------------------------------------------------------
# 5. Message Converter — OpenAI→Gemini edge cases
# ---------------------------------------------------------------------------


class TestMessageConverterEdgeCases:
    """Edge cases in OpenAI→Gemini message conversion."""

    @staticmethod
    def _convert(messages):
        from mirai.agent.providers.message_converter import convert_messages

        return convert_messages(messages)

    def test_malformed_arguments_json(self):
        """Invalid JSON in tool_call arguments should default to {}."""
        result = self._convert(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "broken", "arguments": "{{invalid}}"}},
                    ],
                }
            ]
        )
        fc = result[0]["parts"][0]["functionCall"]
        assert fc["args"] == {}

    def test_tool_call_missing_arguments(self):
        """Missing arguments field should default to {}."""
        result = self._convert(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "noop"}},
                    ],
                }
            ]
        )
        fc = result[0]["parts"][0]["functionCall"]
        assert fc["args"] == {}

    def test_tool_call_missing_id(self):
        """Missing tool_call ID should still produce a functionCall."""
        result = self._convert(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"function": {"name": "test", "arguments": "{}"}},
                    ],
                }
            ]
        )
        fc = result[0]["parts"][0]["functionCall"]
        assert fc["name"] == "test"
        assert "id" not in fc

    def test_tool_result_missing_content(self):
        """Tool result with no content field should default to empty string."""
        result = self._convert([{"role": "tool", "tool_call_id": "c1"}])
        fr = result[0]["parts"][0]["functionResponse"]
        assert fr["response"]["result"] == ""

    def test_tool_result_missing_id(self):
        """Tool result with no tool_call_id should default to 'unknown'."""
        result = self._convert([{"role": "tool", "content": "some result"}])
        fr = result[0]["parts"][0]["functionResponse"]
        assert fr["name"] == "unknown"

    def test_image_block_content(self):
        """Base64 image content should produce inlineData."""
        result = self._convert(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this:"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": "abc123base64data",
                            },
                        },
                    ],
                }
            ]
        )
        parts = result[0]["parts"]
        assert len(parts) == 2
        assert parts[0] == {"text": "Describe this:"}
        assert parts[1]["inlineData"]["mimeType"] == "image/jpeg"
        assert parts[1]["inlineData"]["data"] == "abc123base64data"

    def test_image_block_without_base64_type(self):
        """Non-base64 image source should not produce inlineData."""
        result = self._convert(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "url", "url": "https://example.com/img.png"}},
                    ],
                }
            ]
        )
        # No parts can be created from a URL-type image source
        assert result == []

    def test_empty_text_in_list_content_skipped(self):
        """Empty text blocks in list content should be skipped."""
        result = self._convert([{"role": "user", "content": [{"type": "text", "text": ""}]}])
        # Empty text → no parts → message dropped
        assert result == []

    def test_mixed_roles_ordering_preserved(self):
        """Message ordering must be preserved across different roles."""
        messages = [
            {"role": "user", "content": "Step 1"},
            {"role": "assistant", "content": "Step 2"},
            {"role": "user", "content": "Step 3"},
        ]
        result = self._convert(messages)
        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "model"
        assert result[2]["role"] == "user"

    def test_unknown_block_type_ignored(self):
        """Unknown block types in list content should be silently skipped."""
        result = self._convert([{"role": "user", "content": [{"type": "audio", "data": "..."}]}])
        assert result == []

    def test_thought_signature_only_on_truthy_values(self):
        """Empty or None thought_signature should not produce thoughtSignature key."""
        result = self._convert(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "test", "arguments": "{}"}, "thought_signature": ""},
                    ],
                }
            ]
        )
        part = result[0]["parts"][0]
        assert "functionCall" in part
        assert "thoughtSignature" not in part

    def test_none_thought_signature_excluded(self):
        result = self._convert(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "test", "arguments": "{}"}, "thought_signature": None},
                    ],
                }
            ]
        )
        part = result[0]["parts"][0]
        assert "thoughtSignature" not in part

    def test_assistant_with_text_and_tool_calls(self):
        """Non-empty text + tool_calls should produce text part + functionCall parts."""
        result = self._convert(
            [
                {
                    "role": "assistant",
                    "content": "Let me check...",
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "status", "arguments": "{}"}},
                    ],
                }
            ]
        )
        parts = result[0]["parts"]
        assert len(parts) == 2
        assert parts[0] == {"text": "Let me check..."}
        assert "functionCall" in parts[1]


# ---------------------------------------------------------------------------
# 6. MockProvider helpers
# ---------------------------------------------------------------------------


class TestMockProviderHelpers:
    """Test MockProvider helper methods for OpenAI-format inspection."""

    def test_find_tool_results_match(self):
        from tests.mocks.providers import MockProvider

        messages = [
            {"role": "tool", "tool_call_id": "call_1", "content": "result_1"},
            {"role": "tool", "tool_call_id": "call_2", "content": "result_2"},
            {"role": "tool", "tool_call_id": "call_1", "content": "result_1b"},
        ]
        results = MockProvider._find_tool_results(messages, "call_1")
        assert len(results) == 2
        assert results[0]["content"] == "result_1"
        assert results[1]["content"] == "result_1b"

    def test_find_tool_results_no_match(self):
        from tests.mocks.providers import MockProvider

        messages = [
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        ]
        assert MockProvider._find_tool_results(messages, "call_999") == []

    def test_find_tool_results_ignores_non_tool_roles(self):
        from tests.mocks.providers import MockProvider

        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ]
        results = MockProvider._find_tool_results(messages, "c1")
        assert len(results) == 1

    def test_has_tool_call_match(self):
        from tests.mocks.providers import MockProvider

        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c1", "function": {"name": "echo"}},
                    {"id": "c2", "function": {"name": "shell"}},
                ],
            }
        ]
        assert MockProvider._has_tool_call(messages, "echo") is True
        assert MockProvider._has_tool_call(messages, "shell") is True
        assert MockProvider._has_tool_call(messages, "missing") is False

    def test_has_tool_call_no_assistant(self):
        from tests.mocks.providers import MockProvider

        messages = [{"role": "user", "content": "hi"}]
        assert MockProvider._has_tool_call(messages, "echo") is False

    def test_has_tool_call_multiple_messages(self):
        from tests.mocks.providers import MockProvider

        messages = [
            {
                "role": "assistant",
                "content": "first",
                "tool_calls": [{"id": "c1", "function": {"name": "tool_a"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "c2", "function": {"name": "tool_b"}}],
            },
        ]
        assert MockProvider._has_tool_call(messages, "tool_a") is True
        assert MockProvider._has_tool_call(messages, "tool_b") is True
        assert MockProvider._has_tool_call(messages, "tool_c") is False

    def test_has_tool_call_empty_tool_calls(self):
        from tests.mocks.providers import MockProvider

        messages = [{"role": "assistant", "content": "hi", "tool_calls": []}]
        assert MockProvider._has_tool_call(messages, "anything") is False

    def test_has_tool_call_no_tool_calls_key(self):
        from tests.mocks.providers import MockProvider

        messages = [{"role": "assistant", "content": "hi"}]
        assert MockProvider._has_tool_call(messages, "anything") is False


# ---------------------------------------------------------------------------
# 7. ProviderProtocol compliance
# ---------------------------------------------------------------------------


class TestProviderProtocol:
    """Verify all providers satisfy ProviderProtocol."""

    def test_mock_provider_satisfies_protocol(self):
        from tests.mocks.providers import MockProvider

        p = MockProvider()
        assert isinstance(p, ProviderProtocol)

    def test_openai_provider_satisfies_protocol(self):
        from mirai.agent.providers.openai import OpenAIProvider

        p = OpenAIProvider(api_key="test", model="gpt-4o")
        assert isinstance(p, ProviderProtocol)

    def test_anthropic_provider_satisfies_protocol(self):
        from mirai.agent.providers.anthropic import AnthropicProvider

        p = AnthropicProvider(api_key="test", model="claude-sonnet-4-20250514")
        assert isinstance(p, ProviderProtocol)


# ---------------------------------------------------------------------------
# 8. Cross-provider message round-trip consistency
# ---------------------------------------------------------------------------


class TestCrossProviderRoundTrip:
    """Verify the same OpenAI-format conversation converts correctly to both
    Gemini and Anthropic, preserving semantic equivalence."""

    CONVERSATION = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Check status"},
        {
            "role": "assistant",
            "content": "Let me check.",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "status", "arguments": '{"action": "check"}'}},
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": '{"ok": true}'},
        {"role": "assistant", "content": "Everything is fine."},
    ]

    def test_gemini_conversation_structure(self):
        """Gemini should have 4 messages (system filtered)."""
        from mirai.agent.providers.message_converter import convert_messages

        result = convert_messages(self.CONVERSATION)
        assert len(result) == 4
        roles = [m["role"] for m in result]
        assert roles == ["user", "model", "user", "model"]

    def test_anthropic_conversation_structure(self):
        """Anthropic should have 4 messages (system filtered)."""
        from mirai.agent.providers.anthropic import AnthropicProvider

        result = AnthropicProvider._convert_messages(self.CONVERSATION)
        assert len(result) == 4
        roles = [m["role"] for m in result]
        assert roles == ["user", "assistant", "user", "assistant"]

    def test_tool_call_id_preserved_both_providers(self):
        """Both providers should preserve tool_call_id across call → result."""
        from mirai.agent.providers.anthropic import AnthropicProvider
        from mirai.agent.providers.message_converter import convert_messages

        gemini = convert_messages(self.CONVERSATION)
        anthropic = AnthropicProvider._convert_messages(self.CONVERSATION)

        # Gemini: text is parts[0], functionCall is parts[1]
        fc_parts = [p for p in gemini[1]["parts"] if "functionCall" in p]
        fr_parts = [p for p in gemini[2]["parts"] if "functionResponse" in p]
        assert len(fc_parts) == 1
        assert len(fr_parts) == 1
        fc_id = fc_parts[0]["functionCall"]["id"]
        fr_id = fr_parts[0]["functionResponse"]["id"]
        assert fc_id == fr_id == "c1"

        # Anthropic: tool_use.id == tool_result.tool_use_id
        tc_blocks = [b for b in anthropic[1]["content"] if isinstance(b, dict) and b.get("type") == "tool_use"]
        tr_id = anthropic[2]["content"][0]["tool_use_id"]
        assert tc_blocks[0]["id"] == tr_id == "c1"

    def test_tool_arguments_preserved_both_providers(self):
        """Both providers should correctly parse and preserve tool arguments."""
        from mirai.agent.providers.anthropic import AnthropicProvider
        from mirai.agent.providers.message_converter import convert_messages

        gemini = convert_messages(self.CONVERSATION)
        anthropic = AnthropicProvider._convert_messages(self.CONVERSATION)

        fc_parts = [p for p in gemini[1]["parts"] if "functionCall" in p]
        gemini_args = fc_parts[0]["functionCall"]["args"]
        tc_blocks = [b for b in anthropic[1]["content"] if isinstance(b, dict) and b.get("type") == "tool_use"]
        anthropic_args = tc_blocks[0]["input"]

        assert gemini_args == anthropic_args == {"action": "check"}

    def test_user_text_preserved_both_providers(self):
        from mirai.agent.providers.anthropic import AnthropicProvider
        from mirai.agent.providers.message_converter import convert_messages

        gemini = convert_messages(self.CONVERSATION)
        anthropic = AnthropicProvider._convert_messages(self.CONVERSATION)

        gemini_text = gemini[0]["parts"][0]["text"]
        anthropic_text = anthropic[0]["content"]

        assert gemini_text == "Check status"
        assert anthropic_text == "Check status"

    def test_final_text_preserved_both_providers(self):
        from mirai.agent.providers.anthropic import AnthropicProvider
        from mirai.agent.providers.message_converter import convert_messages

        gemini = convert_messages(self.CONVERSATION)
        anthropic = AnthropicProvider._convert_messages(self.CONVERSATION)

        gemini_text = gemini[3]["parts"][0]["text"]
        anthropic_text = anthropic[3]["content"][0]["text"]

        assert gemini_text == "Everything is fine."
        assert anthropic_text == "Everything is fine."

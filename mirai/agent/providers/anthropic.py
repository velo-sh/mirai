"""Direct Anthropic API provider.

Accepts messages in OpenAI format (the internal canonical format) and
converts them to Anthropic Messages API format before sending.
"""

import json
import os
from typing import Any

import anthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from mirai.agent.models import ProviderResponse, TextBlock, ToolUseBlock
from mirai.logging import get_logger

log = get_logger("mirai.providers.anthropic")


class AnthropicProvider:
    """Direct Anthropic API provider.

    Converts OpenAI-format messages/tools to Anthropic Messages API format.
    """

    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY must be set")
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
        self.model = model

    @retry(
        retry=retry_if_exception_type(anthropic.RateLimitError),
        wait=wait_exponential(min=1, max=30),
        stop=stop_after_attempt(4),
        before_sleep=lambda rs: log.warning(
            "anthropic_rate_limited",
            attempt=rs.attempt_number,
            wait=rs.next_action.sleep,  # type: ignore[union-attr]
        ),
    )
    async def generate_response(
        self, model: str, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ProviderResponse:
        # Convert OpenAI-format messages to Anthropic format
        anthropic_messages = self._convert_messages(messages)
        anthropic_tools = self._convert_tools(tools)

        response = await self.client.messages.create(
            model=model,
            system=system,
            messages=anthropic_messages,  # type: ignore[arg-type]
            tools=anthropic_tools,  # type: ignore[arg-type]
            max_tokens=4096,
        )
        # Convert Anthropic SDK response to our Pydantic model
        content_blocks: list[TextBlock | ToolUseBlock] = []
        for block in response.content:
            if block.type == "text":
                content_blocks.append(TextBlock(text=block.text))
            elif block.type == "tool_use":
                content_blocks.append(
                    ToolUseBlock(id=block.id, name=block.name, input=block.input)  # type: ignore[arg-type]
                )
        return ProviderResponse(
            content=content_blocks, stop_reason=response.stop_reason or "end_turn", model_id=response.model
        )

    # ------------------------------------------------------------------
    # OpenAI → Anthropic format conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI-format messages to Anthropic Messages API format.

        Key differences:
        - OpenAI uses role="tool" + tool_call_id for results;
          Anthropic uses role="user" + type="tool_result" + tool_use_id
        - OpenAI puts tool_calls in a separate field;
          Anthropic puts them inline as content blocks with type="tool_use"
        - OpenAI uses role="system" (filtered out — passed separately)
        """
        result: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "")

            # System messages are passed separately to Anthropic
            if role == "system":
                continue

            if role == "tool":
                # Convert OpenAI tool result to Anthropic format
                result.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.get("tool_call_id", ""),
                            "content": msg.get("content", ""),
                        }
                    ],
                })
            elif role == "assistant":
                content_blocks: list[dict[str, Any]] = []
                # Add text content
                text = msg.get("content")
                if text:
                    content_blocks.append({"type": "text", "text": text})
                # Add tool use blocks from tool_calls
                for tc in msg.get("tool_calls", []):
                    fn = tc.get("function", {})
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "input": args,
                    })
                result.append({
                    "role": "assistant",
                    "content": content_blocks if content_blocks else "",
                })
            else:
                # user messages pass through
                result.append(msg)

        return result

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert internal tool definitions to Anthropic format.

        Internal: {"name": ..., "description": ..., "input_schema": {...}}
        Anthropic: same format (our internal format IS Anthropic format for tools)
        """
        # Tool definitions already match Anthropic format
        return tools

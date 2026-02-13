"""Direct Anthropic API provider."""

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
    """Direct Anthropic API provider."""

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
        response = await self.client.messages.create(
            model=model,
            system=system,
            messages=messages,  # type: ignore[arg-type]
            tools=tools,  # type: ignore[arg-type]
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

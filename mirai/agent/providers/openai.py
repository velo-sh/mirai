"""OpenAI-compatible LLM provider (base class).

Supports the standard OpenAI Chat Completions API. Works out-of-the-box with
any provider that exposes an OpenAI-compatible endpoint (DeepSeek, Moonshot,
Together, Groq, vLLM, Ollama, etc.) — just change ``base_url``.

Subclasses only need to set class-level defaults::

    class DeepSeekProvider(OpenAIProvider):
        DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
        DEFAULT_MODEL = "deepseek-chat"
"""

from __future__ import annotations

import json
from typing import Any

import openai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from mirai.agent.models import ProviderResponse, TextBlock, ToolUseBlock
from mirai.agent.providers.base import ModelInfo, UsageSnapshot
from mirai.logging import get_logger
from mirai.tracing import get_tracer

log = get_logger("mirai.providers.openai")


class OpenAIProvider:
    """Base provider for any OpenAI Chat Completions endpoint.

    Args:
        api_key: API key for the endpoint.
        model: Default model name.
        base_url: Custom API endpoint URL.
        max_tokens: Default max_tokens for completions.
    """

    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    DEFAULT_MODEL = "gpt-4o"

    # Subclasses can override with a static list of ModelInfo.
    # When set, list_models() returns this instead of querying the API.
    MODEL_CATALOG: list[ModelInfo] = []

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 4096,
    ):
        self.model = model or self.DEFAULT_MODEL
        self._max_tokens = max_tokens
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url or self.DEFAULT_BASE_URL,
        )

    # ------------------------------------------------------------------
    # Provider identity
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return "openai"

    # ------------------------------------------------------------------
    # Model discovery & usage
    # ------------------------------------------------------------------

    async def list_models(self) -> list[ModelInfo]:
        """List available models.

        If the subclass defines ``MODEL_CATALOG``, return that directly.
        Otherwise, query the OpenAI-compatible ``GET /models`` endpoint.
        """
        if self.MODEL_CATALOG:
            return list(self.MODEL_CATALOG)
        try:
            resp = await self.client.models.list()
            return [ModelInfo(id=m.id, name=m.id) for m in resp.data]
        except Exception as exc:
            log.warning("list_models_failed", error=str(exc))
            return [ModelInfo(id=self.model, name=self.model)]

    async def get_usage(self) -> UsageSnapshot:
        """Query provider usage / quota. Not supported by default."""
        return UsageSnapshot(provider=self.provider_name, error="not supported")

    # ------------------------------------------------------------------
    # Chat completion
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(openai.RateLimitError),
        wait=wait_exponential(min=1, max=30),
        stop=stop_after_attempt(4),
        before_sleep=lambda rs: log.warning(
            "openai_rate_limited",
            attempt=rs.attempt_number,
            wait=rs.next_action.sleep,  # type: ignore[union-attr]
        ),
    )
    async def generate_response(
        self,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ProviderResponse:
        """Send a chat completion request and return a ProviderResponse.

        Args:
            model: Model name to use.
            system: System prompt text.
            messages: Conversation messages in **OpenAI format**.
            tools: Tool definitions (``name``, ``description``, ``input_schema``).
        """
        tracer = get_tracer()
        with tracer.start_as_current_span("provider.openai.generate") as span:
            span.set_attribute("llm.model", model)

            # Build OpenAI-format messages with system prompt
            oai_messages: list[dict[str, Any]] = []
            if system:
                oai_messages.append({"role": "system", "content": system})
            oai_messages.extend(messages)

            # Convert tool definitions to OpenAI function format
            oai_tools = self._convert_tools(tools) if tools else openai.NOT_GIVEN

            log.info("api_request_sending", requested_model=model, effective_model=model)
            response = await self.client.chat.completions.create(
                model=model,
                messages=oai_messages,  # type: ignore[arg-type]
                tools=oai_tools,  # type: ignore[arg-type]
                max_tokens=self._max_tokens,
            )

            span.set_attribute("http.status_code", 200)
            return self._to_provider_response(response, model_id=model)

    # ------------------------------------------------------------------
    # Format conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert internal tool definitions to OpenAI function-calling format.

        Internal format (shared with Anthropic):
            {"name": ..., "description": ..., "input_schema": {...}}

        OpenAI format:
            {"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            }
            for t in tools
        ]

    @staticmethod
    def _to_provider_response(
        response: Any,
        model_id: str | None = None,
    ) -> ProviderResponse:
        """Convert an OpenAI ChatCompletion response to ProviderResponse."""
        choice = response.choices[0]
        message = choice.message
        content_blocks: list[TextBlock | ToolUseBlock] = []
        stop_reason = "end_turn"

        # Text content
        if message.content:
            content_blocks.append(TextBlock(text=message.content))

        # Tool calls
        if message.tool_calls:
            stop_reason = "tool_use"
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                content_blocks.append(
                    ToolUseBlock(
                        id=tc.id,
                        name=tc.function.name,
                        input=args,
                    )
                )

        # Fallback if no content at all
        if not content_blocks:
            content_blocks.append(TextBlock())

        # Map OpenAI finish_reason to our stop_reason
        if choice.finish_reason == "stop":
            stop_reason = "end_turn"
        elif choice.finish_reason == "length":
            stop_reason = "max_tokens"
        elif choice.finish_reason == "tool_calls":
            stop_reason = "tool_use"

        return ProviderResponse(
            content=content_blocks,
            stop_reason=stop_reason,
            model_id=model_id or response.model,
        )

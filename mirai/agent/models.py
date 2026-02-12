"""
Pydantic models for the Mirai agent system.

Provides typed, validated models for LLM provider responses, messages, tool
definitions, and a Protocol interface for provider implementations.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import orjson
from pydantic import BaseModel, ConfigDict

# ---------------------------------------------------------------------------
# Content blocks (returned inside a ProviderResponse)
# ---------------------------------------------------------------------------


class TextBlock(BaseModel):
    """A text content block from an LLM response."""

    model_config = ConfigDict(frozen=True)

    type: str = "text"
    text: str = ""


class ToolUseBlock(BaseModel):
    """A tool-use content block requesting tool execution."""

    model_config = ConfigDict(frozen=True)

    type: str = "tool_use"
    id: str
    name: str
    input: dict[str, Any]
    # Gemini 3 thought signature — must be preserved and replayed in history
    thought_signature: str | None = None


# Union type for all content block variants
ContentBlock = TextBlock | ToolUseBlock


# ---------------------------------------------------------------------------
# Provider response (unified across Anthropic / Antigravity / Mock)
# ---------------------------------------------------------------------------


class ProviderResponse(BaseModel):
    """Normalized response from any LLM provider."""

    model_config = ConfigDict(frozen=True)

    content: list[ContentBlock]
    stop_reason: str = "end_turn"
    model_id: str | None = None  # The actual model ID used (e.g., if failover occurred)

    def text(self) -> str:
        """Extract concatenated text from all TextBlocks."""
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))

    def tool_use_blocks(self) -> list[ToolUseBlock]:
        """Extract all ToolUseBlock items."""
        return [b for b in self.content if isinstance(b, ToolUseBlock)]


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


class Message(BaseModel):
    """A single conversation message."""

    role: str
    content: str | list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


class ToolDefinition(BaseModel):
    """Schema for an LLM-callable tool."""

    name: str
    description: str
    input_schema: dict[str, Any]


# ---------------------------------------------------------------------------
# Provider protocol (structural typing for static analysis)
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol that all LLM providers must satisfy."""

    model: str

    async def generate_response(
        self,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ProviderResponse: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def orjson_dumps(v: Any) -> str:
    """Serialize to JSON string using orjson (returns str, not bytes)."""
    return orjson.dumps(v).decode("utf-8")


def orjson_loads(v: str | bytes) -> Any:
    """Deserialize JSON using orjson."""
    return orjson.loads(v)

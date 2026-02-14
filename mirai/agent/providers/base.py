"""Provider protocol — the interface all LLM providers must satisfy.

Every provider declares three fundamental capabilities:
  1. generate_response() — run inference (existing)
  2. list_models()       — discover available models
  3. get_usage()         — query quota / usage snapshot
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from mirai.agent.models import ProviderResponse


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass
class ModelInfo:
    """Metadata for an available model.

    Fields follow industry conventions (LiteLLM, OpenRouter):
      - Identity:     id, name, description
      - Limits:       context_window, max_output_tokens
      - Capabilities: reasoning, supports_tool_use, supports_streaming,
                      supports_json_mode, supports_vision
      - Modalities:   input_modalities, output_modalities
      - Pricing:      input_price, output_price (USD per 1M tokens)
      - Lifecycle:    knowledge_cutoff, deprecation_date
    """

    id: str
    name: str
    description: str | None = None

    # --- Limits ---
    context_window: int | None = None
    max_output_tokens: int | None = None

    # --- Capabilities ---
    reasoning: bool = False
    supports_tool_use: bool = True
    supports_streaming: bool = True
    supports_json_mode: bool = False
    supports_vision: bool = False

    # --- Modalities ---
    input_modalities: list[str] = field(default_factory=lambda: ["text"])
    output_modalities: list[str] = field(default_factory=lambda: ["text"])

    # --- Pricing (USD per 1 million tokens) ---
    input_price: float | None = None
    output_price: float | None = None

    # --- Lifecycle ---
    knowledge_cutoff: str | None = None  # e.g. "2025-03"
    deprecation_date: str | None = None  # ISO-8601 date


@dataclass
class UsageSnapshot:
    """Point-in-time usage / quota snapshot for a provider."""

    provider: str = ""
    used_percent: float | None = None
    plan: str | None = None
    reset_at: str | None = None  # ISO-8601 timestamp
    error: str | None = None


# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------
@runtime_checkable
class ProviderProtocol(Protocol):
    """Structural type shared by all LLM providers.

    Every provider **must** expose:
      - ``model``          — current default model id
      - ``provider_name``  — human-readable identifier (e.g. "minimax")
      - ``generate_response()``
      - ``list_models()``
      - ``get_usage()``
    """

    model: str

    @property
    def provider_name(self) -> str: ...

    async def generate_response(
        self,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ProviderResponse: ...

    def config_dict(self) -> dict[str, Any]: ...

    async def list_models(self) -> list[ModelInfo]: ...

    async def get_usage(self) -> UsageSnapshot: ...

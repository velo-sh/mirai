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
    """Metadata for an available model."""

    id: str
    name: str
    context_window: int | None = None
    max_tokens: int | None = None
    reasoning: bool = False
    input_modalities: list[str] = field(default_factory=lambda: ["text"])


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
    ) -> ProviderResponse: ...

    async def list_models(self) -> list[ModelInfo]: ...

    async def get_usage(self) -> UsageSnapshot: ...

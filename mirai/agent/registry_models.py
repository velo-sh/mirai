"""Typed data models for the Model Registry.

Replaces the untyped ``dict[str, Any]`` previously used in
:class:`~mirai.agent.registry.ModelRegistry` with proper dataclasses
for compile-time safety and IDE navigation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mirai.agent.providers.base import ModelInfo


@dataclass
class RegistryModelEntry:
    """A single model discovered from a provider.

    Mirrors :class:`~mirai.agent.providers.base.ModelInfo` so no metadata
    is lost during the ``refresh()`` pipeline.
    """

    id: str
    name: str
    description: str | None = None
    reasoning: bool = False
    vision: bool = False

    # --- Limits ---
    context_window: int | None = None
    max_output_tokens: int | None = None

    # --- Capabilities ---
    supports_tool_use: bool = True
    supports_streaming: bool = True
    supports_json_mode: bool = False

    # --- Modalities ---
    input_modalities: list[str] = field(default_factory=lambda: ["text"])
    output_modalities: list[str] = field(default_factory=lambda: ["text"])

    # --- Pricing (USD per 1 million tokens) ---
    input_price: float | None = None
    output_price: float | None = None

    # --- Lifecycle ---
    knowledge_cutoff: str | None = None
    deprecation_date: str | None = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_model_info(cls, m: ModelInfo) -> RegistryModelEntry:
        """Create a ``RegistryModelEntry`` from a provider's ``ModelInfo``.

        This preserves *all* fields so the registry pipeline is lossless.
        """
        return cls(
            id=m.id,
            name=m.name,
            description=m.description,
            reasoning=m.reasoning,
            vision=getattr(m, "supports_vision", False),
            context_window=m.context_window,
            max_output_tokens=m.max_output_tokens,
            supports_tool_use=m.supports_tool_use,
            supports_streaming=m.supports_streaming,
            supports_json_mode=m.supports_json_mode,
            input_modalities=list(m.input_modalities),
            output_modalities=list(m.output_modalities),
            input_price=m.input_price,
            output_price=m.output_price,
            knowledge_cutoff=m.knowledge_cutoff,
            deprecation_date=m.deprecation_date,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "reasoning": self.reasoning,
            "vision": self.vision,
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "supports_tool_use": self.supports_tool_use,
            "supports_streaming": self.supports_streaming,
            "supports_json_mode": self.supports_json_mode,
            "input_modalities": self.input_modalities,
            "output_modalities": self.output_modalities,
            "input_price": self.input_price,
            "output_price": self.output_price,
            "knowledge_cutoff": self.knowledge_cutoff,
            "deprecation_date": self.deprecation_date,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegistryModelEntry:
        """Reconstruct from a plain dict (loaded from JSON).

        Backward-compatible: missing fields use safe defaults.
        """
        return cls(
            id=data["id"],
            name=data.get("name", data["id"]),
            description=data.get("description"),
            reasoning=data.get("reasoning", False),
            vision=data.get("vision", False),
            context_window=data.get("context_window"),
            max_output_tokens=data.get("max_output_tokens"),
            supports_tool_use=data.get("supports_tool_use", True),
            supports_streaming=data.get("supports_streaming", True),
            supports_json_mode=data.get("supports_json_mode", False),
            input_modalities=data.get("input_modalities", ["text"]),
            output_modalities=data.get("output_modalities", ["text"]),
            input_price=data.get("input_price"),
            output_price=data.get("output_price"),
            knowledge_cutoff=data.get("knowledge_cutoff"),
            deprecation_date=data.get("deprecation_date"),
        )


@dataclass
class RegistryProviderData:
    """Aggregated data for one provider (e.g. minimax, anthropic)."""

    available: bool = False
    env_key: str = ""
    models: list[RegistryModelEntry] = field(default_factory=list)


@dataclass
class RegistryData:
    """Top-level registry state persisted to ``~/.mirai/model_registry.json``."""

    version: int = 1
    last_refreshed: str | None = None
    active_provider: str | None = None
    active_model: str | None = None
    providers: dict[str, RegistryProviderData] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Convert to a plain dict for JSON serialization."""
        return {
            "version": self.version,
            "last_refreshed": self.last_refreshed,
            "active_provider": self.active_provider,
            "active_model": self.active_model,
            "providers": {
                pname: {
                    "available": pdata.available,
                    "env_key": pdata.env_key,
                    "models": [m.to_dict() for m in pdata.models],
                }
                for pname, pdata in self.providers.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> RegistryData:
        """Reconstruct from a plain dict (loaded from JSON)."""
        providers: dict[str, RegistryProviderData] = {}
        for pname, pdata in data.get("providers", {}).items():
            models = [RegistryModelEntry.from_dict(m) for m in pdata.get("models", [])]
            providers[pname] = RegistryProviderData(
                available=pdata.get("available", False),
                env_key=pdata.get("env_key", ""),
                models=models,
            )
        return cls(
            version=data.get("version", 1),
            last_refreshed=data.get("last_refreshed"),
            active_provider=data.get("active_provider"),
            active_model=data.get("active_model"),
            providers=providers,
        )

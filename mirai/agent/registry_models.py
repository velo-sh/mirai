"""Typed data models for the Model Registry.

Replaces the untyped ``dict[str, Any]`` previously used in
:class:`~mirai.agent.registry.ModelRegistry` with proper dataclasses
for compile-time safety and IDE navigation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RegistryModelEntry:
    """A single model discovered from a provider."""

    id: str
    name: str
    description: str | None = None
    reasoning: bool = False
    vision: bool = False

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass
class RegistryProviderData:
    """Aggregated data for one provider (e.g. minimax, anthropic)."""

    available: bool = False
    env_key: str = ""
    models: list[RegistryModelEntry] = field(default_factory=list)

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass
class RegistryData:
    """Top-level registry state persisted to ``~/.mirai/model_registry.json``."""

    version: int = 1
    last_refreshed: str | None = None
    active_provider: str | None = None
    active_model: str | None = None
    providers: dict[str, RegistryProviderData] = field(default_factory=dict)

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

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
                    "models": [
                        {
                            "id": m.id,
                            "name": m.name,
                            "description": m.description,
                            "reasoning": m.reasoning,
                            "vision": m.vision,
                        }
                        for m in pdata.models
                    ],
                }
                for pname, pdata in self.providers.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> RegistryData:
        """Reconstruct from a plain dict (loaded from JSON)."""
        providers: dict[str, RegistryProviderData] = {}
        for pname, pdata in data.get("providers", {}).items():
            models = [
                RegistryModelEntry(
                    id=m["id"],
                    name=m.get("name", m["id"]),
                    description=m.get("description"),
                    reasoning=m.get("reasoning", False),
                    vision=m.get("vision", False),
                )
                for m in pdata.get("models", [])
            ]
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

"""Embedding providers for Mirai's memory subsystem."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbedderProtocol(Protocol):
    """Structural type for embedding providers.

    Any class implementing ``get_embeddings(text) -> list[float]``
    satisfies this protocol.
    """

    async def get_embeddings(self, text: str) -> list[float]: ...


class MockEmbeddingProvider:
    """Provides consistent fake embeddings.

    Acts as a placeholder until a real embedding provider
    (e.g. text-embedding-004) is integrated.
    """

    def __init__(self, dim: int = 1536):
        self.dim = dim

    async def get_embeddings(self, text: str) -> list[float]:
        vec = [0.0] * 1536
        vec[0] = 1.0
        return vec

"""
LLM Provider implementations for Mirai.

Supports OpenAI-compatible (base), Anthropic, and Google Cloud Code Assist
(Antigravity) routing. Internal message format is OpenAI Chat Completions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mirai.agent.providers.anthropic import AnthropicProvider
from mirai.agent.providers.antigravity import AntigravityProvider, _RetryableAPIError
from mirai.agent.providers.base import ModelInfo, ProviderProtocol, UsageSnapshot
from mirai.agent.providers.embeddings import MockEmbeddingProvider
from mirai.agent.providers.factory import create_provider
from mirai.agent.providers.minimax import MiniMaxProvider
from mirai.agent.providers.openai import OpenAIProvider
from mirai.agent.providers.quota import QuotaManager

if TYPE_CHECKING:
    from tests.mocks.providers import MockProvider

__all__ = [
    "AnthropicProvider",
    "AntigravityProvider",
    "MiniMaxProvider",
    "MockEmbeddingProvider",
    "MockProvider",
    "ModelInfo",
    "OpenAIProvider",
    "ProviderProtocol",
    "QuotaManager",
    "UsageSnapshot",
    "_RetryableAPIError",
    "create_provider",
]


def __getattr__(name: str):
    """Lazy import for MockProvider to avoid test dependency in production."""
    if name == "MockProvider":
        from tests.mocks.providers import MockProvider

        return MockProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

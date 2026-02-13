"""
LLM Provider implementations for Mirai.

Supports Anthropic direct API and Google Cloud Code Assist (Antigravity) routing.
"""

from mirai.agent.providers.anthropic import AnthropicProvider
from mirai.agent.providers.antigravity import AntigravityProvider, _RetryableAPIError
from mirai.agent.providers.base import ProviderProtocol
from mirai.agent.providers.embeddings import MockEmbeddingProvider
from mirai.agent.providers.factory import create_provider
from mirai.agent.providers.quota import QuotaManager

__all__ = [
    "AnthropicProvider",
    "AntigravityProvider",
    "MockEmbeddingProvider",
    "ProviderProtocol",
    "QuotaManager",
    "_RetryableAPIError",
    "create_provider",
]


def __getattr__(name: str):
    """Lazy import MockProvider from tests to avoid test dependencies in production."""
    if name == "MockProvider":
        from tests.mocks.providers import MockProvider

        return MockProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

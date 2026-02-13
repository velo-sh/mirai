"""Provider factory — create the appropriate LLM provider based on configuration."""

import os
from typing import Any

from mirai.logging import get_logger

log = get_logger("mirai.providers.factory")


def create_provider(
    provider: str = "antigravity",
    model: str = "claude-sonnet-4-20250514",
    api_key: str | None = None,
    base_url: str | None = None,
) -> Any:
    """Create an LLM provider based on the given configuration.

    Args:
        provider: Provider type — "antigravity", "anthropic", or "openai".
        model: Default model name to use for generation.
        api_key: API key (for anthropic/openai providers).
        base_url: Custom endpoint URL (for OpenAI-compatible providers).

    Returns:
        A provider instance satisfying ProviderProtocol.
    """
    if provider == "antigravity":
        from mirai.agent.providers.antigravity import AntigravityProvider
        from mirai.auth.antigravity_auth import load_credentials

        creds = load_credentials()
        if creds:
            try:
                p = AntigravityProvider(credentials=creds, model=model)
                log.info("provider_initialized", provider="antigravity", model=model)
                return p
            except Exception as e:
                log.warning("antigravity_fallback", error=str(e))

        # Fall through to Anthropic/OpenAI if antigravity fails
        log.warning("antigravity_unavailable_trying_fallbacks")

    if provider == "anthropic" or (provider == "antigravity" and os.getenv("ANTHROPIC_API_KEY")):
        from mirai.agent.providers.anthropic import AnthropicProvider

        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if key:
            log.info("provider_initialized", provider="anthropic", model=model)
            return AnthropicProvider(api_key=key, model=model)

    if provider == "minimax":
        from mirai.agent.providers.minimax import MiniMaxProvider

        key = api_key or os.getenv("MINIMAX_API_KEY")
        if not key:
            raise ValueError(
                "MiniMax provider requires an API key. "
                "Set api_key in config or MINIMAX_API_KEY environment variable."
            )
        log.info("provider_initialized", provider="minimax", model=model, base_url=base_url)
        return MiniMaxProvider(api_key=key, model=model, base_url=base_url)

    if provider == "openai" or provider not in ("antigravity", "anthropic", "minimax"):
        from mirai.agent.providers.openai import OpenAIProvider

        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                f"Provider '{provider}' requires an API key. "
                "Set api_key in config or OPENAI_API_KEY environment variable."
            )
        log.info("provider_initialized", provider="openai", model=model, base_url=base_url)
        return OpenAIProvider(api_key=key, model=model, base_url=base_url)

    raise ValueError(
        "No API credentials available. Either:\n"
        "  1. Run `python -m mirai.auth.auth_cli` for Antigravity auth, or\n"
        "  2. Set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable, or\n"
        "  3. Configure provider and api_key in ~/.mirai/config.toml"
    )

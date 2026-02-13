"""Provider factory — auto-detect and create the best available LLM provider."""

import os
from typing import Union

from mirai.agent.providers.anthropic import AnthropicProvider
from mirai.agent.providers.antigravity import AntigravityProvider
from mirai.logging import get_logger

log = get_logger("mirai.providers.factory")


def create_provider(model: str = "claude-sonnet-4-20250514") -> Union[AnthropicProvider, AntigravityProvider]:
    """
    Auto-detect and create the best available provider.

    Priority:
    1. Antigravity credentials (~/.mirai/antigravity_credentials.json)
    2. ANTHROPIC_API_KEY environment variable

    Args:
        model: Default model name to use for generation.
    """
    # Try Antigravity first
    from mirai.auth.antigravity_auth import load_credentials

    creds = load_credentials()
    if creds:
        try:
            provider = AntigravityProvider(credentials=creds, model=model)
            log.info("provider_initialized", provider="antigravity", model=model)
            return provider
        except Exception as e:
            log.warning("antigravity_fallback", error=str(e))

    # Fall back to direct Anthropic API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        log.info("provider_initialized", provider="anthropic", model=model)
        return AnthropicProvider(api_key=api_key, model=model)

    raise ValueError(
        "No API credentials available. Either:\n"
        "  1. Run `python -m mirai.auth.auth_cli` for Antigravity auth, or\n"
        "  2. Set ANTHROPIC_API_KEY environment variable."
    )

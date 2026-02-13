"""MiniMax LLM provider — OpenAI-compatible.

MiniMax exposes a standard Chat Completions API. This provider just sets
the correct base URL and default model.

Usage in config.toml::

    [llm]
    provider = "minimax"
    api_key = "your-minimax-api-key"
    # base_url defaults to the China endpoint; override for international:
    # base_url = "https://api.minimax.io/v1"
"""

from mirai.agent.providers.openai import OpenAIProvider


class MiniMaxProvider(OpenAIProvider):
    """MiniMax provider via OpenAI-compatible Chat Completions API."""

    DEFAULT_BASE_URL = "https://api.minimaxi.com/v1"
    DEFAULT_MODEL = "MiniMax-Text-01"

"""Provider protocol — the interface all LLM providers must satisfy."""

from typing import Any, Protocol, runtime_checkable

from mirai.agent.models import ProviderResponse


@runtime_checkable
class ProviderProtocol(Protocol):
    """Structural type shared by all LLM providers."""

    model: str

    async def generate_response(
        self,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ProviderResponse: ...

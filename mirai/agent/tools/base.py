from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mirai.agent.agent_loop import AgentLoop
    from mirai.agent.providers.base import ProviderProtocol
    from mirai.agent.registry import ModelRegistry
    from mirai.config import MiraiConfig
    from mirai.db.duck import DuckDBStorage


@dataclass
class ToolContext:
    """Centralized dependencies for all Mirai tools."""

    config: MiraiConfig | None = None
    registry: ModelRegistry | None = None
    agent_loop: AgentLoop | None = None
    provider: ProviderProtocol | None = None
    storage: DuckDBStorage | None = None
    start_time: float = field(default_factory=time.monotonic)


class BaseTool(ABC):
    """Base class for all Mirai tools."""

    def __init__(self, context: ToolContext | None = None):
        self.context = context or ToolContext()

    @property
    @abstractmethod
    def definition(self) -> dict[str, Any]:
        """Returns the Anthropic tool definition JSON."""
        ...

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:  # type: ignore[override]
        """Executes the tool logic."""
        ...

    async def run(self, **kwargs: Any) -> str:
        """Wrapper to call execute() with automated tracing."""
        from opentelemetry.trace import StatusCode

        from mirai.tracing import get_tracer

        name = self.definition.get("name", self.__class__.__name__)
        tracer = get_tracer()

        with tracer.start_as_current_span(f"tool.{name}") as span:
            span.set_attribute("tool.name", name)
            span.set_attribute("tool.input", str(kwargs)[:2000])  # Cap length
            try:
                result = await self.execute(**kwargs)
                span.set_attribute("tool.output_length", len(result))
                span.set_status(StatusCode.OK)
                return result
            except Exception as exc:
                span.set_status(StatusCode.ERROR, str(exc))
                raise

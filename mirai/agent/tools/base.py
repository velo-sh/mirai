from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    @property
    @abstractmethod
    def definition(self) -> dict[str, Any]:
        """Returns the Anthropic tool definition JSON."""
        ...

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:  # type: ignore[override]
        """Executes the tool logic."""
        ...

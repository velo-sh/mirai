from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseTool(ABC):
    @property
    @abstractmethod
    def definition(self) -> Dict[str, Any]:
        """Returns the Anthropic tool definition JSON."""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> str:
        """Executes the tool logic."""
        pass

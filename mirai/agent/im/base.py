from abc import ABC, abstractmethod
from typing import Any


class BaseIMProvider(ABC):
    """Abstract interface for IM notification providers."""

    @abstractmethod
    async def send_message(self, content: str, chat_id: str | None = None) -> bool:
        """Send a message to a specific chat or default channel."""
        ...

    @abstractmethod
    async def send_card(self, card_content: dict[str, Any], chat_id: str | None = None) -> bool:
        """Send a rich card message."""
        ...

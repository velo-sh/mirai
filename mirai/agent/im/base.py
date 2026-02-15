from abc import ABC, abstractmethod
from typing import Any


class BaseIMProvider(ABC):
    """Abstract interface for IM notification providers."""

    @abstractmethod
    async def send_message(self, content: str, chat_id: str | None = None, prefer_p2p: bool = False) -> bool:
        """Send a message to a specific chat or default channel."""
        ...

    @abstractmethod
    async def send_card(self, card_content: dict[str, Any], chat_id: str | None = None) -> bool:
        """Send a rich card message."""
        ...

    async def send_markdown(self, content: str, title: str = "Mira", chat_id: str | None = None) -> bool:
        """Send a markdown-rendered message. Falls back to plain text."""
        return await self.send_message(content, chat_id=chat_id)

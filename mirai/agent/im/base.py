from abc import ABC, abstractmethod

class BaseIMProvider(ABC):
    """Abstract interface for IM notification providers."""
    
    @abstractmethod
    async def send_message(self, content: str, chat_id: str = None) -> bool:
        """Send a message to a specific chat or default channel."""
        pass

    @abstractmethod
    async def send_card(self, card_content: dict, chat_id: str = None) -> bool:
        """Send a rich card message."""
        pass

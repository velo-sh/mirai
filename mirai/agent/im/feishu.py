import lark_oapi as lark
from lark_oapi.api.im.v1 import *
import json
import logging
from typing import Optional, Dict, Any, List
from .base import BaseIMProvider

logger = logging.getLogger(__name__)


class FeishuProvider(BaseIMProvider):
    """Feishu/Lark Provider using the official lark-oapi SDK.

    Only requires APP_ID and APP_SECRET. Chat ID is auto-discovered
    by listing groups the bot has joined.
    """

    def __init__(
        self,
        app_id: Optional[str] = None,
        app_secret: Optional[str] = None,
        webhook_url: Optional[str] = None,
    ):
        self.webhook_url = webhook_url
        self.client: Optional[lark.Client] = None
        self._default_chat_id: Optional[str] = None

        if app_id and app_secret:
            self.client = (
                lark.Client.builder()
                .app_id(app_id)
                .app_secret(app_secret)
                .log_level(lark.LogLevel.INFO)
                .build()
            )

    # ------------------------------------------------------------------
    # Auto-discovery: list groups the bot has joined
    # ------------------------------------------------------------------
    async def _discover_chat_id(self) -> Optional[str]:
        """Fetch the first group chat the bot belongs to (cached)."""
        if self._default_chat_id:
            return self._default_chat_id

        if not self.client:
            return None

        try:
            request = ListChatRequest.builder().build()
            response = await self.client.im.v1.chat.alist(request)

            if not response.success():
                logger.error(
                    "Feishu chat.list failed: code=%s msg=%s",
                    response.code,
                    response.msg,
                )
                return None

            items = response.data.items if response.data else []
            if items:
                self._default_chat_id = items[0].chat_id
                logger.info(
                    "Auto-discovered Feishu chat: %s (%s)",
                    items[0].name,
                    self._default_chat_id,
                )
                return self._default_chat_id

            logger.warning("Bot has not joined any group chats yet.")
            return None

        except Exception as e:
            logger.error("Feishu chat discovery error: %s", e)
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def send_message(self, content: str, chat_id: str = None) -> bool:
        """Send a text message.

        Resolution order:
        1. Explicit chat_id argument
        2. Auto-discovered chat_id (first group the bot joined)
        3. Fallback to webhook
        """
        # --- Webhook shortcut (no client needed) ---
        if self.webhook_url and not self.client:
            return await self._send_via_webhook("text", {"text": content})

        # --- App API path ---
        if self.client:
            target = chat_id or await self._discover_chat_id()
            if target:
                return await self._send_app_message(target, "text", json.dumps({"text": content}))

            # Fall back to webhook if chat discovery failed
            if self.webhook_url:
                return await self._send_via_webhook("text", {"text": content})

        return False

    async def send_card(self, card_content: dict, chat_id: str = None) -> bool:
        """Send an interactive card message."""
        if self.webhook_url and not self.client:
            return await self._send_via_webhook("interactive", card_content)

        if self.client:
            target = chat_id or await self._discover_chat_id()
            if target:
                return await self._send_app_message(
                    target, "interactive", json.dumps(card_content)
                )

            if self.webhook_url:
                return await self._send_via_webhook("interactive", card_content)

        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    async def _send_app_message(self, chat_id: str, msg_type: str, content_json: str) -> bool:
        """Send a message via the App API."""
        try:
            request = (
                CreateMessageRequest.builder()
                .receive_id_type("chat_id")
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(chat_id)
                    .msg_type(msg_type)
                    .content(content_json)
                    .build()
                )
                .build()
            )
            response = await self.client.im.v1.message.acreate(request)
            if not response.success():
                logger.error(
                    "Feishu send error: code=%s msg=%s", response.code, response.msg
                )
                return False
            return True
        except Exception as e:
            logger.error("Feishu send exception: %s", e)
            return False

    async def _send_via_webhook(self, msg_type: str, content: Any) -> bool:
        """POST to a Feishu custom bot webhook."""
        import httpx

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    self.webhook_url,
                    json={"msg_type": msg_type, "content": content},
                )
                return resp.status_code == 200
        except Exception as e:
            logger.error("Feishu webhook exception: %s", e)
            return False

from typing import Any

import lark_oapi as lark
import orjson
from lark_oapi.api.im.v1 import *  # noqa: F403  # type: ignore[import-untyped,attr-defined]

from mirai.logging import get_logger

from .base import BaseIMProvider

log = get_logger("mirai.feishu")


class FeishuProvider(BaseIMProvider):
    """Feishu/Lark Provider using the official lark-oapi SDK.

    Only requires APP_ID and APP_SECRET. Chat ID is auto-discovered
    by listing groups the bot has joined.
    """

    def __init__(
        self,
        app_id: str | None = None,
        app_secret: str | None = None,
        webhook_url: str | None = None,
    ):
        self.webhook_url = webhook_url
        self.client: lark.Client | None = None
        self._default_chat_id: str | None = None

        if app_id and app_secret:
            self.client = (
                lark.Client.builder().app_id(app_id).app_secret(app_secret).log_level(lark.LogLevel.INFO).build()
            )

    # ------------------------------------------------------------------
    # Auto-discovery: list groups the bot has joined
    # ------------------------------------------------------------------
    async def _discover_chat_id(self) -> str | None:
        """Fetch the first group chat the bot belongs to (cached)."""
        if self._default_chat_id:
            return self._default_chat_id

        if not self.client:
            return None

        try:
            request = ListChatRequest.builder().build()  # type: ignore[name-defined]  # noqa: F405
            response = await self.client.im.v1.chat.alist(request)

            if not response.success():
                log.error("feishu_chat_list_failed", code=response.code, msg=response.msg)
                return None

            items = response.data.items if response.data else []
            if items:
                self._default_chat_id = items[0].chat_id
                log.info("feishu_chat_discovered", name=items[0].name, chat_id=self._default_chat_id)
                return self._default_chat_id

            log.warning("feishu_no_chats")
            return None

        except Exception as e:
            log.error("feishu_discovery_error", error=str(e))
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
                return await self._send_app_message(target, "text", orjson.dumps({"text": content}).decode())

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
                return await self._send_app_message(target, "interactive", orjson.dumps(card_content).decode())

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
                CreateMessageRequest.builder()  # type: ignore[name-defined]  # noqa: F405
                .receive_id_type("chat_id")
                .request_body(
                    CreateMessageRequestBody.builder()  # type: ignore[name-defined]  # noqa: F405
                    .receive_id(chat_id)
                    .msg_type(msg_type)
                    .content(content_json)
                    .build()
                )
                .build()
            )
            response = await self.client.im.v1.message.acreate(request)  # type: ignore[union-attr]
            if not response.success():
                log.error("feishu_send_error", code=response.code, msg=response.msg)
                return False
            return True
        except Exception as e:
            log.error("feishu_send_exception", error=str(e))
            return False

    async def _send_via_webhook(self, msg_type: str, content: Any) -> bool:
        """POST to a Feishu custom bot webhook."""
        import httpx

        if not hasattr(self, "_webhook_http"):
            self._webhook_http = httpx.AsyncClient(timeout=10.0, http2=True)

        try:
            assert self.webhook_url is not None
            resp = await self._webhook_http.post(
                self.webhook_url,
                json={"msg_type": msg_type, "content": content},
            )
            return resp.status_code == 200
        except Exception as e:
            log.error("feishu_webhook_error", error=str(e))
            return False

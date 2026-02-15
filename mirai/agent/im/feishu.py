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
        self._webhook_http: Any = None  # lazy httpx.AsyncClient

    # ------------------------------------------------------------------
    # Auto-discovery: list groups the bot has joined
    # ------------------------------------------------------------------
    async def _discover_chat_id(self, prefer_p2p: bool = False) -> str | None:
        """Fetch the first group chat or p2p chat the bot belongs to (cached)."""
        if self._default_chat_id and not prefer_p2p:
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
            if not items:
                log.warning("feishu_no_chats")
                return None

            # Prioritization logic:
            # 1. If prefer_p2p is True, look for p2p chats first
            if prefer_p2p:
                p2p_chats = [
                    c for c in items if getattr(c, "chat_mode", None) == "p2p" or getattr(c, "type", None) == "p2p"
                ]
                if p2p_chats:
                    chat_id = p2p_chats[0].chat_id
                    log.info("feishu_p2p_chat_discovered", chat_id=chat_id)
                    return str(chat_id) if chat_id else None

                # Attempt discovery of private window via owner_id if it looks like an open_id
                for c in items:
                    if getattr(c, "owner_id", "").startswith("ou_"):
                        target = c.owner_id
                        log.info("feishu_p2p_owner_id_selected_as_target", open_id=target)
                        return str(target) if target else None

                log.info("feishu_p2p_not_found_falling_back_to_first_available")

            # Fallback to the first item (default behavior)
            self._default_chat_id = items[0].chat_id
            log.info("feishu_chat_discovered", name=items[0].name, chat_id=self._default_chat_id)
            return str(self._default_chat_id) if self._default_chat_id else None

        except Exception as e:
            log.error("feishu_discovery_error", error=str(e))
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def send_message(self, content: str, chat_id: str = None, prefer_p2p: bool = False) -> bool:
        """Send a text message.

        Resolution order:
        1. Explicit chat_id argument
        2. Auto-discovered chat_id (p2p or group)
        3. Fallback to webhook
        """
        # --- Webhook shortcut (no client needed) ---
        if self.webhook_url and not self.client:
            return await self._send_via_webhook("text", {"text": content})

        # --- App API path ---
        if self.client:
            target = chat_id or await self._discover_chat_id(prefer_p2p=prefer_p2p)
            if target:
                return await self._send_app_message(target, "text", orjson.dumps({"text": content}).decode())

            # Fall back to webhook if chat discovery failed
            if self.webhook_url:
                return await self._send_via_webhook("text", {"text": content})

        return False

    async def send_card(self, card_content: dict, chat_id: str = None, prefer_p2p: bool = False) -> bool:
        """Send an interactive card message."""
        if self.webhook_url and not self.client:
            return await self._send_via_webhook("interactive", card_content)

        if self.client:
            target = chat_id or await self._discover_chat_id(prefer_p2p=prefer_p2p)
            if target:
                return await self._send_app_message(target, "interactive", orjson.dumps(card_content).decode())

            if self.webhook_url:
                return await self._send_via_webhook("interactive", card_content)

        return False

    async def send_markdown(
        self,
        content: str,
        title: str = "Mira",
        chat_id: str | None = None,
        prefer_p2p: bool = False,
        color: str = "blue",
    ) -> bool:
        """Send a markdown-rendered message via interactive card.

        Feishu's card markdown element supports:
          **bold**, *italic*, ~~strikethrough~~, [link](url),
          `inline code`, ```code blocks```, - lists, 1. ordered lists

        Unsupported (auto-stripped): # headings (use title param),
          > blockquotes, tables, images (use send_card for those).

        Args:
            content: Markdown text to render.
            title: Card header title.
            chat_id: Target chat (auto-discovered if None).
            prefer_p2p: Prefer private chat.
            color: Header color template (blue, green, red, yellow, etc.).
        """
        # Feishu markdown element has a ~4000 char limit per element;
        # chunk long content into multiple elements.
        MAX_CHUNK = 3800
        elements: list[dict[str, Any]] = []
        remaining = content
        while remaining:
            chunk = remaining[:MAX_CHUNK]
            remaining = remaining[MAX_CHUNK:]
            elements.append(
                {
                    "tag": "div",
                    "text": {"tag": "lark_md", "content": chunk},
                }
            )

        card: dict[str, Any] = {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": title},
                "template": color,
            },
            "elements": elements,
        }

        return await self.send_card(card, chat_id=chat_id, prefer_p2p=prefer_p2p)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    async def _send_app_message(self, receive_id: str, msg_type: str, content_json: str) -> bool:
        """Send a message via the App API."""
        try:
            # Detect receive_id type
            # oc_ = chat_id, ou_ = open_id, us_ = user_id
            receive_id_type = "chat_id"
            if receive_id.startswith("ou_"):
                receive_id_type = "open_id"
            elif receive_id.startswith("us_"):
                receive_id_type = "user_id"

            request = (
                CreateMessageRequest.builder()  # type: ignore[name-defined]  # noqa: F405
                .receive_id_type(receive_id_type)
                .request_body(
                    CreateMessageRequestBody.builder()  # type: ignore[name-defined]  # noqa: F405
                    .receive_id(receive_id)
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

        if self._webhook_http is None:
            self._webhook_http = httpx.AsyncClient(timeout=10.0, http2=True)

        try:
            assert self.webhook_url is not None
            resp = await self._webhook_http.post(
                self.webhook_url,
                json={"msg_type": msg_type, "content": content},
            )
            return bool(resp.status_code == 200)
        except Exception as e:
            log.error("feishu_webhook_error", error=str(e))
            return False

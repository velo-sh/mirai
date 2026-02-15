"""Feishu Event Receiver: WebSocket-based message handler.

Connects to Feishu via WebSocket long-connection (no public URL needed).
Receives private/group messages and routes them to AgentLoop for processing.
Maintains per-chat conversation history for multi-turn context.
"""

import asyncio
import base64
import threading
from collections.abc import Awaitable, Callable
from typing import Any, cast

import lark_oapi as lark
import orjson
from lark_oapi.api.im.v1 import (
    CreateMessageReactionRequest,
    CreateMessageReactionRequestBody,
    Emoji,
    GetMessageResourceRequest,
    P2ImMessageReceiveV1,
    ReplyMessageRequest,
    ReplyMessageRequestBody,
)

from mirai.db.models import FeishuMessage
from mirai.logging import get_logger

log = get_logger("mirai.feishu_receiver")


class FeishuEventReceiver:
    """Receives Feishu messages via WebSocket and replies via AgentLoop."""

    def __init__(
        self,
        app_id: str,
        app_secret: str,
        message_handler: Callable[[str, str | list[dict[str, Any]], str, list[dict]], Awaitable[str]],
        storage: Any = None,
        encrypt_key: str = "",
        verification_token: str = "",
    ):
        """
        Args:
            app_id: Feishu App ID
            app_secret: Feishu App Secret
            message_handler: async callback(sender_id, message_text, chat_id, history) -> reply_text
            encrypt_key: Optional encryption key from Feishu console
            verification_token: Optional verification token from Feishu console
        """
        self._app_id = app_id
        self._app_secret = app_secret
        self._message_handler = message_handler
        self._storage = storage
        self._encrypt_key = encrypt_key
        self._verification_token = verification_token
        self._ws_client: lark.ws.Client | None = None
        self._reply_client: lark.Client | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        # Per-chat conversation history: chat_id -> [(role, content), ...]
        # Keeps the last MAX_HISTORY_TURNS exchanges for multi-turn context.
        self._conversations: dict[str, list[dict[str, str]]] = {}
        self.MAX_HISTORY_TURNS = 20  # max user+assistant message pairs

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Start the WebSocket receiver in a background thread.

        Args:
            loop: The main asyncio event loop (for scheduling async replies).
        """
        self._loop = loop

        # Build a standard API client for sending replies
        self._reply_client = (
            lark.Client.builder()
            .app_id(self._app_id)
            .app_secret(self._app_secret)
            .log_level(lark.LogLevel.INFO)
            .build()
        )

        # Build event dispatcher
        event_handler = (
            lark.EventDispatcherHandler.builder(self._encrypt_key, self._verification_token)
            .register_p2_im_message_receive_v1(self._on_message_received)
            .register_p2_im_message_reaction_created_v1(lambda data: None)  # ignore reaction events
            .register_p2_im_message_reaction_deleted_v1(lambda data: None)
            .register_p2_im_message_message_read_v1(lambda data: None)
            .build()
        )

        # Build WebSocket client
        self._ws_client = lark.ws.Client(
            app_id=self._app_id,
            app_secret=self._app_secret,
            event_handler=event_handler,
            log_level=lark.LogLevel.INFO,
        )

        # Run WebSocket in a daemon thread with its OWN event loop.
        # The SDK has a module-level `loop = asyncio.get_event_loop()` which
        # grabs uvicorn's running loop at import time. We monkey-patch that
        # variable to a fresh loop so `run_until_complete()` works.
        def _run_ws():
            import lark_oapi.ws.client as ws_mod

            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            ws_mod.loop = new_loop  # patch the SDK's module-level loop
            self._ws_client.start()  # type: ignore[union-attr]

        thread = threading.Thread(target=_run_ws, daemon=True, name="feishu-ws")
        thread.start()
        log.info("feishu_ws_started")

    def _on_message_received(self, data: P2ImMessageReceiveV1) -> None:
        """Synchronous callback from the SDK. Schedules async processing."""
        try:
            event = data.event
            message = event.message
            sender = event.sender

            # Extract content based on message type
            msg_type = message.message_type
            sender_id = sender.sender_id.open_id if sender.sender_id else "unknown"
            message_id = message.message_id
            chat_id = message.chat_id

            content_dict = orjson.loads(message.content)

            if msg_type == "text":
                text = content_dict.get("text", "").strip()
                if not text:
                    return
                message_content: str | list[dict[str, Any]] = text
                log.info(
                    "feishu_msg_received",
                    sender=sender_id,
                    text=text[:50],
                    msg_id=message_id,
                    chat_id=chat_id,
                    msg_type=msg_type,
                )
            elif msg_type == "image":
                image_key = content_dict.get("image_key")
                if not image_key:
                    return

                # Create multi-part message with text placeholder and image
                # We'll download the image in the background processing
                message_content = [
                    {"type": "text", "text": "[Image message]"},
                    {"type": "image", "image_key": image_key, "msg_id": message_id},
                ]
                log.info("feishu_image_received", sender=sender_id, image_key=image_key, msg_id=message_id)
            elif msg_type == "post":
                # Feishu "post" is a rich text structure
                # content: {"title": "", "content": [[{"tag": "text", "text": "..."}, ...], ...]}
                post_content = content_dict.get("content", [])
                text_parts = []
                for paragraph in post_content:
                    for element in paragraph:
                        if element.get("tag") == "text":
                            text_parts.append(element.get("text", ""))
                        elif element.get("tag") == "a":
                            text_parts.append(element.get("text", ""))  # Link text
                        elif element.get("tag") == "at":
                            text_parts.append(element.get("at_name", ""))  # @Name
                    text_parts.append("\n")  # Paragraph break

                text = "".join(text_parts).strip()
                if not text:
                    return
                message_content = text
                log.info("feishu_post_received", sender=sender_id, text=text[:50], msg_id=message_id)
            else:
                log.info("feishu_ignore_msg_type", msg_type=msg_type)
                return

            # Schedule the async handler on the main event loop
            if self._loop and self._loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self._process_and_reply(message_content, sender_id, message_id, chat_id),
                    self._loop,
                )
        except Exception as e:
            log.error("feishu_msg_process_error", error=str(e), exc_info=True)

    async def _send_thinking_reaction(self, message_id: str) -> None:
        """Send a 'thinking' emoji reaction to indicate processing."""
        reaction_request = (
            CreateMessageReactionRequest.builder()
            .message_id(message_id)
            .request_body(
                CreateMessageReactionRequestBody.builder()
                .reaction_type(Emoji.builder().emoji_type("THINKING").build())
                .build()
            )
            .build()
        )
        reaction_resp = await self._reply_client.im.v1.message_reaction.acreate(reaction_request)  # type: ignore[union-attr]
        if reaction_resp.success():
            log.info("feishu_thinking_reaction_sent", msg_id=message_id)
        else:
            log.error("feishu_reaction_failed", code=reaction_resp.code, msg=reaction_resp.msg)

    async def _load_history(self, chat_id: str) -> list[dict[str, str]]:
        """Load or initialize conversation history for a chat."""
        history = self._conversations.get(chat_id)
        if history is not None:
            return history

        if self._storage:
            messages = await self._storage.get_feishu_history(chat_id, limit=self.MAX_HISTORY_TURNS * 2)
            history = [{"role": m.role, "content": m.content} for m in messages]
            self._conversations[chat_id] = history
            log.info("history_loaded_from_storage", chat_id=chat_id, turns=len(history) // 2)
        else:
            history = []
            self._conversations[chat_id] = history
        return history

    async def _process_image_blocks(self, message_content: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Download image resources and convert blocks to base64 format."""
        new_content = []
        for block in message_content:
            if block.get("type") == "image":
                image_key = block["image_key"]
                msg_id = block["msg_id"]
                log.info("downloading_image", image_key=image_key)
                image_data = await self._download_image(msg_id, image_key)
                if image_data:
                    new_content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64.b64encode(image_data).decode(),
                            },
                        }
                    )
                else:
                    new_content.append({"type": "text", "text": "[Image download failed]"})
            else:
                new_content.append(block)
        return new_content

    async def _send_reply_card(self, message_id: str, reply_text: str) -> None:
        """Send a Feishu interactive card reply."""
        card_body = {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": "Mira"},
                "template": "blue",
            },
            "elements": [
                {
                    "tag": "div",
                    "text": {"tag": "lark_md", "content": reply_text},
                }
            ],
        }
        reply_request = (
            ReplyMessageRequest.builder()
            .message_id(message_id)
            .request_body(
                ReplyMessageRequestBody.builder()
                .msg_type("interactive")
                .content(orjson.dumps(card_body).decode())
                .build()
            )
            .build()
        )
        reply_resp = await self._reply_client.im.v1.message.areply(reply_request)  # type: ignore[union-attr]
        if reply_resp.success():
            log.info("feishu_reply_sent", msg_id=message_id)
        else:
            log.error("feishu_reply_failed", code=reply_resp.code, msg=reply_resp.msg)

    async def _process_and_reply(
        self, message_content: str | list[dict[str, Any]], sender_id: str, message_id: str, chat_id: str
    ) -> None:
        """Process the message through AgentLoop and reply.

        Flow:
        1. Immediately reply with "Thinking..." placeholder (typing indicator)
        2. Process the message through AgentLoop with conversation history
        3. Send the real response as a new reply
        4. Record the exchange in conversation history
        """
        try:
            await self._send_thinking_reaction(message_id)

            history = await self._load_history(chat_id)

            # Process image blocks if multimodal message
            processed_content: str | list[dict[str, Any]] = message_content
            if isinstance(message_content, list):
                processed_content = await self._process_image_blocks(message_content)

            reply_text = await self._message_handler(sender_id, processed_content, chat_id, history)
            if not reply_text:
                reply_text = "I received your message but couldn't generate a response."

            await self._send_reply_card(message_id, reply_text)
            await self._record_exchange(chat_id, processed_content, reply_text)

        except Exception as e:
            log.error("feishu_reply_error", error=str(e), exc_info=True)

    async def _download_image(self, message_id: str, image_key: str) -> bytes | None:
        """Download image resource from Feishu with robust async handling."""
        try:
            request = (
                GetMessageResourceRequest.builder().message_id(message_id).file_key(image_key).type("image").build()
            )
            response = await self._reply_client.im.v1.message_resource.aget(request)  # type: ignore[union-attr]

            if not response.success():
                log.error("feishu_image_download_failed", code=response.code, msg=response.msg, msg_id=message_id)
                return None

            # lark-oapi stores the binary content in response.file (as a BytesIO)
            # or in response.raw.content (as bytes)
            if response.file:
                # Ensure we read from the beginning
                response.file.seek(0)
                return cast(bytes, response.file.read())

            if hasattr(response.raw, "content") and response.raw.content:
                return cast(bytes, response.raw.content)

            log.warning("feishu_image_download_empty", msg_id=message_id)
            return None

        except Exception as e:
            log.error("feishu_image_download_error", error=str(e), msg_id=message_id)
            return None

    async def _record_exchange(self, chat_id: str, user_msg: str | list[dict[str, Any]], assistant_msg: str) -> None:
        """Append a user/assistant exchange to the chat's conversation history.

        Maintains a rolling window of MAX_HISTORY_TURNS messages to prevent
        unbounded memory growth and context window overflow.
        """
        if chat_id not in self._conversations:
            self._conversations[chat_id] = []

        history = self._conversations[chat_id]

        # Convert list content to string for L3 storage if needed
        # (Though current L3 implementation handles strings)
        user_msg_str = str(user_msg)

        history.append({"role": "user", "content": user_msg_str})
        history.append({"role": "assistant", "content": assistant_msg})

        # Trim history
        if len(history) > self.MAX_HISTORY_TURNS * 2:
            self._conversations[chat_id] = history[-(self.MAX_HISTORY_TURNS * 2) :]

        # Also persist to L3 storage
        if self._storage:
            try:
                await self._storage.save_feishu_history(
                    FeishuMessage(chat_id=chat_id, role="user", content=user_msg_str)
                )
                await self._storage.save_feishu_history(
                    FeishuMessage(chat_id=chat_id, role="assistant", content=assistant_msg)
                )
            except Exception as e:
                log.error("history_save_failed", chat_id=chat_id, error=str(e))

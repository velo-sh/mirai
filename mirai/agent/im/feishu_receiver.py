"""Feishu Event Receiver: WebSocket-based message handler.

Connects to Feishu via WebSocket long-connection (no public URL needed).
Receives private/group messages and routes them to AgentLoop for processing.
Maintains per-chat conversation history for multi-turn context.
"""

import asyncio
import threading
from collections.abc import Awaitable, Callable
from typing import Any

import lark_oapi as lark
import orjson
from lark_oapi.api.im.v1 import (
    P2ImMessageReceiveV1,
    ReplyMessageRequest,
    ReplyMessageRequestBody,
)

from mirai.logging import get_logger

log = get_logger("mirai.feishu_receiver")


class FeishuEventReceiver:
    """Receives Feishu messages via WebSocket and replies via AgentLoop."""

    def __init__(
        self,
        app_id: str,
        app_secret: str,
        message_handler: Callable[[str, str, str, list[dict]], Awaitable[str]],
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

            # Extract text content
            msg_type = message.message_type
            if msg_type != "text":
                log.info("feishu_ignore_nontext", msg_type=msg_type)
                return

            content = orjson.loads(message.content)
            text = content.get("text", "").strip()
            if not text:
                return

            sender_id = sender.sender_id.open_id if sender.sender_id else "unknown"
            message_id = message.message_id
            chat_id = message.chat_id

            log.info("feishu_msg_received", sender=sender_id, text=text[:50], msg_id=message_id)

            # Schedule the async handler on the main event loop
            if self._loop and self._loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self._process_and_reply(text, sender_id, message_id, chat_id),
                    self._loop,
                )
        except Exception as e:
            log.error("feishu_msg_process_error", error=str(e), exc_info=True)

    async def _process_and_reply(self, text: str, sender_id: str, message_id: str, chat_id: str) -> None:
        """Process the message through AgentLoop and reply.

        Flow:
        1. Immediately reply with "Thinking..." placeholder (typing indicator)
        2. Process the message through AgentLoop with conversation history
        3. Send the real response as a new reply
        4. Record the exchange in conversation history
        """
        try:
            # Step 1: Send "Thinking..." placeholder immediately
            placeholder_request = (
                ReplyMessageRequest.builder()
                .message_id(message_id)
                .request_body(
                    ReplyMessageRequestBody.builder()
                    .msg_type("text")
                    .content(orjson.dumps({"text": "🤔 Thinking..."}).decode())
                    .build()
                )
                .build()
            )
            placeholder_resp = await self._reply_client.im.v1.message.areply(placeholder_request)  # type: ignore[union-attr]
            if placeholder_resp.success():
                log.info("feishu_typing_sent", msg_id=message_id)
            else:
                log.error("feishu_typing_failed", code=placeholder_resp.code, msg=placeholder_resp.msg)

            # Step 2: Get conversation history for this chat
            history = self._conversations.get(chat_id)
            if history is None:
                # Cache miss - try to load from persistent storage
                if self._storage:
                    history = await self._storage.get_feishu_history(chat_id, limit=self.MAX_HISTORY_TURNS * 2)
                    self._conversations[chat_id] = history
                    log.info("history_loaded_from_storage", chat_id=chat_id, turns=len(history) // 2)
                else:
                    history = []
                    self._conversations[chat_id] = history

            log.info("conversation_context", chat_id=chat_id, history_turns=len(history) // 2)

            # Step 3: Process the message through AgentLoop with history
            reply_text = await self._message_handler(sender_id, text, chat_id, history)

            if not reply_text:
                reply_text = "I received your message but couldn't generate a response."

            # Step 4: Send the real response as a new reply
            reply_request = (
                ReplyMessageRequest.builder()
                .message_id(message_id)
                .request_body(
                    ReplyMessageRequestBody.builder()
                    .msg_type("text")
                    .content(orjson.dumps({"text": reply_text}).decode())
                    .build()
                )
                .build()
            )
            reply_resp = await self._reply_client.im.v1.message.areply(reply_request)  # type: ignore[union-attr]
            if reply_resp.success():
                log.info("feishu_reply_sent", msg_id=message_id)
            else:
                log.error("feishu_reply_failed", code=reply_resp.code, msg=reply_resp.msg)

            # Step 5: Record this exchange in conversation history
            await self._record_exchange(chat_id, text, reply_text)

        except Exception as e:
            log.error("feishu_reply_error", error=str(e), exc_info=True)

    async def _record_exchange(self, chat_id: str, user_msg: str, assistant_msg: str) -> None:
        """Append a user/assistant exchange to the chat's conversation history.

        Maintains a rolling window of MAX_HISTORY_TURNS messages to prevent
        unbounded memory growth and context window overflow.
        """
        if chat_id not in self._conversations:
            self._conversations[chat_id] = []

        history = self._conversations[chat_id]
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": assistant_msg})

        # Save to persistent storage if available
        if self._storage:
            try:
                await self._storage.save_feishu_history(chat_id, "user", user_msg)
                await self._storage.save_feishu_history(chat_id, "assistant", assistant_msg)
            except Exception as e:
                log.error("history_save_failed", chat_id=chat_id, error=str(e))

        # Trim to max turns (each turn = 2 messages: user + assistant)
        max_messages = self.MAX_HISTORY_TURNS * 2
        if len(history) > max_messages:
            self._conversations[chat_id] = history[-max_messages:]

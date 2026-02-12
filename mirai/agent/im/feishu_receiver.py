"""Feishu Event Receiver: WebSocket-based message handler.

Connects to Feishu via WebSocket long-connection (no public URL needed).
Receives private/group messages and routes them to AgentLoop for processing.
"""

import asyncio
import json
import logging
import threading
from collections.abc import Awaitable, Callable

import lark_oapi as lark
from lark_oapi.api.im.v1 import (
    P2ImMessageReceiveV1,
    PatchMessageRequest,
    PatchMessageRequestBody,
    ReplyMessageRequest,
    ReplyMessageRequestBody,
)

logger = logging.getLogger(__name__)


class FeishuEventReceiver:
    """Receives Feishu messages via WebSocket and replies via AgentLoop."""

    def __init__(
        self,
        app_id: str,
        app_secret: str,
        message_handler: Callable[[str, str, str], Awaitable[str]],
        encrypt_key: str = "",
        verification_token: str = "",
    ):
        """
        Args:
            app_id: Feishu App ID
            app_secret: Feishu App Secret
            message_handler: async callback(sender_id, message_text, chat_id) -> reply_text
            encrypt_key: Optional encryption key from Feishu console
            verification_token: Optional verification token from Feishu console
        """
        self._app_id = app_id
        self._app_secret = app_secret
        self._message_handler = message_handler
        self._encrypt_key = encrypt_key
        self._verification_token = verification_token
        self._ws_client: lark.ws.Client | None = None
        self._reply_client: lark.Client | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

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
        logger.info("Feishu WebSocket receiver started (background thread).")

    def _on_message_received(self, data: P2ImMessageReceiveV1) -> None:
        """Synchronous callback from the SDK. Schedules async processing."""
        try:
            event = data.event
            message = event.message
            sender = event.sender

            # Extract text content
            msg_type = message.message_type
            if msg_type != "text":
                logger.info("Ignoring non-text message type: %s", msg_type)
                return

            content = json.loads(message.content)
            text = content.get("text", "").strip()
            if not text:
                return

            sender_id = sender.sender_id.open_id if sender.sender_id else "unknown"
            message_id = message.message_id
            chat_id = message.chat_id

            logger.info(
                "Received message from %s: %s (msg_id=%s)",
                sender_id,
                text[:50],
                message_id,
            )

            # Schedule the async handler on the main event loop
            if self._loop and self._loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self._process_and_reply(text, sender_id, message_id, chat_id),
                    self._loop,
                )
        except Exception as e:
            logger.error("Error processing Feishu message: %s", e, exc_info=True)

    async def _process_and_reply(self, text: str, sender_id: str, message_id: str, chat_id: str) -> None:
        """Process the message through AgentLoop and reply.

        Flow:
        1. Immediately reply with "Thinking..." placeholder (typing indicator)
        2. Process the message through AgentLoop
        3. Patch the placeholder with the real response
        """
        placeholder_msg_id = None

        try:
            # Step 1: Send "Thinking..." placeholder immediately
            placeholder_request = (
                ReplyMessageRequest.builder()
                .message_id(message_id)
                .request_body(
                    ReplyMessageRequestBody.builder()
                    .msg_type("text")
                    .content(json.dumps({"text": "🤔 Thinking..."}))
                    .build()
                )
                .build()
            )
            placeholder_resp = await self._reply_client.im.v1.message.areply(placeholder_request)  # type: ignore[union-attr]
            if placeholder_resp.success():
                placeholder_msg_id = placeholder_resp.data.message_id
                logger.info("Sent typing indicator for msg %s", message_id)
            else:
                logger.error(
                    "Failed to send typing indicator: code=%s msg=%s",
                    placeholder_resp.code,
                    placeholder_resp.msg,
                )

            # Step 2: Process the message through AgentLoop
            reply_text = await self._message_handler(sender_id, text, chat_id)

            if not reply_text:
                reply_text = "I received your message but couldn't generate a response."

            # Step 3: Patch the placeholder with the real response
            if placeholder_msg_id:
                patch_request = (
                    PatchMessageRequest.builder()
                    .message_id(placeholder_msg_id)
                    .request_body(PatchMessageRequestBody.builder().content(json.dumps({"text": reply_text})).build())
                    .build()
                )
                patch_resp = await self._reply_client.im.v1.message.apatch(patch_request)  # type: ignore[union-attr]
                if not patch_resp.success():
                    logger.error(
                        "Failed to patch message: code=%s msg=%s",
                        patch_resp.code,
                        patch_resp.msg,
                    )
                else:
                    logger.info("Patched reply for message %s.", message_id)
            else:
                # Fallback: send a new reply if placeholder failed
                fallback_request = (
                    ReplyMessageRequest.builder()
                    .message_id(message_id)
                    .request_body(
                        ReplyMessageRequestBody.builder()
                        .msg_type("text")
                        .content(json.dumps({"text": reply_text}))
                        .build()
                    )
                    .build()
                )
                await self._reply_client.im.v1.message.areply(fallback_request)  # type: ignore[union-attr]

        except Exception as e:
            logger.error("Error replying to message: %s", e, exc_info=True)

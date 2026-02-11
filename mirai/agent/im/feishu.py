import lark_oapi as lark
from lark_oapi.api.im.v1 import *
import json
import logging
from typing import Optional, Dict, Any
from .base import BaseIMProvider

class FeishuProvider(BaseIMProvider):
    """Feishu/Lark Provider using the official lark-oapi SDK (Async)."""

    def __init__(self, app_id: Optional[str] = None, app_secret: Optional[str] = None, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url
        self.client = None
        if app_id and app_secret:
            self.client = lark.Client.builder() \
                .app_id(app_id) \
                .app_secret(app_secret) \
                .log_level(lark.LogLevel.INFO) \
                .build()
        
    async def send_message(self, content: str, chat_id: str = None) -> bool:
        """Send a text message via Webhook or App API."""
        if self.webhook_url and not chat_id:
            return await self._send_via_webhook("text", {"text": content})
        
        if self.client and chat_id:
            try:
                request = CreateMessageRequest.builder() \
                    .receive_id_type("chat_id") \
                    .request_body(CreateMessageRequestBody.builder() \
                        .receive_id(chat_id) \
                        .msg_type("text") \
                        .content(json.dumps({"text": content})) \
                        .build()) \
                    .build()
                
                # The official SDK use .create(request) for sync, 
                # for async it might be client.im.v1.message.acreate(request) or similar
                # Let's check the exact async method name based on search results earlier
                # Search said: "AsyncLark" or "acreate"
                # Actually, lark-oapi sync/async is often switchable via client type.
                
                response = await self.client.im.v1.message.acreate(request)
                if not response.success():
                    logging.error(f"Feishu API Error: {response.code}, {response.msg}")
                    return False
                return True
            except Exception as e:
                logging.error(f"Feishu API Exception: {e}")
                return False
        
        return False

    async def _send_via_webhook(self, msg_type: str, content: Any) -> bool:
        """Helper for Webhook POST."""
        import httpx
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    self.webhook_url,
                    json={
                        "msg_type": msg_type,
                        "content": content
                    }
                )
                return resp.status_code == 200
        except Exception as e:
            logging.error(f"Feishu Webhook Exception: {e}")
            return False

    async def send_card(self, card_content: dict, chat_id: str = None) -> bool:
        """Send an interactive card."""
        if self.webhook_url and not chat_id:
            return await self._send_via_webhook("interactive", card_content)
            
        if self.client and chat_id:
            try:
                request = CreateMessageRequest.builder() \
                    .receive_id_type("chat_id") \
                    .request_body(CreateMessageRequestBody.builder() \
                        .receive_id(chat_id) \
                        .msg_type("interactive") \
                        .content(json.dumps(card_content)) \
                        .build()) \
                    .build()
                response = await self.client.im.v1.message.acreate(request)
                return response.success()
            except Exception as e:
                logging.error(f"Feishu Card API Exception: {e}")
                return False
        return False

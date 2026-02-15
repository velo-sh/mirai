"""IM (Instant Messaging) tool — lets the agent send messages to the user via Feishu."""

from typing import Any

from mirai.agent.tools.base import BaseTool
from mirai.logging import get_logger

log = get_logger("mirai.tools.im")


class IMTool(BaseTool):
    """Send messages to the user via the configured IM provider (Feishu)."""

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "name": "im_tool",
            "description": (
                "Send a message to the user via Feishu. "
                "Use this when you need to proactively communicate with the user, "
                "for example when woken by a cron job."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["send_message"],
                        "description": "The action to perform.",
                    },
                    "message": {
                        "type": "string",
                        "description": "The message content to send to the user.",
                    },
                },
                "required": ["action", "message"],
            },
        }

    async def execute(self, action: str = "send_message", message: str = "", **_: Any) -> str:
        im = self.context.im_provider if self.context else None

        if action != "send_message":
            return f"Unknown action: {action}. Supported: send_message"

        if not message:
            return "Error: message cannot be empty."

        if not im:
            return "Error: IM provider not configured. Cannot send messages."

        try:
            sent = await im.send_message(content=message, prefer_p2p=True)
            if sent:
                log.info("im_tool_message_sent", length=len(message))
                return "Message sent successfully."
            else:
                log.warning("im_tool_send_returned_false")
                return "Failed to send message. No available chat found."
        except Exception as exc:
            log.error("im_tool_send_error", error=str(exc))
            return f"Error sending message: {exc}"

"""Third-party integration helpers extracted from bootstrap.

Keeps :class:`MiraiApp` focused on lifecycle orchestration while
Feishu, dreamer, and check-in logic live here.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from mirai.logging import get_logger

if TYPE_CHECKING:
    from mirai.agent.agent_loop import AgentLoop
    from mirai.agent.im.feishu import FeishuProvider
    from mirai.config import MiraiConfig

log = get_logger("mirai.integrations")


# ---------------------------------------------------------------------------
# Feishu IM
# ---------------------------------------------------------------------------


def create_im_provider(config: MiraiConfig) -> FeishuProvider | None:
    """Create a :class:`FeishuProvider` from config, or *None* if disabled."""
    if not config.feishu.enabled:
        return None

    if config.feishu.app_id and config.feishu.app_secret:
        from mirai.agent.im.feishu import FeishuProvider

        log.info("feishu_enabled", mode="app_api")
        return FeishuProvider(app_id=config.feishu.app_id, app_secret=config.feishu.app_secret)

    if config.feishu.webhook_url:
        from mirai.agent.im.feishu import FeishuProvider

        log.info("feishu_enabled", mode="webhook")
        return FeishuProvider(webhook_url=config.feishu.webhook_url)

    return None


def start_feishu_receiver(agent: AgentLoop, config: MiraiConfig) -> None:
    """Start the Feishu WebSocket event receiver if configured."""
    if not (config.feishu.enabled and config.feishu.app_id and config.feishu.app_secret):
        return

    from mirai.agent.im.feishu_receiver import FeishuEventReceiver

    async def handle_feishu_message(
        sender_id: str,
        text: str | list[dict[str, Any]],
        chat_id: str,
        history: list[dict],
    ) -> str:
        """Route incoming Feishu messages to AgentLoop with conversation history."""
        return await agent.run(text, history=history)

    receiver = FeishuEventReceiver(
        app_id=config.feishu.app_id,
        app_secret=config.feishu.app_secret,
        message_handler=handle_feishu_message,
        storage=agent.l3_storage,
    )
    receiver.start(loop=asyncio.get_running_loop())
    log.info("feishu_receiver_started")


async def send_checkin(agent: AgentLoop, im_provider: FeishuProvider | None, config: MiraiConfig) -> None:
    """Send an interactive check-in card to Feishu on startup."""
    if not im_provider:
        return

    checkin_card = {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": "blue",
            "title": {"content": f"🤖 {agent.name} is Online", "tag": "plain_text"},
        },
        "elements": [
            {
                "tag": "div",
                "text": {
                    "content": (
                        f"**Status:** Ready to collaborate\n"
                        f"**Model:** `{config.llm.default_model}`\n"
                        f"**Version:** `v1.2.0`"
                    ),
                    "tag": "lark_md",
                },
            },
            {"tag": "hr"},
            {
                "tag": "note",
                "elements": [{"tag": "plain_text", "content": "Send me a message anytime to start!"}],
            },
        ],
    }
    checkin_ok = await im_provider.send_card(
        card_content=checkin_card,
        chat_id=config.feishu.curator_chat_id,
        prefer_p2p=False,
    )
    if checkin_ok:
        log.info("feishu_checkin_sent", agent=agent.name, style="card")
    else:
        log.warning("feishu_checkin_failed")


# ---------------------------------------------------------------------------
# Dreamer
# ---------------------------------------------------------------------------


def start_dreamer(agent: AgentLoop, config: MiraiConfig) -> Any:
    """Create and start the :class:`AgentDreamer` background service.

    Returns the dreamer instance so the caller can track it for shutdown.
    """
    from mirai.agent.agent_dreamer import AgentDreamer

    dreamer = AgentDreamer(
        agent,
        agent.l3_storage,
        interval_seconds=config.dreamer.interval,
    )
    dreamer.start(loop=asyncio.get_running_loop())
    return dreamer

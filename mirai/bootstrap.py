"""Mirai application bootstrap — encapsulates subsystem initialization and shutdown."""

import asyncio
import os
import time
from typing import Any

from mirai.agent.dreamer import Dreamer
from mirai.agent.heartbeat import HeartbeatManager
from mirai.agent.loop import AgentLoop
from mirai.agent.providers import create_provider
from mirai.agent.tools.echo import EchoTool
from mirai.agent.tools.system import SystemTool
from mirai.agent.tools.workspace import WorkspaceTool
from mirai.config import MiraiConfig
from mirai.db.session import init_db
from mirai.logging import get_logger, setup_logging
from mirai.tracing import setup_tracing

log = get_logger("mirai.bootstrap")


class MiraiApp:
    """Owns all subsystem instances and their lifecycle."""

    def __init__(self) -> None:
        self.agent: AgentLoop | None = None
        self.heartbeat: HeartbeatManager | None = None
        self.dreamer: Dreamer | None = None
        self.config: MiraiConfig | None = None
        self.start_time: float = time.monotonic()

    @classmethod
    async def create(cls) -> "MiraiApp":
        """Build and initialize all subsystems; return a ready-to-use instance."""
        self = cls()
        self.start_time = time.monotonic()

        # Load configuration (TOML > env vars > defaults)
        self.config = MiraiConfig.load()
        config = self.config

        # Setup structured logging
        json_output = os.getenv("MIRAI_LOG_FORMAT", "console") == "json"
        setup_logging(json_output=json_output, level=config.server.log_level)

        log.info(
            "config_loaded",
            model=config.llm.default_model,
            host=config.server.host,
            port=config.server.port,
        )

        # Setup OpenTelemetry tracing
        console_traces = os.getenv("OTEL_TRACES_CONSOLE", "").lower() in ("1", "true")
        setup_tracing(service_name="mirai", console=console_traces)

        # Initialize SQLite tables
        await init_db(config.database.sqlite_url)

        # Initialize Agent components
        try:
            provider = create_provider(model=config.llm.default_model)

            from mirai.agent.tools.editor import EditorTool
            from mirai.agent.tools.git import GitTool
            from mirai.agent.tools.shell import ShellTool

            system_tool = SystemTool(config=config, start_time=self.start_time, provider=provider)
            tools = [EchoTool(), WorkspaceTool(), ShellTool(), EditorTool(), GitTool(), system_tool]
            self.agent = await AgentLoop.create(
                provider=provider,
                tools=tools,
                collaborator_id=config.agent.collaborator_id,
                # In production, we use defaults (real DuckDB/LanceDB)
                # Future: pass config.database.duckdb_path
            )
            log.info("agent_initialized", collaborator=self.agent.name)

            # Start heartbeat with optional IM integration
            im_provider = self._create_im_provider(config)
            if config.heartbeat.enabled:
                self.heartbeat = HeartbeatManager(
                    self.agent,
                    interval_seconds=config.heartbeat.interval,
                    im_provider=im_provider,
                )
                await self.heartbeat.start()

            # Start Feishu WebSocket receiver
            self._start_feishu_receiver(config)

            # Send check-in card
            await self._send_checkin(im_provider, config)

            # Start Dreamer service
            self._start_dreamer(config)

        except Exception as e:
            log.error("agent_init_failed", error=str(e))
            self.agent = None

        return self

    async def shutdown(self) -> None:
        """Gracefully shut down all subsystems."""
        if self.heartbeat:
            await self.heartbeat.stop()

    # ----- Private helpers -----

    @staticmethod
    def _create_im_provider(config: MiraiConfig):
        """Create FeishuProvider if configured, else None."""
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

    def _start_feishu_receiver(self, config: MiraiConfig) -> None:
        """Start Feishu WebSocket if configured."""
        if not (config.feishu.enabled and config.feishu.app_id and config.feishu.app_secret):
            return

        from mirai.agent.im.feishu_receiver import FeishuEventReceiver

        agent = self.agent

        async def handle_feishu_message(
            sender_id: str,
            text: str | list[dict[str, Any]],
            chat_id: str,
            history: list[dict],
        ) -> str:
            """Route incoming Feishu messages to AgentLoop with conversation history."""
            if agent:
                return await agent.run(text, history=history)
            return "Agent is not initialized."

        receiver = FeishuEventReceiver(
            app_id=config.feishu.app_id,
            app_secret=config.feishu.app_secret,
            message_handler=handle_feishu_message,
            storage=agent.l3_storage if agent else None,
        )
        receiver.start(loop=asyncio.get_running_loop())
        log.info("feishu_receiver_started")

    async def _send_checkin(self, im_provider, config: MiraiConfig) -> None:
        """Send check-in card to Feishu on startup."""
        if not (im_provider and self.agent):
            return

        checkin_card = {
            "config": {"wide_screen_mode": True},
            "header": {
                "template": "blue",
                "title": {"content": f"🤖 {self.agent.name} is Online", "tag": "plain_text"},
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
            log.info("feishu_checkin_sent", agent=self.agent.name, style="card")
        else:
            log.warning("feishu_checkin_failed")

    def _start_dreamer(self, config: MiraiConfig) -> None:
        """Start the Dreamer background service."""
        if not self.agent:
            return
        dream_interval = int(os.getenv("MIRAI_DREAM_INTERVAL", "3600"))
        self.dreamer = Dreamer(self.agent, self.agent.l3_storage, interval_seconds=dream_interval)
        self.dreamer.start(loop=asyncio.get_running_loop())

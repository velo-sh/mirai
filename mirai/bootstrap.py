"""Mirai application bootstrap — encapsulates subsystem initialization and shutdown."""

import asyncio
import os
import signal
import time
from pathlib import Path
from typing import Any

from mirai.agent.agent_dreamer import AgentDreamer
from mirai.agent.agent_loop import AgentLoop
from mirai.agent.heartbeat import HeartbeatManager
from mirai.agent.providers import create_provider
from mirai.agent.registry import ModelRegistry, registry_refresh_loop
from mirai.agent.tools.system import SystemTool
from mirai.config import MiraiConfig
from mirai.db.session import init_db
from mirai.logging import get_logger, setup_logging
from mirai.tracing import setup_tracing

log = get_logger("mirai.bootstrap")


def _wait_for_duckdb_lock(db_path: str, timeout: float = 10.0) -> None:
    """Wait for a stale DuckDB lock to be released, or clean up if PID is dead.

    DuckDB uses a .wal file that can persist after a crash.  If the PID
    that held the lock is dead, we can safely proceed (DuckDB will recover
    on connect).  If the PID is still alive we poll until timeout.
    """
    wal_path = Path(db_path + ".wal")
    if not wal_path.exists():
        return

    log.warning("duckdb_wal_detected", wal=str(wal_path))

    # Try to determine if any process still holds the lock
    # by attempting a brief connect / close cycle
    import duckdb

    deadline = time.monotonic() + timeout
    attempt = 0
    while True:
        attempt += 1
        try:
            conn = duckdb.connect(db_path)
            conn.close()
            log.info("duckdb_lock_cleared", attempts=attempt)
            return
        except duckdb.IOException:
            if time.monotonic() >= deadline:
                log.warning(
                    "duckdb_lock_timeout",
                    db=db_path,
                    timeout=timeout,
                    msg="Proceeding anyway — DuckDB may raise on first write",
                )
                return
            log.info("duckdb_lock_waiting", attempt=attempt, remaining=round(deadline - time.monotonic(), 1))
            time.sleep(1.0)


class MiraiApp:
    """Owns all subsystem instances and their lifecycle.

    The ``create()`` classmethod initializes subsystems in four phases:

    1. **Config & observability** — load TOML, set up logging & tracing.
    2. **Storage** — initialize SQLite + DuckDB lock check.
    3. **Agent stack** — registry → provider → tools → AgentLoop.
    4. **Integrations & background** — IM, heartbeat, dreamer, refresh loop.
    """

    def __init__(self) -> None:
        self.agent: AgentLoop | None = None
        self.heartbeat: HeartbeatManager | None = None
        self.dreamer: AgentDreamer | None = None
        self.registry: ModelRegistry | None = None
        self.config: MiraiConfig | None = None
        self.start_time: float = time.monotonic()
        self._tasks: set[asyncio.Task[None]] = set()

    # ------------------------------------------------------------------
    # Lifecycle: public entry points
    # ------------------------------------------------------------------

    @classmethod
    async def create(cls) -> "MiraiApp":
        """Build and initialize all subsystems; return a ready-to-use instance."""
        self = cls()
        self.start_time = time.monotonic()

        self._init_config()
        await self._init_storage()
        await self._init_agent_stack()
        await self._init_integrations()
        self._init_background_tasks()

        return self

    async def shutdown(self) -> None:
        """Gracefully shut down all subsystems."""
        log.info("graceful_shutdown_started")

        # Cancel all tracked background tasks
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
            log.info("background_tasks_cancelled", count=len(self._tasks))
            self._tasks.clear()

        if self.heartbeat:
            await self.heartbeat.stop()
        if self.agent and hasattr(self.agent, "l3_storage") and self.agent.l3_storage:
            try:
                self.agent.l3_storage.close()
            except Exception:
                log.warning("l3_storage_close_failed", exc_info=True)
        log.info("graceful_shutdown_complete")

    def install_signal_handlers(self, loop: asyncio.AbstractEventLoop) -> None:
        """Register SIGTERM/SIGINT handlers for graceful shutdown."""

        def _handle_signal(sig: int) -> None:
            sig_name = signal.Signals(sig).name
            log.info("signal_received", signal=sig_name)
            loop.create_task(self.shutdown())
            loop.stop()

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, _handle_signal, sig)

    # ------------------------------------------------------------------
    # Phase 1: Config & observability
    # ------------------------------------------------------------------

    def _init_config(self) -> None:
        """Load configuration and set up logging + tracing."""
        self.config = MiraiConfig.load()
        config = self.config

        json_output = os.getenv("MIRAI_LOG_FORMAT", "console") == "json"
        setup_logging(json_output=json_output, level=config.server.log_level)

        log.info(
            "config_loaded",
            model=config.llm.default_model,
            host=config.server.host,
            port=config.server.port,
        )

        console_traces = os.getenv("OTEL_TRACES_CONSOLE", "").lower() in ("1", "true")
        setup_tracing(service_name="mirai", console=console_traces)

    # ------------------------------------------------------------------
    # Phase 2: Storage
    # ------------------------------------------------------------------

    async def _init_storage(self) -> None:
        """Initialize SQLite + check DuckDB lock."""
        assert self.config is not None
        await init_db(self.config.database.sqlite_url)
        _wait_for_duckdb_lock(self.config.database.duckdb_path)

    # ------------------------------------------------------------------
    # Phase 3: Agent stack (registry → provider → tools → loop)
    # ------------------------------------------------------------------

    async def _init_agent_stack(self) -> None:
        """Build the agent: registry, provider, tools, and AgentLoop."""
        assert self.config is not None
        config = self.config

        try:
            # Initialize model registry first to discover persisted active model
            self.registry = ModelRegistry(
                config_provider=config.llm.provider,
                config_model=config.llm.default_model,
            )

            # Use persisted active model/provider if available, fall back to config
            effective_provider = self.registry.active_provider
            effective_model = self.registry.active_model
            if effective_provider != config.llm.provider or effective_model != config.llm.default_model:
                log.info(
                    "registry_override",
                    config_provider=config.llm.provider,
                    config_model=config.llm.default_model,
                    effective_provider=effective_provider,
                    effective_model=effective_model,
                )

            provider = create_provider(
                provider=effective_provider,
                model=effective_model,
                api_key=config.llm.api_key,
                base_url=config.llm.base_url,
            )

            from mirai.agent.tools.base import ToolContext
            from mirai.agent.tools.echo import EchoTool
            from mirai.agent.tools.editor import EditorTool
            from mirai.agent.tools.git import GitTool
            from mirai.agent.tools.shell import ShellTool
            from mirai.agent.tools.workspace import WorkspaceTool

            context = ToolContext(
                config=config,
                registry=self.registry,
                provider=provider,
                start_time=self.start_time,
            )

            system_tool = SystemTool(context=context)
            tools = [
                EchoTool(context=context),
                WorkspaceTool(context=context),
                ShellTool(context=context),
                EditorTool(context=context),
                GitTool(context=context),
                system_tool,
            ]
            self.agent = await AgentLoop.create(
                provider=provider,
                tools=tools,
                collaborator_id=config.agent.collaborator_id,
                fallback_models=config.registry.fallback_models,
            )
            # Wire agent_loop reference for model hot-swap (circular dep resolved post-init)
            system_tool._agent_loop = self.agent
            log.info("agent_initialized", collaborator=self.agent.name)

        except Exception as e:
            log.error("agent_init_failed", error=str(e))
            self.agent = None

    # ------------------------------------------------------------------
    # Phase 4: Integrations (IM, heartbeat, dreamer) + background tasks
    # ------------------------------------------------------------------

    async def _init_integrations(self) -> None:
        """Start heartbeat, Feishu, check-in card, and dreamer."""
        assert self.config is not None
        config = self.config

        if not self.agent:
            return

        im_provider = self._create_im_provider(config)

        if config.heartbeat.enabled:
            self.heartbeat = HeartbeatManager(
                self.agent,
                interval_seconds=config.heartbeat.interval,
                im_provider=im_provider,
            )
            self.heartbeat.start()

        self._start_feishu_receiver(config)
        await self._send_checkin(im_provider, config)
        self._start_dreamer(config)

    def _init_background_tasks(self) -> None:
        """Register long-running background tasks (registry refresh, etc.)."""
        assert self.config is not None
        config = self.config

        if not self.agent or not self.registry:
            return

        provider = self.agent.provider
        quota_mgr = getattr(provider, "quota_manager", None)
        self._track_task(
            registry_refresh_loop(
                self.registry,
                interval=config.registry.refresh_interval,
                quota_manager=quota_mgr,
            ),
            name="registry_refresh",
        )

    # ------------------------------------------------------------------
    # Background task tracking
    # ------------------------------------------------------------------

    def _track_task(self, coro: Any, *, name: str = "unnamed") -> asyncio.Task[None]:
        """Create and track an asyncio background task.

        The task is added to ``self._tasks`` and automatically removed
        on completion.  Exceptions are logged but never propagated.
        """
        task: asyncio.Task[None] = asyncio.get_running_loop().create_task(coro)
        self._tasks.add(task)

        def _done(t: asyncio.Task[None]) -> None:
            self._tasks.discard(t)
            if t.cancelled():
                return
            exc = t.exception()
            if exc:
                log.error("background_task_failed", task=name, error=str(exc))

        task.add_done_callback(_done)
        return task

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
        self.dreamer = AgentDreamer(self.agent, self.agent.l3_storage, interval_seconds=dream_interval)
        self.dreamer.start(loop=asyncio.get_running_loop())

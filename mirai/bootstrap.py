"""Mirai application bootstrap — encapsulates subsystem initialization and shutdown."""

import asyncio
import os
import signal
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mirai.agent.tools.base import ToolContext

from mirai import integrations
from mirai.agent.agent_dreamer import AgentDreamer
from mirai.agent.agent_loop import AgentLoop
from mirai.agent.heartbeat import HeartbeatManager
from mirai.agent.providers import create_provider
from mirai.agent.registry import ModelRegistry, registry_refresh_loop
from mirai.agent.tools.system import SystemTool
from mirai.config import CONFIG_DIR, MiraiConfig
from mirai.cron import CronScheduler, ensure_system_jobs
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
        self.cron: CronScheduler | None = None
        self.registry: ModelRegistry | None = None
        self.config: MiraiConfig | None = None
        self.start_time: float = time.monotonic()
        self._tasks: set[asyncio.Task[None]] = set()
        self._im_provider: Any = None
        self._tool_context: ToolContext | None = None

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
        self._init_cron()  # before agent stack so ToolContext can reference it
        await self._init_agent_stack()
        # Wire agent into cron (cron was created before agent existed)
        if self.cron and self.agent:
            self.cron.agent = self.agent
        await self._init_integrations()
        # Wire im_provider into tools and cron (for curator alerts)
        if self._im_provider:
            if self.cron:
                self.cron.im_provider = self._im_provider
            if self._tool_context:
                self._tool_context.im_provider = self._im_provider
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

        # Stop background services
        if self.cron:
            self.cron.stop()
        if self.dreamer:
            await self.dreamer.stop()
        if self.heartbeat:
            await self.heartbeat.stop()

        if self.agent and self.agent.l3_storage:
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

        json_output = config.server.log_format == "json"
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
            # Wire external enrichment source (models.dev metadata)
            from mirai.agent.models_dev import ModelsDevSource

            self.registry.set_enrichment_source(ModelsDevSource())

            # Wire free-provider discovery source (OpenRouter, SambaNova, etc.)
            from mirai.agent.free_providers import FreeProviderSource

            self.registry.set_free_source(FreeProviderSource())

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
            from mirai.agent.tools.config_tool import ConfigTool
            from mirai.agent.tools.cron_tool import CronTool
            from mirai.agent.tools.echo import EchoTool
            from mirai.agent.tools.editor import EditorTool
            from mirai.agent.tools.git import GitTool
            from mirai.agent.tools.im import IMTool
            from mirai.agent.tools.model import ModelTool
            from mirai.agent.tools.shell import ShellTool
            from mirai.agent.tools.workspace import WorkspaceTool

            context = ToolContext(
                config=config,
                registry=self.registry,
                provider=provider,
                cron_scheduler=self.cron,
                start_time=self.start_time,
            )

            system_tool = SystemTool(context=context)
            im_tool = IMTool(context=context)
            tools = [
                EchoTool(context=context),
                WorkspaceTool(context=context),
                ShellTool(context=context),
                EditorTool(context=context),
                GitTool(context=context),
                system_tool,
                ModelTool(context=context),
                ConfigTool(context=context),
                CronTool(context=context),
                im_tool,
            ]
            self._tool_context = context  # keep ref for im_provider wiring
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
        """Start heartbeat, Feishu, check-in card, and dreamer.

        Delegates to :mod:`mirai.integrations` for Feishu and dreamer setup.
        """
        assert self.config is not None
        config = self.config

        if not self.agent:
            return

        im_provider = integrations.create_im_provider(config)
        self._im_provider = im_provider  # stored for cron wiring

        if config.heartbeat.enabled:
            self.heartbeat = HeartbeatManager(
                self.agent,
                interval_seconds=config.heartbeat.interval,
                im_provider=im_provider,
            )
            self.heartbeat.start()

        integrations.start_feishu_receiver(self.agent, config)
        await integrations.send_checkin(self.agent, im_provider, config)
        self.dreamer = integrations.start_dreamer(self.agent, config)

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

        # Periodic health checks for configured free providers
        self._track_task(
            _health_check_loop(self.registry, interval=600),
            name="provider_health_check",
        )

    def _init_cron(self) -> None:
        """Initialize the cron scheduler with JSON5 file persistence.

        Note: agent and im_provider are wired later in create() after init.
        """
        state_dir = CONFIG_DIR / "state" / "cron"
        ensure_system_jobs(state_dir)

        self.cron = CronScheduler(
            state_dir=state_dir,
            agent=None,  # wired after agent stack init
        )
        self.cron.start()
        log.info("cron_scheduler_started", state_dir=str(state_dir))

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


# ---------------------------------------------------------------------------
# Health check background loop (runs outside MiraiApp)
# ---------------------------------------------------------------------------


async def _health_check_loop(
    registry: ModelRegistry,
    interval: int = 600,
) -> None:
    """Periodically probe configured free providers and update registry.

    Args:
        registry: The ModelRegistry instance to update.
        interval: Seconds between health checks (default: 10 minutes).
    """
    from mirai.agent.free_providers import check_provider_health

    # Initial probe shortly after startup (let registry refresh settle first)
    await asyncio.sleep(30)

    while True:
        try:
            health = await check_provider_health()
            if health:
                registry.update_health(health)
                log.info(
                    "provider_health_check_complete",
                    checked=len(health),
                    healthy=sum(1 for h in health.values() if h.healthy),
                )
        except Exception as exc:
            log.warning("provider_health_check_failed", error=str(exc))

        await asyncio.sleep(interval)

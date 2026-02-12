"""Mirai Application Entry Point."""

import os
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel

load_dotenv()

import asyncio  # noqa: E402

from mirai.agent.heartbeat import HeartbeatManager  # noqa: E402
from mirai.agent.loop import AgentLoop  # noqa: E402
from mirai.agent.providers import create_provider  # noqa: E402
from mirai.agent.tools.echo import EchoTool  # noqa: E402
from mirai.agent.tools.workspace import WorkspaceTool  # noqa: E402
from mirai.config import MiraiConfig  # noqa: E402
from mirai.db.session import init_db  # noqa: E402
from mirai.logging import get_logger, setup_logging  # noqa: E402

log = get_logger("mirai.main")


class ChatRequest(BaseModel):
    message: str


# Global instances
agent: AgentLoop | None = None
heartbeat: HeartbeatManager | None = None
config: MiraiConfig | None = None


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    global agent, heartbeat, config
    # Load configuration (TOML > env vars > defaults)
    config = MiraiConfig.load()

    # Setup structured logging (JSON in production, colored in dev)
    json_output = os.getenv("MIRAI_LOG_FORMAT", "console") == "json"
    setup_logging(json_output=json_output, level=config.server.log_level)

    log.info("config_loaded", model=config.llm.default_model, host=config.server.host, port=config.server.port)

    # Initialize SQLite tables
    await init_db(config.database.sqlite_url)

    # Initialize Agent components
    try:
        provider = create_provider(model=config.llm.default_model)

        from mirai.agent.tools.editor import EditorTool
        from mirai.agent.tools.shell import ShellTool

        tools = [EchoTool(), WorkspaceTool(), ShellTool(), EditorTool()]
        agent = await AgentLoop.create(
            provider=provider,
            tools=tools,
            collaborator_id=config.agent.collaborator_id,
        )
        log.info("agent_initialized", collaborator=agent.name)

        # Start Heartbeat with Optional IM Integration
        im_provider = None
        if config.feishu.enabled and (config.feishu.app_id and config.feishu.app_secret):
            from mirai.agent.im.feishu import FeishuProvider

            im_provider = FeishuProvider(app_id=config.feishu.app_id, app_secret=config.feishu.app_secret)
            log.info("feishu_enabled", mode="app_api")
        elif config.feishu.enabled and config.feishu.webhook_url:
            from mirai.agent.im.feishu import FeishuProvider

            im_provider = FeishuProvider(webhook_url=config.feishu.webhook_url)
            log.info("feishu_enabled", mode="webhook")

        if config.heartbeat.enabled:
            heartbeat = HeartbeatManager(
                agent,
                interval_seconds=config.heartbeat.interval,
                im_provider=im_provider,
            )
            await heartbeat.start()

        # Start Feishu WebSocket receiver for private/group chat
        if config.feishu.enabled and config.feishu.app_id and config.feishu.app_secret:
            from mirai.agent.im.feishu_receiver import FeishuEventReceiver

            async def handle_feishu_message(sender_id: str, text: str, chat_id: str) -> str:
                """Route incoming Feishu messages to AgentLoop."""
                if agent:
                    return await agent.run(text)
                return "Agent is not initialized."

            receiver = FeishuEventReceiver(
                app_id=config.feishu.app_id,
                app_secret=config.feishu.app_secret,
                message_handler=handle_feishu_message,
            )
            receiver.start(loop=asyncio.get_running_loop())
            log.info("feishu_receiver_started")

    except Exception as e:
        log.error("agent_init_failed", error=str(e))
        agent = None

    yield
    if heartbeat:
        await heartbeat.stop()


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# Structlog request logging middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def structlog_access_log(request: Request, call_next) -> Response:  # type: ignore[type-arg]
    """Log every HTTP request with method, path, status, and duration."""
    start = time.perf_counter()
    response: Response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1_000
    log.info(
        "http_request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=round(duration_ms, 1),
    )
    return response


@app.get("/health")
async def health_check():
    """Simple health check for the watchdog."""
    return {"status": "ok", "pid": os.getpid()}


@app.post("/chat")
async def chat(request: ChatRequest):
    """Entry point for interacting with the AI Collaborator."""
    if not agent:
        raise HTTPException(status_code=500, detail="AgentLoop not initialized. Check API keys.")

    try:
        response = await agent.run(request.message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


def main():
    """Start the Mirai server using Granian (Rust ASGI server)."""
    from granian import Granian
    from granian.constants import Interfaces

    cfg = config or MiraiConfig.load()
    log.info("server_starting", host=cfg.server.host, port=cfg.server.port, server="granian")

    server = Granian(
        "main:app",
        address=cfg.server.host,
        port=cfg.server.port,
        interface=Interfaces.ASGI,
        workers=1,
    )
    server.serve()


if __name__ == "__main__":
    main()

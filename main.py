"""Mirai Application Entry Point."""

import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.status import HTTP_429_TOO_MANY_REQUESTS

load_dotenv()

import asyncio  # noqa: E402

from mirai.agent.dreamer import Dreamer  # noqa: E402
from mirai.agent.heartbeat import HeartbeatManager  # noqa: E402
from mirai.agent.loop import AgentLoop  # noqa: E402
from mirai.agent.providers import create_provider  # noqa: E402
from mirai.agent.tools.echo import EchoTool  # noqa: E402
from mirai.agent.tools.workspace import WorkspaceTool  # noqa: E402
from mirai.config import MiraiConfig  # noqa: E402
from mirai.db.session import init_db  # noqa: E402
from mirai.logging import get_logger, setup_logging  # noqa: E402
from mirai.tracing import setup_tracing  # noqa: E402

log = get_logger("mirai.main")


class ChatRequest(BaseModel):
    message: str


# ---------------------------------------------------------------------------
# Global instances
# ---------------------------------------------------------------------------
agent: AgentLoop | None = None
heartbeat: HeartbeatManager | None = None
dreamer: Dreamer | None = None
config: MiraiConfig | None = None
_start_time: float = time.monotonic()

# ---------------------------------------------------------------------------
# Rate limiting (zero-dependency, in-memory)
# ---------------------------------------------------------------------------
# Concurrency gate: max simultaneous LLM calls
_chat_semaphore = asyncio.Semaphore(3)

# Per-IP sliding window: {ip: [timestamps]}
_rate_limits: dict[str, list[float]] = defaultdict(list)
_RATE_WINDOW = 60.0  # seconds
_RATE_MAX = 20  # requests per window


def _check_rate_limit(client_ip: str) -> bool:
    """Return True if the request is within rate limits."""
    now = time.monotonic()
    timestamps = _rate_limits[client_ip]
    # Purge expired entries
    _rate_limits[client_ip] = [t for t in timestamps if now - t < _RATE_WINDOW]
    if len(_rate_limits[client_ip]) >= _RATE_MAX:
        return False
    _rate_limits[client_ip].append(now)
    return True


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    global agent, heartbeat, config, _start_time
    _start_time = time.monotonic()

    # Load configuration (TOML > env vars > defaults)
    config = MiraiConfig.load()

    # Setup structured logging (JSON in production, colored in dev)
    json_output = os.getenv("MIRAI_LOG_FORMAT", "console") == "json"
    setup_logging(json_output=json_output, level=config.server.log_level)

    log.info("config_loaded", model=config.llm.default_model, host=config.server.host, port=config.server.port)

    # Setup OpenTelemetry tracing
    console_traces = os.getenv("OTEL_TRACES_CONSOLE", "").lower() in ("1", "true")
    setup_tracing(service_name="mirai", console=console_traces)

    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(app_instance)

    # Initialize SQLite tables
    await init_db(config.database.sqlite_url)

    # Initialize Agent components
    try:
        provider = create_provider(model=config.llm.default_model)

        from mirai.agent.tools.editor import EditorTool
        from mirai.agent.tools.git import GitTool
        from mirai.agent.tools.shell import ShellTool

        tools = [EchoTool(), WorkspaceTool(), ShellTool(), EditorTool(), GitTool()]
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

        # Send check-in message to Feishu on startup (using a "panel" style card)
        if im_provider and agent:
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
                            "content": f"**Status:** Ready to collaborate\n**Model:** `{config.llm.default_model}`\n**Version:** `v1.2.0`",
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

        # Start Dreamer Service
        if agent:
            # For "dreaming" to work, we need at least some interval.
            # Default to 1 hour, or shorter for dev if needed.
            dream_interval = int(os.getenv("MIRAI_DREAM_INTERVAL", "3600"))
            dreamer = Dreamer(agent, agent.l3_storage, interval_seconds=dream_interval)
            dreamer.start(loop=asyncio.get_running_loop())

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


# ---------------------------------------------------------------------------
# Health check (enhanced)
# ---------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    """Enhanced health check with system metrics."""
    import resource

    rusage = resource.getrusage(resource.RUSAGE_SELF)
    memory_mb = round(rusage.ru_maxrss / (1024 * 1024), 1)  # macOS: bytes → MB

    provider_name = type(agent.provider).__name__ if agent else None
    model_name = getattr(agent.provider, "model", None) if agent else None

    return {
        "status": "ok" if agent else "degraded",
        "pid": os.getpid(),
        "uptime_seconds": round(time.monotonic() - _start_time, 1),
        "memory_mb": memory_mb,
        "agent_ready": agent is not None,
        "provider": provider_name,
        "model": model_name,
    }


# ---------------------------------------------------------------------------
# Chat endpoint (with rate limiting)
# ---------------------------------------------------------------------------
@app.post("/chat")
async def chat(request: ChatRequest, req: Request):
    """Entry point for interacting with the AI Collaborator."""
    if not agent:
        raise HTTPException(status_code=500, detail="AgentLoop not initialized. Check API keys.")

    # Rate limit check
    client_ip = req.client.host if req.client else "unknown"
    if not _check_rate_limit(client_ip):
        raise HTTPException(
            status_code=HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded ({_RATE_MAX} requests per {int(_RATE_WINDOW)}s). Try again later.",
            headers={"Retry-After": str(int(_RATE_WINDOW))},
        )

    # Concurrency gate
    try:
        async with asyncio.timeout(1.0):
            await _chat_semaphore.acquire()
    except TimeoutError:
        raise HTTPException(
            status_code=HTTP_429_TOO_MANY_REQUESTS,
            detail="Server busy — too many concurrent requests. Try again shortly.",
            headers={"Retry-After": "5"},
        ) from None

    try:
        response = await agent.run(request.message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        _chat_semaphore.release()


# ---------------------------------------------------------------------------
# SSE Streaming endpoint
# ---------------------------------------------------------------------------
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest, req: Request):
    """Stream the agent response as Server-Sent Events."""
    if not agent:
        raise HTTPException(status_code=500, detail="AgentLoop not initialized.")

    client_ip = req.client.host if req.client else "unknown"
    if not _check_rate_limit(client_ip):
        raise HTTPException(
            status_code=HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded.",
            headers={"Retry-After": str(int(_RATE_WINDOW))},
        )

    async def _event_generator():
        try:
            async with asyncio.timeout(1.0):
                await _chat_semaphore.acquire()
        except TimeoutError:
            yield "event: error\ndata: Server busy\n\n"
            return

        try:
            async for event in agent.stream_run(request.message):
                yield f"event: {event['event']}\ndata: {event['data']}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {e!s}\n\n"
        finally:
            _chat_semaphore.release()

    return StreamingResponse(_event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# WebSocket endpoint for multi-turn chat
# ---------------------------------------------------------------------------
@app.websocket("/ws/chat")
async def websocket_chat(ws: WebSocket):
    """Full-duplex WebSocket for multi-turn conversations."""
    await ws.accept()
    log.info("ws_connected", client=ws.client.host if ws.client else "unknown")

    try:
        while True:
            data = await ws.receive_json()
            message = data.get("message", "")
            if not message:
                await ws.send_json({"event": "error", "data": "Empty message"})
                continue

            if not agent:
                await ws.send_json({"event": "error", "data": "Agent not initialized"})
                continue

            try:
                async for event in agent.stream_run(message):
                    await ws.send_json(event)
            except Exception as e:
                await ws.send_json({"event": "error", "data": str(e)})

    except WebSocketDisconnect:
        log.info("ws_disconnected")
    except Exception as e:
        log.error("ws_error", error=str(e))


# ---------------------------------------------------------------------------
# Server startup
# ---------------------------------------------------------------------------
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

"""Mirai Application Entry Point."""

import asyncio
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.status import HTTP_429_TOO_MANY_REQUESTS

load_dotenv()

from mirai.bootstrap import MiraiApp  # noqa: E402
from mirai.config import MiraiConfig  # noqa: E402
from mirai.logging import get_logger  # noqa: E402

log = get_logger("mirai.main")


class ChatRequest(BaseModel):
    message: str


# ---------------------------------------------------------------------------
# Rate limiting (zero-dependency, in-memory)
# ---------------------------------------------------------------------------
_chat_semaphore = asyncio.Semaphore(3)
_rate_limits: dict[str, list[float]] = defaultdict(list)
_RATE_WINDOW = 60.0  # seconds
_RATE_MAX = 20  # requests per window

# Shared application state — set during lifespan
_mirai: MiraiApp | None = None


def _check_rate_limit(client_ip: str) -> bool:
    """Return True if the request is within rate limits."""
    now = time.monotonic()
    timestamps = _rate_limits[client_ip]
    _rate_limits[client_ip] = [t for t in timestamps if now - t < _RATE_WINDOW]
    if len(_rate_limits[client_ip]) >= _RATE_MAX:
        return False
    _rate_limits[client_ip].append(now)
    return True


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    global _mirai

    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    _mirai = await MiraiApp.create()
    FastAPIInstrumentor.instrument_app(app_instance)

    yield

    await _mirai.shutdown()
    _mirai = None


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
    try:
        import resource

        agent = _mirai.agent if _mirai else None

        rusage = resource.getrusage(resource.RUSAGE_SELF)
        memory_mb = round(rusage.ru_maxrss / (1024 * 1024), 1)  # macOS: bytes → MB

        provider_name = type(agent.provider).__name__ if agent else None
        model_name = getattr(agent.provider, "model", None) if agent else None
        start_time = getattr(_mirai, "start_time", 0) if _mirai else 0

        return {
            "status": "ok" if agent else "degraded",
            "pid": os.getpid(),
            "uptime_seconds": round(time.monotonic() - start_time, 1),
            "memory_mb": memory_mb,
            "agent_ready": agent is not None,
            "provider": provider_name,
            "model": model_name,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e


# ---------------------------------------------------------------------------
# Chat endpoint (with rate limiting)
# ---------------------------------------------------------------------------
@app.post("/chat")
async def chat(request: ChatRequest, req: Request):
    """Entry point for interacting with the AI Collaborator."""
    agent = _mirai.agent if _mirai else None
    if not agent:
        raise HTTPException(status_code=500, detail="AgentLoop not initialized. Check API keys.")

    client_ip = req.client.host if req.client else "unknown"
    if not _check_rate_limit(client_ip):
        raise HTTPException(
            status_code=HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded ({_RATE_MAX} requests per {int(_RATE_WINDOW)}s). Try again later.",
            headers={"Retry-After": str(int(_RATE_WINDOW))},
        )

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
    agent = _mirai.agent if _mirai else None
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

            agent = _mirai.agent if _mirai else None
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

    cfg = MiraiConfig.load()
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

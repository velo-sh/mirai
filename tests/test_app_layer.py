"""Tests for application-layer features: rate limiter, SSE stream, WebSocket, health.

Creates a lightweight test FastAPI app that re-uses main.py's route handlers
but skips the lifespan (which tries to init real providers).
"""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.testclient import TestClient

import main as main_module
from mirai.bootstrap import MiraiApp


def _make_mock_agent():
    """Create a mock agent for testing endpoints."""
    agent = MagicMock()
    agent.run = AsyncMock(return_value="Mock response from agent")
    agent.provider = MagicMock()
    type(agent.provider).__name__ = "MockProvider"
    agent.provider.model = "mock-model"

    async def _stream(message):
        yield {"event": "thinking", "data": "thinking..."}
        yield {"event": "chunk", "data": "Hello "}
        yield {"event": "chunk", "data": "world"}
        yield {"event": "done", "data": "Hello world"}

    agent.stream_run = _stream
    return agent


def _make_mirai_app(agent=None):
    """Create a MiraiApp instance with the given agent (or None)."""
    mirai_app = MiraiApp()
    mirai_app.agent = agent
    mirai_app.start_time = time.monotonic()
    return mirai_app


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset rate limit state between tests."""
    main_module._rate_limits.clear()
    yield
    main_module._rate_limits.clear()


@pytest.fixture
def agent_client():
    """Client with mock agent, bypassing lifespan."""
    saved = main_module._mirai
    main_module._mirai = _make_mirai_app(agent=_make_mock_agent())

    from fastapi import FastAPI

    test_app = FastAPI()
    test_app.add_api_route("/health", main_module.health_check, methods=["GET"])
    test_app.add_api_route("/chat", main_module.chat, methods=["POST"])
    test_app.add_api_route("/chat/stream", main_module.chat_stream, methods=["POST"])
    test_app.add_api_websocket_route("/ws/chat", main_module.websocket_chat)

    with TestClient(test_app, raise_server_exceptions=False) as c:
        yield c
    main_module._mirai = saved


@pytest.fixture
def no_agent_client():
    """Client with agent=None, bypassing lifespan."""
    saved = main_module._mirai
    main_module._mirai = _make_mirai_app(agent=None)

    from fastapi import FastAPI

    test_app = FastAPI()
    test_app.add_api_route("/health", main_module.health_check, methods=["GET"])
    test_app.add_api_route("/chat", main_module.chat, methods=["POST"])
    test_app.add_api_route("/chat/stream", main_module.chat_stream, methods=["POST"])
    test_app.add_api_websocket_route("/ws/chat", main_module.websocket_chat)

    with TestClient(test_app, raise_server_exceptions=False) as c:
        yield c
    main_module._mirai = saved


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    def test_health_ok_with_agent(self, agent_client):
        resp = agent_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["agent_ready"] is True
        assert data["provider"] == "MockProvider"
        assert data["model"] == "mock-model"
        assert "uptime_seconds" in data
        assert "memory_mb" in data
        assert "pid" in data

    def test_health_degraded_without_agent(self, no_agent_client):
        resp = no_agent_client.get("/health")
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["agent_ready"] is False
        assert data["provider"] is None


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------


class TestRateLimiter:
    def test_chat_returns_response(self, agent_client):
        resp = agent_client.post("/chat", json={"message": "hello"})
        assert resp.status_code == 200
        assert resp.json()["response"] == "Mock response from agent"

    def test_rate_limit_exceeded(self, agent_client):
        # Simulate filling the rate limit bucket
        client_ip = "testclient"
        main_module._rate_limits[client_ip] = [time.monotonic()] * main_module._RATE_MAX

        resp = agent_client.post("/chat", json={"message": "hello"})
        assert resp.status_code == 429
        assert "Rate limit exceeded" in resp.json()["detail"]
        assert "Retry-After" in resp.headers

    def test_chat_without_agent_returns_500(self, no_agent_client):
        resp = no_agent_client.post("/chat", json={"message": "hello"})
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# SSE Streaming
# ---------------------------------------------------------------------------


class TestSSEStreaming:
    def test_stream_returns_events(self, agent_client):
        resp = agent_client.post("/chat/stream", json={"message": "hello"})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        body = resp.text
        assert "event: thinking" in body
        assert "event: chunk" in body
        assert "event: done" in body

    def test_stream_without_agent_returns_500(self, no_agent_client):
        resp = no_agent_client.post("/chat/stream", json={"message": "hello"})
        assert resp.status_code == 500

    def test_stream_rate_limited(self, agent_client):
        client_ip = "testclient"
        main_module._rate_limits[client_ip] = [time.monotonic()] * main_module._RATE_MAX

        resp = agent_client.post("/chat/stream", json={"message": "hello"})
        assert resp.status_code == 429


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------


class TestWebSocket:
    def test_websocket_send_receive(self, agent_client):
        with agent_client.websocket_connect("/ws/chat") as ws:
            ws.send_json({"message": "hello"})
            events = []
            for _ in range(4):  # thinking, chunk, chunk, done
                data = ws.receive_json()
                events.append(data)
                if data.get("event") == "done":
                    break

            event_types = [e["event"] for e in events]
            assert "thinking" in event_types
            assert "chunk" in event_types
            assert "done" in event_types

    def test_websocket_empty_message(self, agent_client):
        with agent_client.websocket_connect("/ws/chat") as ws:
            ws.send_json({"message": ""})
            data = ws.receive_json()
            assert data["event"] == "error"
            assert "Empty" in data["data"]

    def test_websocket_no_agent(self, no_agent_client):
        with no_agent_client.websocket_connect("/ws/chat") as ws:
            ws.send_json({"message": "hello"})
            data = ws.receive_json()
            assert data["event"] == "error"
            assert "not initialized" in data["data"]

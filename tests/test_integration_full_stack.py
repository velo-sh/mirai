"""Integration tests: full stack with real AgentLoop + MockProvider.

Tests the complete flow through HTTP endpoints (/chat, /chat/stream, /ws/chat)
using a real AgentLoop instance (not mocked), backed by MockProvider.
This validates the entire pipeline: request → AgentLoop → Think → Act → Critique → response.
"""

import time

import pytest
from starlette.testclient import TestClient

import main as main_module
from mirai.agent.loop import AgentLoop
from mirai.agent.providers import MockProvider
from mirai.agent.tools.echo import EchoTool
from mirai.agent.tools.workspace import WorkspaceTool


def _create_real_agent():
    """Create a real AgentLoop with MockProvider — no DB dependency."""
    from mirai.db.duck import DuckDBStorage

    provider = MockProvider()
    tools = [EchoTool(), WorkspaceTool()]
    agent = AgentLoop(
        provider, tools, collaborator_id="integration-test",
        l3_storage=DuckDBStorage(db_path=":memory:"),
    )
    agent.name = "IntegrationTestBot"
    agent.role = "test"
    agent.base_system_prompt = "You are a test agent."
    agent.soul_content = ""
    return agent


@pytest.fixture(autouse=True)
def _reset():
    main_module._rate_limits.clear()
    main_module._start_time = time.monotonic()
    yield
    main_module._rate_limits.clear()


@pytest.fixture
def real_app():
    """TestClient with a real AgentLoop backed by MockProvider."""
    # We need to patch main._mirai since 'agent' is no longer global
    # We'll create a dummy MiraiApp wrapper or just patch _mirai directly if accessible
    
    # Check if _mirai exists, if not (app not started), we might need to simulate it
    # However, main.py lifespan handles init.
    # But this fixture wants to INJECT a specific agent.
    
    from mirai.bootstrap import MiraiApp
    from unittest.mock import MagicMock, AsyncMock
    
    # Create a wrapper that mimics MiraiApp but with our agent
    real_agent = _create_real_agent()
    
    original_mirai = main_module._mirai
    
    # Mock app that holds our agent
    mock_app = MagicMock(spec=MiraiApp)
    mock_app.agent = real_agent
    mock_app.config = MagicMock()
    mock_app.start =  AsyncMock()
    mock_app.shutdown = AsyncMock()
    
    main_module._mirai = mock_app

    from fastapi import FastAPI

    app = FastAPI()
    app.add_api_route("/health", main_module.health_check, methods=["GET"])
    app.add_api_route("/chat", main_module.chat, methods=["POST"])
    app.add_api_route("/chat/stream", main_module.chat_stream, methods=["POST"])
    app.add_api_websocket_route("/ws/chat", main_module.websocket_chat)

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    
    main_module._mirai = original_mirai


# ---------------------------------------------------------------------------
# Integration: /chat (synchronous)
# ---------------------------------------------------------------------------


class TestIntegrationChat:
    def test_full_pipeline_returns_response(self, real_app):
        """POST /chat with real AgentLoop should return a non-empty response."""
        resp = real_app.post("/chat", json={"message": "hello"})
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        assert len(data["response"]) > 0

    def test_provider_called_three_times(self, real_app):
        """Think + Act + Critique = at least 3 provider calls."""
        resp = real_app.post("/chat", json={"message": "hello"})
        assert resp.status_code == 200
        # MockProvider tracks call_count
        assert main_module._mirai.agent.provider.call_count >= 3

    def test_response_is_string(self, real_app):
        resp = real_app.post("/chat", json={"message": "test"})
        assert isinstance(resp.json()["response"], str)


# ---------------------------------------------------------------------------
# Integration: /chat/stream (SSE)
# ---------------------------------------------------------------------------


class TestIntegrationStream:
    def test_stream_full_pipeline(self, real_app):
        """SSE stream should contain thinking, chunks, and done events."""
        resp = real_app.post("/chat/stream", json={"message": "hello"})
        assert resp.status_code == 200
        body = resp.text

        assert "event: thinking" in body
        assert "event: done" in body
        assert "event: chunk" in body

    def test_stream_done_contains_full_text(self, real_app):
        """The 'done' event data should contain the full refined response."""
        resp = real_app.post("/chat/stream", json={"message": "hello"})
        body = resp.text

        # Parse SSE to find done event data
        for line in body.split("\n"):
            if line.startswith("event: done"):
                # Next line starting with "data:" has the full text
                continue
            if "data:" in line and "event: done" in body[: body.index(line)]:
                data = line.split("data: ", 1)[-1]
                assert len(data) > 0
                break

    def test_stream_thinking_not_empty(self, real_app):
        """The thinking event should contain monologue text."""
        resp = real_app.post("/chat/stream", json={"message": "hello"})
        body = resp.text

        # Find thinking data
        lines = body.split("\n")
        for i, line in enumerate(lines):
            if line == "event: thinking":
                data_line = lines[i + 1]
                assert data_line.startswith("data: ")
                thinking_text = data_line[6:]
                assert len(thinking_text) > 0
                break


# ---------------------------------------------------------------------------
# Integration: /ws/chat (WebSocket)
# ---------------------------------------------------------------------------


class TestIntegrationWebSocket:
    def test_websocket_full_conversation(self, real_app):
        """WebSocket should relay all stream events from real AgentLoop."""
        with real_app.websocket_connect("/ws/chat") as ws:
            ws.send_json({"message": "hello"})

            events = []
            for _ in range(50):  # Safety limit
                data = ws.receive_json()
                events.append(data)
                if data.get("event") == "done":
                    break

            event_types = [e["event"] for e in events]
            assert "thinking" in event_types
            assert "done" in event_types
            assert events[-1]["event"] == "done"
            assert len(events[-1]["data"]) > 0

    def test_websocket_multi_turn(self, real_app):
        """Multiple messages in the same WebSocket session should work."""
        with real_app.websocket_connect("/ws/chat") as ws:
            for turn in range(2):
                ws.send_json({"message": f"turn {turn}"})

                events = []
                for _ in range(50):
                    data = ws.receive_json()
                    events.append(data)
                    if data.get("event") == "done":
                        break

                assert events[-1]["event"] == "done"


# ---------------------------------------------------------------------------
# Integration: /health (with real agent)
# ---------------------------------------------------------------------------


class TestIntegrationHealth:
    def test_health_with_real_agent(self, real_app):
        """Health check should report MockProvider info."""
        resp = real_app.get("/health")
        data = resp.json()
        assert data["status"] == "ok"
        assert data["agent_ready"] is True
        assert data["provider"] == "MockProvider"
        assert data["model"] == "mock-model"

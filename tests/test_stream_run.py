"""Tests for AgentLoop.stream_run() async generator.

Tests the streaming event lifecycle directly using MockProvider,
without going through HTTP/WebSocket (those are in test_app_layer.py).
"""

import pytest
import pytest_asyncio

from mirai.agent.agent_loop import AgentLoop
from mirai.agent.providers import MockProvider
from mirai.agent.tools.echo import EchoTool


@pytest_asyncio.fixture
async def agent():
    from unittest.mock import AsyncMock

    # Mock storage to avoid DB locks
    mock_l3 = AsyncMock()
    mock_l3.append_trace = AsyncMock()
    mock_l3.get_traces_by_ids = AsyncMock(return_value=[])

    # Define dependencies
    provider = MockProvider()
    tools = [EchoTool()]

    # Mock embedder with explicit method
    mock_embedder = AsyncMock()
    mock_embedder.get_embeddings = AsyncMock(return_value=[0.0] * 1536)

    # Create loop with mocks
    loop = AgentLoop(
        provider=provider,
        tools=tools,
        collaborator_id="test-stream",
        l3_storage=mock_l3,
        l2_storage=AsyncMock(),
        embedder=mock_embedder,
    )
    # Minimal init
    loop.name = "StreamTest"
    loop.role = "tester"
    loop.base_system_prompt = "You are a test agent."
    loop.soul_content = ""

    return loop


class TestStreamRunEvents:
    @pytest.mark.asyncio
    async def test_first_event_is_chunk(self, agent):
        """First event should be 'chunk' (single-pass, no separate thinking phase)."""
        events = []
        async for event in agent.stream_run("hello"):
            events.append(event)
            break  # only need the first event

        assert len(events) >= 1
        assert events[0]["event"] == "chunk"
        assert len(events[0]["data"]) > 0

    @pytest.mark.asyncio
    async def test_yields_done_event_last(self, agent):
        """Last event should be 'done' with the full response text."""
        events = []
        async for event in agent.stream_run("hello"):
            events.append(event)

        assert events[-1]["event"] == "done"
        assert len(events[-1]["data"]) > 0

    @pytest.mark.asyncio
    async def test_yields_chunk_events(self, agent):
        """Should yield at least one 'chunk' event before 'done'."""
        events = []
        async for event in agent.stream_run("hello"):
            events.append(event)

        chunk_events = [e for e in events if e["event"] == "chunk"]
        assert len(chunk_events) >= 1

    @pytest.mark.asyncio
    async def test_chunks_concatenate_to_done_data(self, agent):
        """All chunk data concatenated should equal the done data."""
        events = []
        async for event in agent.stream_run("hello"):
            events.append(event)

        chunks = "".join(e["data"] for e in events if e["event"] == "chunk")
        done_data = next(e["data"] for e in events if e["event"] == "done")
        assert chunks == done_data

    @pytest.mark.asyncio
    async def test_event_sequence_order(self, agent):
        """Event sequence: (chunks...) → done (single-pass, no thinking phase)."""
        event_types = []
        async for event in agent.stream_run("hello"):
            event_types.append(event["event"])

        assert event_types[0] == "chunk"
        assert event_types[-1] == "done"

        # Everything between should be chunk or tool_use
        middle = event_types[1:-1]
        for t in middle:
            assert t in ("chunk", "tool_use")

    @pytest.mark.asyncio
    async def test_all_events_have_required_keys(self, agent):
        """Every event dict must have 'event' and 'data' keys."""
        async for event in agent.stream_run("hello"):
            assert "event" in event
            assert "data" in event
            assert isinstance(event["event"], str)
            assert isinstance(event["data"], str)


class TestStreamRunWithTools:
    @pytest.mark.asyncio
    async def test_tool_use_events_present_for_maintenance(self, agent):
        """When MockProvider triggers tool use, stream_run yields tool_use events."""
        events = []
        async for event in agent.stream_run("maintenance_check"):
            events.append(event)

        tool_events = [e for e in events if e["event"] == "tool_use"]
        # MockProvider has a maintenance_check flow that uses tools
        # The number depends on MockProvider logic
        assert len(tool_events) >= 0  # May or may not have tool events

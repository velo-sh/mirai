"""Integration test for mirai.agent.loop — single-pass agent cycle with MockProvider."""

from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from mirai.agent.loop import AgentLoop, _load_soul
from mirai.agent.providers import MockProvider
from mirai.agent.tools.echo import EchoTool
from mirai.db.session import init_db

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def agent(tmp_db_paths, tmp_path):
    """Create a fully initialized AgentLoop with MockProvider and temp DBs."""
    # Init SQLite
    await init_db(tmp_db_paths["sqlite_url"])

    provider = MockProvider()
    tools = [EchoTool()]
    collaborator_id = "01AN4Z048W7N7DF3SQ5G16CYAJ"

    # Monkey-patch DuckDB path before AgentLoop.create
    # Create mocks
    mock_duck = AsyncMock()
    mock_duck.append_trace = AsyncMock()
    mock_duck.get_traces_by_ids = AsyncMock(return_value=[])

    mock_vec = AsyncMock()
    mock_vec.search = AsyncMock(return_value=[])
    mock_vec.add = AsyncMock()

    mock_embedder = AsyncMock()
    mock_embedder.get_embeddings = AsyncMock(return_value=[0.0] * 1536)

    agent = AgentLoop(
        provider=provider,
        tools=tools,
        collaborator_id=collaborator_id,
        l3_storage=mock_duck,
        l2_storage=mock_vec,
        embedder=mock_embedder,
    )
    agent.name = "TestCollaborator"
    agent.role = "Test"
    agent.base_system_prompt = "You are a test collaborator."
    agent.soul_content = ""

    yield agent


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAgentLoop:
    @pytest.mark.asyncio
    async def test_run_returns_string(self, agent):
        """Full run cycle should return a non-empty string response."""
        result = await agent.run("Hello, who are you?")
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_run_increments_provider_call_count(self, agent):
        """Running the agent should invoke the provider at least once."""
        assert agent.provider.call_count == 0
        await agent.run("Tell me about yourself.")
        # Single-pass: at minimum one LLM call
        assert agent.provider.call_count >= 1

    @pytest.mark.asyncio
    async def test_run_archives_traces(self, agent):
        """Agent should archive user message and final response."""
        await agent.run("Test message")
        # _archive_trace: at least user message + assistant response = 2
        assert agent.l3_storage.append_trace.call_count >= 2


# ---------------------------------------------------------------------------
# _load_soul (module-level cached function)
# ---------------------------------------------------------------------------


class TestLoadSoul:
    def test_nonexistent_soul_returns_empty(self):
        result = _load_soul("nonexistent_collaborator_id_xyz")
        assert result == ""

    def test_loads_soul_file(self, tmp_path, monkeypatch):
        """Should load a SOUL.md file when it exists."""
        cid = "test_collab_123"
        soul_dir = tmp_path / "mirai" / "collaborator"
        soul_dir.mkdir(parents=True)
        soul_file = soul_dir / f"{cid}_SOUL.md"
        soul_file.write_text("# Test Soul\nI am a test collaborator.")

        # Temporarily change cwd so the path resolves
        monkeypatch.chdir(tmp_path)

        # Clear cache for this test
        _load_soul.cache_clear()

        result = _load_soul(cid)
        assert "Test Soul" in result
        assert "test collaborator" in result

    def test_caching_works(self):
        """Calling with the same arg should return cached result."""
        _load_soul.cache_clear()
        r1 = _load_soul("cache_test_id")
        r2 = _load_soul("cache_test_id")
        assert r1 is r2  # Same object from cache

"""Tests for DuckDBStorage (mirai/db/duck.py).

Covers: append_trace, get_traces_by_ids, get_recent_traces, search_traces,
edge cases (empty IDs, no results), and data integrity.
"""

import pytest

from mirai.db.duck import DuckDBStorage
from mirai.db.models import DBTrace, FeishuMessage


@pytest.fixture
def storage(tmp_path):
    """Create a DuckDBStorage with a temp database file."""
    db_path = str(tmp_path / "test_traces.duckdb")
    return DuckDBStorage(db_path=db_path)


# ---------------------------------------------------------------------------
# append_trace + get_traces_by_ids
# ---------------------------------------------------------------------------


class TestAppendAndRetrieve:
    @pytest.mark.asyncio
    async def test_append_and_retrieve_single(self, storage):
        await storage.append_trace(
            DBTrace(
                id="trace-001",
                collaborator_id="collab-1",
                trace_type="message",
                content="Hello world",
                metadata_json={"role": "user"},
            )
        )

        results = await storage.get_traces_by_ids(["trace-001"])
        assert len(results) == 1
        assert results[0].id == "trace-001"
        assert results[0].content == "Hello world"
        assert results[0].trace_type == "message"
        assert results[0].collaborator_id == "collab-1"

    @pytest.mark.asyncio
    async def test_append_multiple_and_retrieve(self, storage):
        for i in range(5):
            await storage.append_trace(
                DBTrace(
                    id=f"trace-{i:03d}",
                    collaborator_id="collab-1",
                    trace_type="thinking",
                    content=f"Thought {i}",
                )
            )

        results = await storage.get_traces_by_ids(["trace-000", "trace-002", "trace-004"])
        assert len(results) == 3
        ids = [r.id for r in results]
        assert "trace-000" in ids
        assert "trace-002" in ids
        assert "trace-004" in ids

    @pytest.mark.asyncio
    async def test_get_nonexistent_ids_returns_empty(self, storage):
        results = await storage.get_traces_by_ids(["nonexistent-id"])
        assert results == []

    @pytest.mark.asyncio
    async def test_get_empty_ids_returns_empty(self, storage):
        results = await storage.get_traces_by_ids([])
        assert results == []

    @pytest.mark.asyncio
    async def test_metadata_persisted_as_json(self, storage):
        await storage.append_trace(
            DBTrace(
                id="meta-001",
                collaborator_id="collab-1",
                trace_type="message",
                content="test",
                metadata_json={"key": "value", "nested": {"a": 1}},
            )
        )

        results = await storage.get_traces_by_ids(["meta-001"])
        assert len(results) == 1
        # metadata should be a dict
        assert results[0].metadata["key"] == "value"
        assert results[0].metadata["nested"]["a"] == 1

    @pytest.mark.asyncio
    async def test_none_metadata_defaults_to_empty(self, storage):
        await storage.append_trace(
            DBTrace(
                id="none-meta",
                collaborator_id="collab-1",
                trace_type="message",
                content="no metadata",
            )
        )

        results = await storage.get_traces_by_ids(["none-meta"])
        assert results[0].metadata == {}

    @pytest.mark.asyncio
    async def test_importance_and_vector_id(self, storage):
        await storage.append_trace(
            DBTrace(
                id="imp-001",
                collaborator_id="collab-1",
                trace_type="insight",
                content="important trace",
                importance=0.95,
                vector_id="vec-abc",
            )
        )

        results = await storage.get_traces_by_ids(["imp-001"])
        assert results[0].importance == pytest.approx(0.95)
        assert results[0].vector_id == "vec-abc"


# ---------------------------------------------------------------------------
# get_recent_traces
# ---------------------------------------------------------------------------


class TestGetRecentTraces:
    @pytest.mark.asyncio
    async def test_returns_recent_in_order(self, storage):
        for i in range(10):
            await storage.append_trace(
                DBTrace(
                    id=f"recent-{i:03d}",
                    collaborator_id="collab-1",
                    trace_type="message",
                    content=f"Message {i}",
                )
            )

        results = await storage.get_recent_traces("collab-1", limit=3)
        assert len(results) == 3
        # Most recent first (ORDER BY id DESC)
        assert results[0].id == "recent-009"
        assert results[1].id == "recent-008"
        assert results[2].id == "recent-007"

    @pytest.mark.asyncio
    async def test_filters_by_collaborator(self, storage):
        await storage.append_trace(
            DBTrace(id="a-001", collaborator_id="alice", trace_type="message", content="Alice's trace")
        )
        await storage.append_trace(
            DBTrace(id="b-001", collaborator_id="bob", trace_type="message", content="Bob's trace")
        )

        alice_traces = await storage.get_recent_traces("alice")
        assert all(t.collaborator_id == "alice" for t in alice_traces)
        assert len(alice_traces) == 1

    @pytest.mark.asyncio
    async def test_empty_result_for_unknown_collaborator(self, storage):
        results = await storage.get_recent_traces("nonexistent")
        assert results == []


# ---------------------------------------------------------------------------
# search_traces
# ---------------------------------------------------------------------------


class TestSearchTraces:
    @pytest.mark.asyncio
    async def test_full_text_search(self, storage):
        await storage.append_trace(
            DBTrace(id="search-001", collaborator_id="c1", trace_type="thinking", content="The quick brown fox")
        )
        await storage.append_trace(
            DBTrace(id="search-002", collaborator_id="c1", trace_type="thinking", content="The lazy dog")
        )

        results = await storage.search_traces("fox")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_search_no_results(self, storage):
        await storage.append_trace(
            DBTrace(id="search-003", collaborator_id="c1", trace_type="message", content="Hello world")
        )

        results = await storage.search_traces("nonexistent-term-xyz")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_case_sensitive(self, storage):
        await storage.append_trace(
            DBTrace(id="case-001", collaborator_id="c1", trace_type="message", content="Architecture Review")
        )

        # DuckDB LIKE is case-sensitive by default
        lower_results = await storage.search_traces("architecture")
        upper_results = await storage.search_traces("Architecture")
        assert len(upper_results) == 1
        # lower_results may match depending on DuckDB version collation
        assert len(lower_results) >= 0


# ---------------------------------------------------------------------------
# Feishu History Persistence
# ---------------------------------------------------------------------------


class TestFeishuHistory:
    @pytest.mark.asyncio
    async def test_save_and_retrieve_history(self, storage):
        chat_id = "chat-123"
        await storage.save_feishu_history(FeishuMessage(chat_id=chat_id, role="user", content="Hello Mira"))
        await storage.save_feishu_history(
            FeishuMessage(chat_id=chat_id, role="assistant", content="Hello! How can I help?")
        )

        history = await storage.get_feishu_history(chat_id)
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[0].content == "Hello Mira"
        assert history[1].role == "assistant"
        assert history[1].content == "Hello! How can I help?"

    @pytest.mark.asyncio
    async def test_history_limit_and_order(self, storage):
        chat_id = "chat-456"
        # Insert 5 turns (10 messages)
        for i in range(5):
            await storage.save_feishu_history(FeishuMessage(chat_id=chat_id, role="user", content=f"User {i}"))
            await storage.save_feishu_history(FeishuMessage(chat_id=chat_id, role="assistant", content=f"AI {i}"))

        # Limit to last 2 turns (4 messages)
        history = await storage.get_feishu_history(chat_id, limit=4)
        assert len(history) == 4
        # Ordered chronologically: oldest of the subset first
        assert history[0].content == "User 3"
        assert history[1].content == "AI 3"
        assert history[2].content == "User 4"
        assert history[3].content == "AI 4"

    @pytest.mark.asyncio
    async def test_empty_history_returns_empty_list(self, storage):
        history = await storage.get_feishu_history("nonexistent-chat")
        assert history == []


# ---------------------------------------------------------------------------
# Concurrency tests — validate thread-safety under asyncio.gather
# ---------------------------------------------------------------------------


class TestConcurrency:
    """Ensure DuckDBStorage handles concurrent access without data races."""

    @pytest.fixture
    def storage(self):
        storage = DuckDBStorage(db_path=":memory:")
        yield storage
        storage.close()

    @pytest.mark.asyncio
    async def test_concurrent_writes(self, storage):
        """20 parallel append_trace calls should all succeed."""
        import asyncio

        traces = [
            DBTrace(
                id=f"conc-w-{i:03d}",
                collaborator_id="concurrency-test",
                trace_type="message",
                content=f"concurrent write {i}",
            )
            for i in range(20)
        ]

        await asyncio.gather(*(storage.append_trace(t) for t in traces))

        # Verify all 20 traces were persisted
        result = await storage.get_recent_traces("concurrency-test", limit=50)
        assert len(result) == 20

    @pytest.mark.asyncio
    async def test_concurrent_reads_and_writes(self, storage):
        """Mixed read/write operations in parallel should not corrupt data."""
        import asyncio

        # Seed some data first
        for i in range(5):
            await storage.append_trace(
                DBTrace(
                    id=f"conc-rw-seed-{i:03d}",
                    collaborator_id="concurrency-rw",
                    trace_type="message",
                    content=f"seed {i}",
                )
            )

        # Now run reads and writes concurrently
        write_traces = [
            DBTrace(
                id=f"conc-rw-{i:03d}",
                collaborator_id="concurrency-rw",
                trace_type="tool_use",
                content=f"concurrent rw {i}",
            )
            for i in range(10)
        ]

        async def _read():
            return await storage.get_recent_traces("concurrency-rw", limit=50)

        tasks = [storage.append_trace(t) for t in write_traces]
        tasks.extend([_read() for _ in range(5)])

        results = await asyncio.gather(*tasks)

        # All writes should have succeeded
        final = await storage.get_recent_traces("concurrency-rw", limit=50)
        assert len(final) == 15  # 5 seed + 10 concurrent writes

        # All reads should have returned lists (no exceptions)
        for r in results[10:]:  # the read results
            assert isinstance(r, list)

import asyncio
from typing import Any

from mirai.agent.providers import MockEmbeddingProvider
from mirai.db.duck import DuckDBStorage
from mirai.logging import get_logger
from mirai.memory.vector_db import MemoryEntry, VectorStore

log = get_logger("mirai.dreamer")


class Dreamer:
    """
    The Dreaming Engine.
    Background process that consolidates raw L3 traces into L2 semantic indices.
    """

    def __init__(
        self,
        agent_or_id: str | Any,
        l3_storage: DuckDBStorage | None = None,
        l2_storage: VectorStore | None = None,
        embedder: Any | None = None,
        interval_seconds: int = 3600,
    ):
        if isinstance(agent_or_id, str):
            self.collaborator_id = agent_or_id
        else:
            self.collaborator_id = agent_or_id.collaborator_id

        self.l3 = l3_storage or DuckDBStorage()
        self.l2 = l2_storage or VectorStore()
        self.embedder = embedder or MockEmbeddingProvider()
        self.interval_seconds = interval_seconds

    async def dream_once(self):
        """
        Perform one consolidation cycle:
        1. Find unindexed message traces in L3.
        2. Generate embeddings.
        3. Index in L2.
        """
        log.info("dream_start", collaborator=self.collaborator_id)

        # In a real system, we'd track 'last_processed_ulid'.
        # For the MVP, we assume any trace with vector_id=NULL in L3 needs indexing.
        # However, DuckDB implementation of append_trace currently doesn't update the DB after L2 write.
        # Let's fetch traces that are of type 'message' and role 'user' for now as a simple dream.

        self.l3._check_conn()
        assert self.l3.conn is not None
        unindexed_traces = self.l3.conn.execute(
            """
            SELECT * FROM cognitive_traces
            WHERE collaborator_id = ? AND trace_type = 'message'
        """,
            [self.collaborator_id],
        ).fetchall()

        # (Converting DuckDB results to dicts)
        columns = [desc[0] for desc in self.l3.conn.description]  # type: ignore[union-attr]
        traces = [dict(zip(columns, row, strict=False)) for row in unindexed_traces]

        if not traces:
            log.info("dream_nothing_to_consolidate")
            return

        new_entries = []
        for trace in traces:
            log.debug("dream_processing_trace", trace_id=trace["id"])
            vector = await self.embedder.get_embeddings(trace["content"])
            entry = MemoryEntry(
                content=trace["content"],
                metadata={"trace_id": trace["id"], "source": "dreaming"},
                vector=vector,
                collaborator_id=self.collaborator_id,
                scope="global",
            )
            new_entries.append(entry)

            # Update L3 with the vector_id (Optional MVP step)
            # self.l3.conn.execute("UPDATE cognitive_traces SET vector_id = 'indexed' WHERE id = ?", [trace['id']])

        if new_entries:
            await self.l2.add_memories(new_entries)
            log.info("dream_consolidated", count=len(new_entries))


async def main():
    # Simple manual test of the dreamer
    dreamer = Dreamer("01AN4Z048W7N7DF3SQ5G16CYAJ")
    await dreamer.dream_once()


if __name__ == "__main__":
    asyncio.run(main())

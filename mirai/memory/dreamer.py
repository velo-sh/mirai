import asyncio
from typing import List, Optional
from mirai.db.duck import DuckDBStorage
from mirai.memory.vector_db import VectorStore, MemoryEntry
from mirai.agent.providers import MockEmbeddingProvider
import json

class Dreamer:
    """
    The Dreaming Engine.
    Background process that consolidates raw L3 traces into L2 semantic indices.
    """
    def __init__(self, collaborator_id: str):
        self.collaborator_id = collaborator_id
        self.l3 = DuckDBStorage()
        self.l2 = VectorStore()
        self.embedder = MockEmbeddingProvider()

    async def dream_once(self):
        """
        Perform one consolidation cycle:
        1. Find unindexed message traces in L3.
        2. Generate embeddings.
        3. Index in L2.
        """
        print(f"[dreamer] {self.collaborator_id} is starting to dream...")
        
        # In a real system, we'd track 'last_processed_ulid'.
        # For the MVP, we assume any trace with vector_id=NULL in L3 needs indexing.
        # However, DuckDB implementation of append_trace currently doesn't update the DB after L2 write.
        # Let's fetch traces that are of type 'message' and role 'user' for now as a simple dream.
        
        unindexed_traces = self.l3.conn.execute(f"""
            SELECT * FROM cognitive_traces 
            WHERE collaborator_id = ? AND trace_type = 'message'
        """, [self.collaborator_id]).fetchall()
        
        # (Converting DuckDB results to dicts)
        columns = [desc[0] for desc in self.l3.conn.description]
        traces = [dict(zip(columns, row)) for row in unindexed_traces]
        
        if not traces:
            print("[dreamer] Nothing to consolidate.")
            return

        new_entries = []
        for trace in traces:
            print(f"[dreamer] Processing trace: {trace['id']}")
            vector = await self.embedder.get_embeddings(trace['content'])
            entry = MemoryEntry(
                content=trace['content'],
                metadata={"trace_id": trace['id'], "source": "dreaming"},
                vector=vector,
                collaborator_id=self.collaborator_id,
                scope="global"
            )
            new_entries.append(entry)
            
            # Update L3 with the vector_id (Optional MVP step)
            # self.l3.conn.execute("UPDATE cognitive_traces SET vector_id = 'indexed' WHERE id = ?", [trace['id']])

        if new_entries:
            await self.l2.add_memories(new_entries)
            print(f"[dreamer] Successfully consolidated {len(new_entries)} memories.")

async def main():
    # Simple manual test of the dreamer
    dreamer = Dreamer("01AN4Z048W7N7DF3SQ5G16CYAJ")
    await dreamer.dream_once()

if __name__ == "__main__":
    asyncio.run(main())

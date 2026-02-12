from typing import Any

from mirai.agent.tools.base import BaseTool


class MemorizeTool(BaseTool):
    def __init__(self, collaborator_id: str):
        self.collaborator_id = collaborator_id

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "name": "memorize",
            "description": "Archives important information to long-term memory (L3/HDD). Use this for technical decisions, project state changes, or user preferences.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The information to preserve."},
                    "importance": {
                        "type": "number",
                        "description": "A score from 0.0 to 1.0 indicating memory significance.",
                        "minimum": 0,
                        "maximum": 1,
                    },
                },
                "required": ["content", "importance"],
            },
        }

    async def execute(self, content: str, importance: float) -> str:  # type: ignore[override]
        # 1. Archive to L3 (HDD) using DuckDB
        from ulid import ULID

        from mirai.db.duck import DuckDBStorage

        l3 = DuckDBStorage()
        trace_id = str(ULID())
        await l3.append_trace(
            id=trace_id,
            collaborator_id=self.collaborator_id,
            trace_type="insight",
            content=content,
            importance=importance,
            metadata={"source": "manual_memorize"},
        )

        # 2. Index in L2 (RAM - Vector Store)
        from mirai.agent.providers import MockEmbeddingProvider
        from mirai.memory.vector_db import MemoryEntry, VectorStore

        embedder = MockEmbeddingProvider()
        vector = await embedder.get_embeddings(content)

        vdb = VectorStore()
        entry = MemoryEntry(
            content=content,
            metadata={"trace_id": trace_id},
            vector=vector,
            collaborator_id=self.collaborator_id,
            scope="manual",
        )
        await vdb.add_memories([entry])

        return f"Successfully archived and indexed insight with importance {importance}."

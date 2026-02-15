from typing import Any

from ulid import ULID

from mirai.agent.providers import MockEmbeddingProvider
from mirai.agent.tools.base import BaseTool
from mirai.db.duck import DuckDBStorage
from mirai.db.models import DBTrace
from mirai.memory.vector_db import MemoryEntry, VectorStore


class MemorizeTool(BaseTool):
    def __init__(self, context: Any = None, **kwargs: Any) -> None:
        super().__init__(context)
        # Convenience aliases from ToolContext
        self.collaborator_id = (
            self.context.config.agent.collaborator_id if self.context and self.context.config else "unknown"
        )
        self.l3_storage = self.context.storage if self.context else None
        self.l2_storage = self.context.agent_loop.l2_storage if self.context and self.context.agent_loop else None

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
        l3 = self.l3_storage or DuckDBStorage()
        trace = DBTrace(
            id=str(ULID()),
            collaborator_id=self.collaborator_id,
            trace_type="insight",
            content=content,
            importance=importance,
            metadata_json={"source": "manual_memorize"},
        )
        await l3.append_trace(trace)
        trace_id = trace.id

        # 2. Index in L2 (RAM - Vector Store)

        embedder = MockEmbeddingProvider()
        vector = await embedder.get_embeddings(content)

        vdb = self.l2_storage or VectorStore()
        entry = MemoryEntry(
            content=content,
            metadata={"trace_id": trace_id},
            vector=vector,
            collaborator_id=self.collaborator_id,
            scope="manual",
        )
        await vdb.add_memories([entry])

        return f"Successfully archived and indexed insight with importance {importance}."

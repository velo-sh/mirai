from mirai.agent.tools.base import BaseTool
from mirai.memory.models import CognitiveTrace
from mirai.db.session import async_session
from typing import Dict, Any
import json

class MemorizeTool(BaseTool):
    def __init__(self, collaborator_id: int):
        self.collaborator_id = collaborator_id

    @property
    def definition(self) -> Dict[str, Any]:
        return {
            "name": "memorize",
            "description": "Archives important information to long-term memory (L3/HDD). Use this for technical decisions, project state changes, or user preferences.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The information to preserve."
                    },
                    "importance": {
                        "type": "number",
                        "description": "A score from 0.0 to 1.0 indicating memory significance.",
                        "minimum": 0,
                        "maximum": 1
                    }
                },
                "required": ["content", "importance"]
            }
        }

    async def execute(self, content: str, importance: float) -> str:
        # 1. Archive to L3 (HDD)
        async with async_session() as session:
            trace = CognitiveTrace(
                collaborator_id=self.collaborator_id,
                trace_type="insight",
                content=content,
                importance=importance,
                metadata_json=json.dumps({"source": "manual_memorize"})
            )
            session.add(trace)
            await session.commit()
            await session.refresh(trace)
            
        # 2. Index in L2 (RAM - Vector Store)
        # For the MVP, we assume a local MockEmbeddingProvider for now
        from mirai.agent.providers import MockEmbeddingProvider
        from mirai.memory.vector_db import VectorStore, MemoryEntry
        
        embedder = MockEmbeddingProvider()
        vector = await embedder.get_embeddings(content)
        
        vdb = VectorStore()
        entry = MemoryEntry(
            content=content,
            metadata={"trace_id": str(trace.id)},
            vector=vector,
            collaborator_id=self.collaborator_id,
            scope="manual"
        )
        await vdb.add_memories([entry])
            
        return f"Successfully archived and indexed insight with importance {importance}."

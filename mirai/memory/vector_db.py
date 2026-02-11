import lancedb
import pyarrow as pa
import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class MemoryEntry(BaseModel):
    content: str
    metadata: Dict[str, Any]
    vector: List[float]
    collaborator_id: str
    scope: str  # e.g., "global", "group", "private"

class VectorStore:
    def __init__(self, db_path: str = "./mirai_vectors"):
        self.db_path = db_path
        self.db = lancedb.connect(db_path)
        self.table_name = "memories"
        
    def _get_schema(self, dim: int):
        return pa.schema([
            pa.field("vector", pa.list_(pa.float32(), dim)),
            pa.field("content", pa.string()),
            pa.field("collaborator_id", pa.string()),
            pa.field("scope", pa.string()),
            pa.field("metadata", pa.string()), # Store as JSON string
        ])

    async def add_memories(self, entries: List[MemoryEntry]):
        if not entries:
            return
            
        dim = len(entries[0].vector)
        import json
        
        data = [
            {
                "vector": entry.vector,
                "content": entry.content,
                "collaborator_id": entry.collaborator_id,
                "scope": entry.scope,
                "metadata": json.dumps(entry.metadata)
            }
            for entry in entries
        ]
        
        if self.table_name in self.db.table_names():
            table = self.db.open_table(self.table_name)
            table.add(data)
        else:
            self.db.create_table(self.table_name, data=data, schema=self._get_schema(dim))

    async def search(self, vector: List[float], limit: int = 5, filter: Optional[str] = None):
        if self.table_name not in self.db.list_tables():
            return []
            
        table = self.db.open_table(self.table_name)
        query = table.search(vector).limit(limit)
        if filter:
            query = query.where(filter)
        
        return query.to_list()

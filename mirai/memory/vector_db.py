import os
from typing import Any

import lancedb
import orjson
import pyarrow as pa
from pydantic import BaseModel


class MemoryEntry(BaseModel):
    content: str
    metadata: dict[str, Any]
    vector: list[float]
    collaborator_id: str
    scope: str  # e.g., "global", "group", "private"


class VectorStore:
    def __init__(self, db_path: str | None = None):
        if db_path is None:
            # Standardize on a project-relative path
            cwd = os.getcwd()
            db_path = os.path.join(cwd, "mirai_vectors")
        self.db_path = db_path
        self.db = lancedb.connect(db_path)
        self.table_name = "memories"

    def _get_schema(self, dim: int):
        return pa.schema(
            [
                pa.field("vector", pa.list_(pa.float32(), dim)),
                pa.field("content", pa.string()),
                pa.field("collaborator_id", pa.string()),
                pa.field("scope", pa.string()),
                pa.field("metadata", pa.string()),  # Store as JSON string
            ]
        )

    async def add_memories(self, entries: list[MemoryEntry]):
        if not entries:
            return

        dim = len(entries[0].vector)
        data = [
            {
                "vector": entry.vector,
                "content": entry.content,
                "collaborator_id": entry.collaborator_id,
                "scope": entry.scope,
                "metadata": orjson.dumps(entry.metadata).decode(),
            }
            for entry in entries
        ]

        print(f"DEBUG: VectorStore adding {len(data)} entries to {self.db_path}")
        if self.table_name in self.db.list_tables():
            table = self.db.open_table(self.table_name)
            table.add(data)
            print(f"DEBUG: Added rows. Total: {table.count_rows()}")
        else:
            print("DEBUG: Creating table")
            table = self.db.create_table(self.table_name, data=data, schema=self._get_schema(dim), exist_ok=True)
            print(f"DEBUG: Created table. Rows: {table.count_rows()}")

    async def search(self, vector: list[float], limit: int = 5, filter: str | None = None):
        try:
            table = self.db.open_table(self.table_name)
        except Exception:
            return []

        # print(f"DEBUG: Search rows: {table.count_rows()}. Filter: {filter}")
        # ...

        table = self.db.open_table(self.table_name)
        query = table.search(vector).limit(limit)
        if filter:
            query = query.where(filter)
        return query.to_list()

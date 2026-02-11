import duckdb
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import json

class DuckDBStorage:
    def __init__(self, db_path: str = "mirai_hdd.duckdb"):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self._init_schema()

    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cognitive_traces (
                id VARCHAR PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                collaborator_id VARCHAR,
                trace_type VARCHAR,
                content TEXT,
                metadata_json JSON,
                importance DOUBLE,
                vector_id VARCHAR
            )
        """)

    async def append_trace(self, 
                     id: str, 
                     collaborator_id: str, 
                     trace_type: str, 
                     content: str, 
                     metadata: Dict[str, Any] = None,
                     importance: float = 0.0,
                     vector_id: str = None):
        
        metadata_json = json.dumps(metadata or {})
        self.conn.execute("""
            INSERT INTO cognitive_traces 
            (id, collaborator_id, trace_type, content, metadata_json, importance, vector_id) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [id, collaborator_id, trace_type, content, metadata_json, importance, vector_id])

    async def get_recent_traces(self, collaborator_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        rel = self.conn.execute("""
            SELECT * FROM cognitive_traces 
            WHERE collaborator_id = ? 
            ORDER BY id DESC 
            LIMIT ?
        """, [collaborator_id, limit])
        return rel.fetchall()

    async def search_traces(self, query: str) -> List[Dict[str, Any]]:
        # DuckDB's full-text search capability
        # For now, a simple LIKE. In real Dreaming, we'd use FTS or vector pointers.
        rel = self.conn.execute("""
            SELECT * FROM cognitive_traces 
            WHERE content LIKE ? 
            ORDER BY id DESC
        """, [f"%{query}%"])
        return rel.fetchall()

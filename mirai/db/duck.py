"""DuckDB storage layer with proper async wrapping.

DuckDB has no native async driver, so all blocking I/O
is offloaded to a thread via ``asyncio.to_thread()``.
"""

import asyncio
from typing import Any

import duckdb
import orjson


class DuckDBStorage:
    def __init__(self, db_path: str = "mirai_hdd.duckdb"):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self._init_schema()

    def close(self):
        """Close the DuckDB connection and release the file lock."""
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None

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
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS feishu_history (
                chat_id VARCHAR,
                role VARCHAR,
                content TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def _execute(self, sql: str, params: list[Any] | None = None):
        """Execute a statement synchronously (called via to_thread)."""
        if params:
            return self.conn.execute(sql, params)
        return self.conn.execute(sql)

    def _fetch_dicts(self, sql: str, params: list[Any]) -> list[dict[str, Any]]:
        """Execute + fetchall as dicts (called via to_thread)."""
        rel = self.conn.execute(sql, params)
        columns = [desc[0] for desc in rel.description]
        return [dict(zip(columns, row, strict=False)) for row in rel.fetchall()]

    async def append_trace(
        self,
        id: str,
        collaborator_id: str,
        trace_type: str,
        content: str,
        metadata: dict[str, Any] = None,
        importance: float = 0.0,
        vector_id: str = None,
    ):
        metadata_json = orjson.dumps(metadata or {}).decode()
        await asyncio.to_thread(
            self._execute,
            """
            INSERT INTO cognitive_traces
            (id, collaborator_id, trace_type, content, metadata_json, importance, vector_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [id, collaborator_id, trace_type, content, metadata_json, importance, vector_id],
        )

    async def get_traces_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []

        placeholders = ", ".join(["?"] * len(ids))
        return await asyncio.to_thread(
            self._fetch_dicts,
            f"""
            SELECT * FROM cognitive_traces
            WHERE id IN ({placeholders})
            ORDER BY id ASC
            """,
            ids,
        )

    async def get_recent_traces(self, collaborator_id: str, limit: int = 10) -> list[dict[str, Any]]:
        return await asyncio.to_thread(
            self._fetch_dicts,
            """
            SELECT * FROM cognitive_traces
            WHERE collaborator_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            [collaborator_id, limit],
        )

    async def save_feishu_history(self, chat_id: str, role: str, content: str):
        """Save a message turn to the Feishu history table."""
        await asyncio.to_thread(
            self._execute,
            """
            INSERT INTO feishu_history (chat_id, role, content)
            VALUES (?, ?, ?)
            """,
            [chat_id, role, content],
        )

    async def get_feishu_history(self, chat_id: str, limit: int = 20) -> list[dict[str, str]]:
        """Retrieve recent conversation history for a specific chat."""

        def _query():
            rel = self.conn.execute(
                """
                SELECT role, content FROM feishu_history
                WHERE chat_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                [chat_id, limit],
            )
            # DuckDB returns latest first due to DESC, but LLM context needs chronological.
            rows = rel.fetchall()
            return [{"role": row[0], "content": row[1]} for row in reversed(rows)]

        return await asyncio.to_thread(_query)

    async def search_traces(self, query: str) -> list[tuple[Any, ...]]:
        """Full-text search on cognitive traces."""

        def _query():
            rel = self.conn.execute(
                """
                SELECT * FROM cognitive_traces
                WHERE content LIKE ?
                ORDER BY id DESC
                """,
                [f"%{query}%"],
            )
            return rel.fetchall()

        return await asyncio.to_thread(_query)

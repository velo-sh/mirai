"""DuckDB storage layer with proper async wrapping.

DuckDB has no native async driver, so all blocking I/O
is offloaded to a thread via ``asyncio.to_thread()``.
"""

import asyncio
from typing import Any

import duckdb
import orjson

from mirai.db.models import DBTrace, FeishuMessage
from mirai.errors import StorageError


class DuckDBStorage:
    def __init__(self, db_path: str = "mirai_hdd.duckdb"):
        self.db_path = db_path
        self.conn: duckdb.DuckDBPyConnection | None = duckdb.connect(db_path)
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
        assert self.conn is not None
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

    def _check_conn(self):
        """Raise if connection has been closed."""
        if self.conn is None:
            raise StorageError("DuckDB connection is closed. Reinitialize DuckDBStorage to reconnect.")

    def _execute(self, sql: str, params: list[Any] | None = None):
        """Execute a statement synchronously (called via to_thread)."""
        self._check_conn()
        assert self.conn is not None
        if params:
            return self.conn.execute(sql, params)
        return self.conn.execute(sql)

    def _fetch_dicts(self, sql: str, params: list[Any]) -> list[dict[str, Any]]:
        """Execute + fetchall as dicts (called via to_thread)."""
        self._check_conn()
        assert self.conn is not None
        rel = self.conn.execute(sql, params)
        columns = [desc[0] for desc in rel.description]
        return [dict(zip(columns, row, strict=False)) for row in rel.fetchall()]

    async def append_trace(self, trace: DBTrace | None = None, **kwargs: Any) -> None:
        """Append a cognitive trace using the DBTrace model."""
        if trace is None:
            # Legacy support for tests that pass keyword arguments directly
            trace = DBTrace.model_validate(kwargs)

        metadata_json = orjson.dumps(
            trace.metadata_json if hasattr(trace, "metadata_json") else trace.metadata
        ).decode()
        await asyncio.to_thread(
            self._execute,
            """
            INSERT INTO cognitive_traces
            (id, collaborator_id, trace_type, content, metadata_json, importance, vector_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                trace.id,
                trace.collaborator_id,
                trace.trace_type,
                trace.content,
                metadata_json,
                trace.importance,
                trace.vector_id,
            ],
        )

    async def get_traces_by_ids(self, ids: list[str]) -> list[DBTrace]:
        if not ids:
            return []

        placeholders = ", ".join(["?"] * len(ids))
        dicts = await asyncio.to_thread(
            self._fetch_dicts,
            f"""
            SELECT * FROM cognitive_traces
            WHERE id IN ({placeholders})
            ORDER BY id ASC
            """,
            ids,
        )
        return [DBTrace.model_validate(d) for d in dicts]

    async def get_recent_traces(self, collaborator_id: str, limit: int = 10) -> list[DBTrace]:
        dicts = await asyncio.to_thread(
            self._fetch_dicts,
            """
            SELECT * FROM cognitive_traces
            WHERE collaborator_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            [collaborator_id, limit],
        )
        return [DBTrace.model_validate(d) for d in dicts]

    async def save_feishu_history(self, msg: FeishuMessage):
        """Save a message turn using the FeishuMessage model."""
        await asyncio.to_thread(
            self._execute,
            """
            INSERT INTO feishu_history (chat_id, role, content)
            VALUES (?, ?, ?)
            """,
            [msg.chat_id, msg.role, msg.content],
        )

    async def get_feishu_history(self, chat_id: str, limit: int = 20) -> list[FeishuMessage]:
        """Retrieve recent conversation history as FeishuMessage models."""

        def _query():
            self._check_conn()
            assert self.conn is not None
            rel = self.conn.execute(
                """
                SELECT chat_id, role, content, timestamp FROM feishu_history
                WHERE chat_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                [chat_id, limit],
            )
            columns = [desc[0] for desc in rel.description]
            rows = rel.fetchall()
            # DuckDB returns latest first due to DESC, but context needs chronological.
            # We map to dict first to use model_validate
            dicts = [dict(zip(columns, row, strict=False)) for row in reversed(rows)]
            return [FeishuMessage.model_validate(d) for d in dicts]

        return await asyncio.to_thread(_query)

    async def search_traces(self, query: str) -> list[tuple[Any, ...]]:
        """Full-text search on cognitive traces."""

        def _query():
            self._check_conn()
            assert self.conn is not None
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

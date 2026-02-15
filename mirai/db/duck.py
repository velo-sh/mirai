"""DuckDB storage layer with proper async wrapping.

DuckDB has no native async driver, so all blocking I/O
is offloaded to a thread via ``asyncio.to_thread()``.

Thread-safety: DuckDB connections are **not** thread-safe.  Since
``asyncio.to_thread()`` dispatches to a thread-pool, we create an
independent *cursor* for each operation and serialize cursor creation
with a ``threading.Lock``.
"""

import asyncio
import threading
import time
from typing import Any

import duckdb
import orjson

from mirai.db.models import DBTrace, FeishuMessage
from mirai.errors import StorageError
from mirai.logging import get_logger

_log = get_logger("mirai.db.duck")

# Queries slower than this threshold (seconds) are logged at warning level.
_SLOW_QUERY_THRESHOLD = 0.1


class DuckDBStorage:
    def __init__(self, db_path: str = "mirai_hdd.duckdb"):
        self.db_path = db_path
        self.conn: duckdb.DuckDBPyConnection | None = duckdb.connect(db_path)
        self._lock = threading.Lock()
        self._init_schema()

    @classmethod
    def for_testing(
        cls,
        *,
        conn: duckdb.DuckDBPyConnection | None = None,
    ) -> "DuckDBStorage":
        """Create a DuckDBStorage for testing without opening a database file.

        Useful for testing error paths (e.g. ``conn=None`` → ``_check_conn``
        raises ``StorageError``).
        """
        storage = cls.__new__(cls)
        storage.db_path = ":testing:"
        storage.conn = conn
        storage._lock = threading.Lock()
        return storage

    def close(self):
        """Close the DuckDB connection and release the file lock."""
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None

    # Schema version. Bump when adding new migrations.
    SCHEMA_VERSION = 1

    # Ordered list of migration functions: (version, callable).
    # Each callable receives the DuckDB connection.
    _MIGRATIONS: list[tuple[int, str]] = [
        (
            1,
            """
            CREATE TABLE IF NOT EXISTS cognitive_traces (
                id VARCHAR PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                collaborator_id VARCHAR,
                trace_type VARCHAR,
                content TEXT,
                metadata_json JSON,
                importance DOUBLE,
                vector_id VARCHAR
            );
            CREATE TABLE IF NOT EXISTS feishu_history (
                chat_id VARCHAR,
                role VARCHAR,
                content TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """,
        ),
        # Future migrations go here:
        # (2, "ALTER TABLE cognitive_traces ADD COLUMN source VARCHAR DEFAULT 'unknown';"),
    ]

    def _init_schema(self) -> None:
        assert self.conn is not None

        # Create schema_meta table if it doesn't exist
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_meta (
                key VARCHAR PRIMARY KEY,
                value VARCHAR
            )
        """)

        # Read current version
        result = self.conn.execute("SELECT value FROM schema_meta WHERE key = 'schema_version'").fetchone()
        current_version = int(result[0]) if result else 0

        # Apply pending migrations
        for version, sql in self._MIGRATIONS:
            if version > current_version:
                _log.info("applying_migration", version=version)
                self.conn.execute(sql)
                self.conn.execute(
                    """
                    INSERT INTO schema_meta (key, value)
                    VALUES ('schema_version', ?)
                    ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
                    """,
                    [str(version)],
                )
                current_version = version

        _log.debug("schema_ready", version=current_version)

    def _check_conn(self):
        """Raise if connection has been closed."""
        if self.conn is None:
            raise StorageError("DuckDB connection is closed. Reinitialize DuckDBStorage to reconnect.")

    def _execute(self, sql: str, params: list[Any] | None = None):
        """Execute a statement synchronously (called via to_thread).

        Creates an independent cursor so concurrent ``to_thread`` calls
        do not share mutable state.
        """
        self._check_conn()
        assert self.conn is not None
        with self._lock:
            cursor = self.conn.cursor()
        t0 = time.perf_counter()
        try:
            if params:
                return cursor.execute(sql, params)
            return cursor.execute(sql)
        finally:
            elapsed = time.perf_counter() - t0
            cursor.close()
            if elapsed >= _SLOW_QUERY_THRESHOLD:
                _log.warning("slow_query", sql=sql[:120], duration_ms=round(elapsed * 1000, 1))

    def _fetch_dicts(self, sql: str, params: list[Any]) -> list[dict[str, Any]]:
        """Execute + fetchall as dicts (called via to_thread)."""
        self._check_conn()
        assert self.conn is not None
        with self._lock:
            cursor = self.conn.cursor()
        t0 = time.perf_counter()
        try:
            rel = cursor.execute(sql, params)
            columns = [desc[0] for desc in rel.description]
            return [dict(zip(columns, row, strict=False)) for row in rel.fetchall()]
        finally:
            elapsed = time.perf_counter() - t0
            cursor.close()
            if elapsed >= _SLOW_QUERY_THRESHOLD:
                _log.warning("slow_query", sql=sql[:120], duration_ms=round(elapsed * 1000, 1))

    async def append_trace(self, trace: DBTrace | None = None, **kwargs: Any) -> None:
        """Append a cognitive trace using the DBTrace model."""
        if trace is None:
            # Legacy support for tests that pass keyword arguments directly
            trace = DBTrace.model_validate(kwargs)

        metadata_json = orjson.dumps(trace.metadata_json).decode()
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
            with self._lock:
                cursor = self.conn.cursor()
            try:
                rel = cursor.execute(
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
            finally:
                cursor.close()
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
            with self._lock:
                cursor = self.conn.cursor()
            try:
                rel = cursor.execute(
                    """
                    SELECT * FROM cognitive_traces
                    WHERE content LIKE ?
                    ORDER BY id DESC
                    """,
                    [f"%{query}%"],
                )
                return rel.fetchall()
            finally:
                cursor.close()

        return await asyncio.to_thread(_query)

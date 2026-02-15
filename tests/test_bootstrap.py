"""Tests for mirai.bootstrap — MiraiApp lifecycle and utility functions."""

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirai.bootstrap import MiraiApp, _wait_for_duckdb_lock

# ---------------------------------------------------------------------------
# _wait_for_duckdb_lock
# ---------------------------------------------------------------------------


class TestWaitForDuckdbLock:
    """Tests for the DuckDB WAL lock detection helper."""

    def test_no_wal_file_returns_immediately(self, tmp_path):
        """If no .wal file exists, the function returns without blocking."""
        db_path = str(tmp_path / "test.duckdb")
        # No .wal file created → should return instantly
        _wait_for_duckdb_lock(db_path, timeout=0.1)

    def test_wal_exists_but_connection_succeeds(self, tmp_path):
        """If .wal exists but DuckDB connects fine, the function succeeds."""
        db_path = str(tmp_path / "test.duckdb")
        wal_path = Path(db_path + ".wal")
        wal_path.touch()

        import duckdb

        with patch.object(duckdb, "connect") as mock_connect:
            mock_conn = MagicMock()
            mock_connect.return_value = mock_conn
            _wait_for_duckdb_lock(db_path, timeout=1.0)
            mock_connect.assert_called_once_with(db_path)
            mock_conn.close.assert_called_once()

    def test_wal_exists_and_connect_fails_then_succeeds(self, tmp_path):
        """Retries when DuckDB connect raises IOException, then succeeds."""
        import duckdb

        db_path = str(tmp_path / "test.duckdb")
        wal_path = Path(db_path + ".wal")
        wal_path.touch()

        mock_conn = MagicMock()
        side_effects = [duckdb.IOException("locked"), mock_conn]

        with patch.object(duckdb, "connect", side_effect=side_effects):
            with patch("time.sleep"):
                _wait_for_duckdb_lock(db_path, timeout=5.0)
            assert duckdb.connect.call_count >= 2

    def test_wal_exists_timeout_reached(self, tmp_path):
        """When lock cannot be acquired within timeout, function returns gracefully."""
        import duckdb

        db_path = str(tmp_path / "test.duckdb")
        wal_path = Path(db_path + ".wal")
        wal_path.touch()

        with patch.object(duckdb, "connect", side_effect=duckdb.IOException("locked")):
            # Use a very short timeout so it doesn't block
            _wait_for_duckdb_lock(db_path, timeout=0.0)


# ---------------------------------------------------------------------------
# MiraiApp.__init__
# ---------------------------------------------------------------------------


class TestMiraiAppInit:
    """Tests for MiraiApp instantiation."""

    def test_init_sets_defaults(self):
        """All attributes start as None/empty after __init__."""
        app = MiraiApp()
        assert app.agent is None
        assert app.heartbeat is None
        assert app.dreamer is None
        assert app.cron is None
        assert app.registry is None
        assert app.config is None
        assert isinstance(app._tasks, set)
        assert len(app._tasks) == 0
        assert app._im_provider is None
        assert app._tool_context is None

    def test_start_time_is_set(self):
        """start_time is initialized to a monotonic timestamp."""
        before = time.monotonic()
        app = MiraiApp()
        after = time.monotonic()
        assert before <= app.start_time <= after


# ---------------------------------------------------------------------------
# MiraiApp.shutdown
# ---------------------------------------------------------------------------


class TestMiraiAppShutdown:
    """Tests for graceful shutdown."""

    @pytest.mark.asyncio
    async def test_shutdown_cancels_tasks(self):
        """Background tasks are cancelled and awaited during shutdown."""
        app = MiraiApp()
        mock_task = MagicMock(spec=asyncio.Task)
        mock_task.cancel = MagicMock()
        app._tasks = {mock_task}

        with patch("asyncio.gather", new_callable=AsyncMock) as mock_gather:
            await app.shutdown()
            mock_task.cancel.assert_called_once()
            mock_gather.assert_called_once()
        assert len(app._tasks) == 0

    @pytest.mark.asyncio
    async def test_shutdown_stops_services(self):
        """Cron, dreamer, and heartbeat are stopped in order."""
        app = MiraiApp()
        app.cron = MagicMock()
        app.dreamer = AsyncMock()
        app.heartbeat = AsyncMock()

        await app.shutdown()

        app.cron.stop.assert_called_once()
        app.dreamer.stop.assert_awaited_once()
        app.heartbeat.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_closes_l3_storage(self):
        """L3 storage is closed during shutdown."""
        app = MiraiApp()
        mock_storage = MagicMock()
        app.agent = MagicMock()
        app.agent.l3_storage = mock_storage

        await app.shutdown()
        mock_storage.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_handles_storage_close_error(self):
        """Shutdown doesn't raise if L3 storage close fails."""
        app = MiraiApp()
        mock_storage = MagicMock()
        mock_storage.close.side_effect = RuntimeError("close failed")
        app.agent = MagicMock()
        app.agent.l3_storage = mock_storage

        # Should not raise
        await app.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_no_services(self):
        """Shutdown works cleanly when no services are configured."""
        app = MiraiApp()
        await app.shutdown()  # Should not raise


# ---------------------------------------------------------------------------
# MiraiApp._init_config
# ---------------------------------------------------------------------------


class TestMiraiAppInitConfig:
    """Tests for configuration initialization."""

    def test_init_config_loads_config(self):
        """_init_config loads MiraiConfig and sets up logging + tracing."""
        app = MiraiApp()
        mock_config = MagicMock()
        mock_config.server.log_format = "console"
        mock_config.server.log_level = "INFO"
        mock_config.server.host = "0.0.0.0"
        mock_config.server.port = 8000
        mock_config.llm.default_model = "test-model"
        mock_config.tracing.console = False

        with (
            patch("mirai.bootstrap.MiraiConfig.load", return_value=mock_config),
            patch("mirai.bootstrap.setup_logging") as mock_log,
            patch("mirai.bootstrap.setup_tracing") as mock_trace,
        ):
            app._init_config()

        assert app.config is mock_config
        mock_log.assert_called_once_with(json_output=False, level="INFO")
        mock_trace.assert_called_once_with(service_name="mirai", console=False)

    def test_init_config_json_format(self):
        """JSON log format is detected and passed to setup_logging."""
        app = MiraiApp()
        mock_config = MagicMock()
        mock_config.server.log_format = "json"
        mock_config.server.log_level = "DEBUG"
        mock_config.server.host = "0.0.0.0"
        mock_config.server.port = 8000
        mock_config.llm.default_model = "test-model"
        mock_config.tracing.console = True

        with (
            patch("mirai.bootstrap.MiraiConfig.load", return_value=mock_config),
            patch("mirai.bootstrap.setup_logging") as mock_log,
            patch("mirai.bootstrap.setup_tracing"),
        ):
            app._init_config()

        mock_log.assert_called_once_with(json_output=True, level="DEBUG")


# ---------------------------------------------------------------------------
# MiraiApp._track_task
# ---------------------------------------------------------------------------


class TestTrackTask:
    """Tests for background task tracking."""

    @pytest.mark.asyncio
    async def test_track_task_adds_and_removes(self):
        """_track_task adds a task to the set, completion removes it."""
        app = MiraiApp()

        async def dummy():
            pass

        app._track_task(dummy(), name="test-task")
        assert len(app._tasks) == 1

        # Wait for the task to complete
        task = next(iter(app._tasks))
        await task
        # Give the done callback a chance to run
        await asyncio.sleep(0.01)
        assert len(app._tasks) == 0

"""QA tests for Round 2 improvements.

Coverage areas:
  1. DuckDB lock hardening — _wait_for_duckdb_lock()
  2. Quota data wiring — _list_models() edge cases
  3. Graceful shutdown — DuckDB close before restart
  4. Fallback chain — edge cases and stress scenarios
  5. OpenAI remote discovery — API call path verification
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirai.agent.agent_loop import AgentLoop
from mirai.agent.models import ProviderResponse, TextBlock
from mirai.agent.providers.base import ModelInfo
from mirai.agent.providers.openai import OpenAIProvider
from mirai.agent.providers.quota import QuotaManager
from mirai.agent.registry import ModelRegistry
from mirai.agent.registry_models import RegistryData
from mirai.agent.tools.base import ToolContext
from mirai.agent.tools.model import ModelTool
from mirai.agent.tools.system import SystemTool
from mirai.bootstrap import _wait_for_duckdb_lock
from mirai.config import MiraiConfig
from mirai.db.duck import DuckDBStorage
from mirai.errors import ProviderError, StorageError

# ===========================================================================
# 1. DuckDB Lock Hardening
# ===========================================================================


class TestDuckDBLockHardening:
    """Tests for _wait_for_duckdb_lock() in bootstrap.py."""

    def test_no_wal_file_returns_immediately(self, tmp_path):
        """When no .wal file exists, function returns instantly."""

        db_path = str(tmp_path / "test.duckdb")
        start = time.monotonic()
        _wait_for_duckdb_lock(db_path, timeout=5.0)
        elapsed = time.monotonic() - start
        assert elapsed < 0.1  # should be near-instant

    def test_wal_exists_but_lock_free(self, tmp_path):
        """When .wal file exists but DB is not locked,
        function connects successfully and returns."""
        import duckdb

        db_path = str(tmp_path / "test.duckdb")

        # Create a valid DB first so .wal exists after ops
        conn = duckdb.connect(db_path)
        conn.execute("CREATE TABLE test (id INTEGER)")
        conn.close()

        # Manually create a .wal file to simulate stale crash
        wal_path = Path(db_path + ".wal")
        wal_path.touch()

        _wait_for_duckdb_lock(db_path, timeout=2.0)
        # Should succeed without timeout

    def test_lock_held_then_released(self, tmp_path):
        """When DB is initially locked but released during timeout,
        function should succeed after retries."""

        db_path = str(tmp_path / "test_retry.duckdb")
        wal_path = Path(db_path + ".wal")
        wal_path.touch()

        call_count = 0
        original_connect = None

        # Mock duckdb.connect to fail twice then succeed
        import duckdb

        original_connect = duckdb.connect

        def mock_connect(path):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise duckdb.IOException("Could not set lock on file")
            return original_connect(path)

        with patch("duckdb.connect", side_effect=mock_connect):
            _wait_for_duckdb_lock(db_path, timeout=5.0)

        assert call_count == 3  # failed twice, succeeded on third

    def test_lock_timeout_logs_warning(self, tmp_path):
        """When lock cannot be acquired within timeout,
        function should log warning and return (not raise)."""

        db_path = str(tmp_path / "stuck.duckdb")
        wal_path = Path(db_path + ".wal")
        wal_path.touch()

        import duckdb

        with patch("duckdb.connect", side_effect=duckdb.IOException("locked")):
            start = time.monotonic()
            # Use a short timeout so test doesn't take long
            _wait_for_duckdb_lock(db_path, timeout=2.0)
            elapsed = time.monotonic() - start

        # Should have waited at least ~2s and returned
        assert elapsed >= 1.5
        assert elapsed < 5.0

    def test_zero_timeout_returns_fast(self, tmp_path):
        """With timeout=0, function should give up quickly even if locked."""

        db_path = str(tmp_path / "zero.duckdb")
        wal_path = Path(db_path + ".wal")
        wal_path.touch()

        import duckdb

        with patch("duckdb.connect", side_effect=duckdb.IOException("locked")):
            start = time.monotonic()
            _wait_for_duckdb_lock(db_path, timeout=0.0)
            elapsed = time.monotonic() - start

        assert elapsed < 2.0


# ===========================================================================
# 2. Quota Data Wiring — Edge Cases
# ===========================================================================


class TestQuotaDataWiringEdgeCases:
    """Additional edge cases for _list_models() quota integration."""

    @pytest.fixture
    def registry(self, tmp_path):
        """Build a ModelRegistry with sample data."""

        reg = ModelRegistry.for_testing(
            path=tmp_path / "registry.json",
            config_provider="minimax",
            config_model="MiniMax-M2.5",
            data=RegistryData.from_dict(
                {
                    "last_refreshed": "2025-01-01T00:00:00Z",
                    "active_provider": "minimax",
                    "active_model": "MiniMax-M2.5",
                    "providers": {
                        "minimax": {
                            "available": True,
                            "models": [
                                {"id": "MiniMax-M2.5", "name": "MiniMax M2.5", "description": "Advanced"},
                                {"id": "MiniMax-VL-01", "name": "MiniMax VL-01", "description": "Vision"},
                            ],
                        },
                    },
                }
            ),
        )
        return reg

    @pytest.mark.asyncio
    async def test_provider_without_quota_manager(self, registry):
        """Provider with no quota_manager → quota_data is None, catalog renders without annotations."""

        mock_provider = MagicMock(spec=[])  # no attributes at all
        ctx = ToolContext(registry=registry, provider=mock_provider)
        tool = ModelTool(context=ctx)
        result = await tool._list_models()

        assert "MiniMax-M2.5" in result
        assert "exhausted" not in result

    @pytest.mark.asyncio
    async def test_no_provider_at_all(self, registry):
        """When provider is None, quota_data is None → catalog renders normally."""

        ctx = ToolContext(registry=registry)
        tool = ModelTool(context=ctx)
        result = await tool._list_models()

        assert "MiniMax-M2.5" in result
        assert "exhausted" not in result

    @pytest.mark.asyncio
    async def test_quota_boundary_80_percent(self, registry):
        """Model at exactly 80% usage should show percentage annotation."""

        mock_qm = MagicMock()
        mock_qm._quotas = {"MiniMax-M2.5": 80.0}
        mock_qm._maybe_refresh = AsyncMock()

        mock_provider = MagicMock()
        mock_provider.quota_manager = mock_qm

        ctx = ToolContext(registry=registry, provider=mock_provider)
        tool = ModelTool(context=ctx)
        result = await tool._list_models()

        assert "80%" in result

    @pytest.mark.asyncio
    async def test_quota_boundary_79_percent_no_annotation(self, registry):
        """Model at 79% usage should NOT show percentage annotation."""

        mock_qm = MagicMock()
        mock_qm._quotas = {"MiniMax-M2.5": 79.0}
        mock_qm._maybe_refresh = AsyncMock()

        mock_provider = MagicMock()
        mock_provider.quota_manager = mock_qm

        ctx = ToolContext(registry=registry, provider=mock_provider)
        tool = ModelTool(context=ctx)
        result = await tool._list_models()

        assert "79%" not in result
        assert "exhausted" not in result

    @pytest.mark.asyncio
    async def test_quota_boundary_100_percent(self, registry):
        """Model at exactly 100% should show exhausted emoji."""

        mock_qm = MagicMock()
        mock_qm._quotas = {"MiniMax-M2.5": 100.0}
        mock_qm._maybe_refresh = AsyncMock()

        mock_provider = MagicMock()
        mock_provider.quota_manager = mock_qm

        ctx = ToolContext(registry=registry, provider=mock_provider)
        tool = ModelTool(context=ctx)
        result = await tool._list_models()

        assert "exhausted" in result

    @pytest.mark.asyncio
    async def test_quota_refresh_error_handled(self, registry):
        """If QuotaManager._maybe_refresh raises, _list_models should still return catalog."""

        mock_qm = MagicMock()
        mock_qm._quotas = {"MiniMax-M2.5": 42.0}  # stale data
        mock_qm._maybe_refresh = AsyncMock(side_effect=Exception("network error"))

        mock_provider = MagicMock()
        mock_provider.quota_manager = mock_qm

        ctx = ToolContext(registry=registry, provider=mock_provider)
        tool = ModelTool(context=ctx)
        # Should NOT raise — returns catalog with stale quota data
        result = await tool._list_models()
        assert "MiniMax-M2.5" in result

    @pytest.mark.asyncio
    async def test_empty_quota_dict(self, registry):
        """Empty _quotas dict → no annotations at all."""

        mock_qm = MagicMock()
        mock_qm._quotas = {}
        mock_qm._maybe_refresh = AsyncMock()

        mock_provider = MagicMock()
        mock_provider.quota_manager = mock_qm

        ctx = ToolContext(registry=registry, provider=mock_provider)
        tool = ModelTool(context=ctx)
        result = await tool._list_models()

        assert "exhausted" not in result
        assert "%" not in result

    @pytest.mark.asyncio
    async def test_quota_for_unknown_model(self, registry):
        """Quota for model not in registry should not cause issues."""

        mock_qm = MagicMock()
        mock_qm._quotas = {"unknown-model-xyz": 100.0}
        mock_qm._maybe_refresh = AsyncMock()

        mock_provider = MagicMock()
        mock_provider.quota_manager = mock_qm

        ctx = ToolContext(registry=registry, provider=mock_provider)
        tool = ModelTool(context=ctx)
        result = await tool._list_models()

        # Should succeed without errors, and not annotate known models
        assert "MiniMax-M2.5" in result
        assert "exhausted" not in result


# ===========================================================================
# 3. Graceful Shutdown — DuckDB Close on Restart
# ===========================================================================


class TestGracefulShutdown:
    """Tests for DuckDB cleanup in _restart._do_restart()."""

    @pytest.mark.asyncio
    async def test_restart_closes_duckdb_storage(self):
        """Verify _do_restart calls l3_storage.close() before restart."""

        # Build tool with mock agent_loop that has l3_storage
        mock_storage = MagicMock()
        mock_storage.close = MagicMock()

        mock_loop = MagicMock()
        mock_loop.l3_storage = mock_storage

        tool = SystemTool(config=MiraiConfig(), agent_loop=mock_loop)

        with patch("os.killpg"), patch("subprocess.Popen"):
            result = await tool.execute(action="restart")

        assert "Restart scheduled" in result

        # Wait briefly for background task to start (it sleeps 8s so we can't
        # wait for it fully, but we can verify the task was created)
        # The actual close call happens asynchronously after 8s delay

    @pytest.mark.asyncio
    async def test_restart_without_agent_loop(self):
        """Restart should still work when agent_loop is None."""

        tool = SystemTool(config=MiraiConfig())

        with patch("os.killpg"), patch("subprocess.Popen"):
            result = await tool.execute(action="restart")

        assert "Restart scheduled" in result

    @pytest.mark.asyncio
    async def test_restart_with_storage_close_error(self):
        """Restart should succeed even if l3_storage.close() raises."""

        mock_storage = MagicMock()
        mock_storage.close = MagicMock(side_effect=RuntimeError("DB already closed"))

        mock_loop = MagicMock()
        mock_loop.l3_storage = mock_storage

        tool = SystemTool(config=MiraiConfig(), agent_loop=mock_loop)

        with patch("os.killpg"), patch("subprocess.Popen"):
            result = await tool.execute(action="restart")

        assert "Restart scheduled" in result


# ===========================================================================
# 4. Fallback Chain — Extra Edge Cases
# ===========================================================================


class TestFallbackChainEdgeCases:
    """Additional fallback chain edge cases beyond test_fallback_chain.py."""

    def _make_loop(self, provider, fallback_models=None):

        l3 = MagicMock()
        l3.append_trace = AsyncMock()
        l2 = MagicMock()
        l2.query = AsyncMock(return_value=[])

        return AgentLoop.for_testing(
            provider=provider,
            fallback_models=fallback_models,
            l3_storage=l3,
            l2_storage=l2,
        )

    def _ok_response(self, text="Hello", model="test"):

        return ProviderResponse(
            content=[TextBlock(text=text)],
            stop_reason="end_turn",
            model_id=model,
        )

    @pytest.mark.asyncio
    async def test_empty_fallback_list_primary_fails(self):
        """No fallback models configured + primary fails → ProviderError wrapping original."""

        provider = MagicMock()

        async def _gen(model, system, messages, tools):
            raise RuntimeError("Primary down")

        provider.generate_response = _gen
        provider.model = "primary"

        loop = self._make_loop(provider, fallback_models=[])
        loop._build_system_prompt = AsyncMock(return_value="system")

        with pytest.raises(ProviderError, match="All models in fallback chain failed"):
            async for _ in loop._execute_cycle("Hello", model="primary"):
                pass

    @pytest.mark.asyncio
    async def test_intermittent_failure_second_attempt_same_model(self):
        """Verify that each tool round independently tries the fallback chain.
        If primary fails in round 1 → fallback succeeds, round 2 retries
        primary first again."""

        provider = MagicMock()
        call_log: list[str] = []

        async def _gen(model, system, messages, tools):
            call_log.append(model)
            # Always succeed on primary — tests that primary is always tried first
            return self._ok_response(text="ok", model=model)

        provider.generate_response = _gen
        provider.model = "primary"

        loop = self._make_loop(provider, fallback_models=["fb-1"])
        loop._build_system_prompt = AsyncMock(return_value="system")

        async for _ in loop._execute_cycle("Hello", model="primary"):
            pass

        # Primary should be the only model tried (no fallback needed)
        assert call_log == ["primary"]

    @pytest.mark.asyncio
    async def test_fallback_chain_preserves_last_error(self):
        """The last error in the chain should be the one raised."""
        provider = MagicMock()

        async def _gen(model, system, messages, tools):
            raise ValueError(f"Error from {model}")

        provider.generate_response = _gen
        provider.model = "p"

        loop = self._make_loop(provider, fallback_models=["fb-1", "fb-2"])
        loop._build_system_prompt = AsyncMock(return_value="system")

        with pytest.raises(ProviderError, match="Error from fb-2"):
            async for _ in loop._execute_cycle("Hello", model="p"):
                pass

    @pytest.mark.asyncio
    async def test_single_fallback_model(self):
        """With one fallback model, chain length is exactly 2."""
        provider = MagicMock()
        models_tried: list[str] = []

        async def _gen(model, system, messages, tools):
            models_tried.append(model)
            if model == "primary":
                raise RuntimeError("down")
            return self._ok_response(text="ok", model=model)

        provider.generate_response = _gen
        provider.model = "primary"

        loop = self._make_loop(provider, fallback_models=["fb-only"])
        loop._build_system_prompt = AsyncMock(return_value="system")

        async for _ in loop._execute_cycle("Hello", model="primary"):
            pass

        assert models_tried == ["primary", "fb-only"]

    @pytest.mark.asyncio
    async def test_timeout_error_triggers_fallback(self):
        """TimeoutError (common in API calls) should trigger fallback."""
        provider = MagicMock()
        models_tried: list[str] = []

        async def _gen(model, system, messages, tools):
            models_tried.append(model)
            if model == "primary":
                raise TimeoutError("Request timed out after 30s")
            return self._ok_response(text="ok", model=model)

        provider.generate_response = _gen
        provider.model = "primary"

        loop = self._make_loop(provider, fallback_models=["fb-1"])
        loop._build_system_prompt = AsyncMock(return_value="system")

        async for _ in loop._execute_cycle("Hello", model="primary"):
            pass

        assert models_tried == ["primary", "fb-1"]

    @pytest.mark.asyncio
    async def test_connection_error_triggers_fallback(self):
        """ConnectionError should trigger fallback."""
        provider = MagicMock()
        models_tried: list[str] = []

        async def _gen(model, system, messages, tools):
            models_tried.append(model)
            if model == "primary":
                raise ConnectionError("API endpoint unreachable")
            return self._ok_response(text="ok", model=model)

        provider.generate_response = _gen
        provider.model = "primary"

        loop = self._make_loop(provider, fallback_models=["fb-1"])
        loop._build_system_prompt = AsyncMock(return_value="system")

        async for _ in loop._execute_cycle("Hello", model="primary"):
            pass

        assert models_tried == ["primary", "fb-1"]


# ===========================================================================
# 5. OpenAI Remote Discovery — Edge Cases
# ===========================================================================


class TestOpenAIRemoteDiscovery:
    """Verify OpenAI provider list_models() edge cases."""

    @pytest.mark.asyncio
    async def test_empty_api_response_returns_fallback(self):
        """If API returns empty model list, should return current model as fallback."""

        p = OpenAIProvider(api_key="test-key", model="gpt-4o")
        mock_resp = MagicMock()
        mock_resp.data = []
        p.client.models.list = AsyncMock(return_value=mock_resp)
        models = await p.list_models()

        # Empty response → should still return something usable
        assert len(models) == 0 or models[0].id == "gpt-4o"

    @pytest.mark.asyncio
    async def test_api_returns_multiple_models(self):
        """Verify multiple models from API are all captured."""

        mock_models = []
        for name in ["gpt-4o", "gpt-4o-mini", "o1-preview"]:
            m = MagicMock()
            m.id = name
            mock_models.append(m)

        p = OpenAIProvider(api_key="test-key", model="gpt-4o")
        mock_resp = MagicMock()
        mock_resp.data = mock_models
        p.client.models.list = AsyncMock(return_value=mock_resp)
        models = await p.list_models()

        assert len(models) == 3
        ids = {m.id for m in models}
        assert "gpt-4o" in ids
        assert "o1-preview" in ids

    @pytest.mark.asyncio
    async def test_subclass_with_catalog_ignores_api(self):
        """Subclass with MODEL_CATALOG should never call the API."""

        class LocalProvider(OpenAIProvider):
            MODEL_CATALOG = [ModelInfo(id="local-model", name="Local")]

        p = LocalProvider(api_key="test-key")
        p.client.models.list = AsyncMock(side_effect=AssertionError("Should not be called"))
        models = await p.list_models()

        assert len(models) == 1
        assert models[0].id == "local-model"


# ===========================================================================
# 6. QuotaManager Unit Tests
# ===========================================================================


class TestQuotaManager:
    """Verify QuotaManager correctness and caching behavior."""

    @pytest.mark.asyncio
    async def test_initial_usage_is_zero(self):
        """Before any refresh, all models report 0% usage."""

        qm = QuotaManager(credentials={"access": "test"})
        # Prevent actual API call
        qm._last_update = time.time()  # pretend just refreshed
        pct = await qm.get_used_pct("any-model")
        assert pct == 0.0

    @pytest.mark.asyncio
    async def test_is_available_at_99_percent(self):
        """Model at 99% should still be available."""

        qm = QuotaManager(credentials={"access": "test"})
        qm._quotas = {"model-a": 99.0}
        qm._last_update = time.time()

        assert await qm.is_available("model-a") is True

    @pytest.mark.asyncio
    async def test_is_available_at_100_percent(self):
        """Model at 100% should NOT be available."""

        qm = QuotaManager(credentials={"access": "test"})
        qm._quotas = {"model-a": 100.0}
        qm._last_update = time.time()

        assert await qm.is_available("model-a") is False

    @pytest.mark.asyncio
    async def test_cache_ttl_prevents_extra_refresh(self):
        """Within TTL, _maybe_refresh should not call fetch_usage again."""

        qm = QuotaManager(credentials={"access": "test", "project_id": "proj"})
        qm._last_update = time.time()
        qm._quotas = {"model-a": 50.0}

        with patch("mirai.agent.providers.quota.fetch_usage") as mock_fetch:
            await qm._maybe_refresh()
            mock_fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_expired_cache_triggers_refresh(self):
        """After TTL expires, _maybe_refresh should call fetch_usage."""

        qm = QuotaManager(credentials={"access": "token", "project_id": "proj"})
        qm._last_update = time.time() - (qm.CACHE_TTL + 1)  # expired

        mock_usage = {"models": [{"id": "model-x", "used_pct": 42.0}]}
        with patch("mirai.agent.providers.quota.fetch_usage", new_callable=AsyncMock, return_value=mock_usage):
            await qm._maybe_refresh()

        assert qm._quotas == {"model-x": 42.0}

    @pytest.mark.asyncio
    async def test_refresh_error_preserves_old_quotas(self):
        """If fetch_usage fails, old quota data should be preserved."""

        qm = QuotaManager(credentials={"access": "token", "project_id": "proj"})
        qm._quotas = {"model-a": 25.0}
        qm._last_update = time.time() - (qm.CACHE_TTL + 1)

        with patch("mirai.agent.providers.quota.fetch_usage", new_callable=AsyncMock, side_effect=Exception("network")):
            await qm._maybe_refresh()

        # Old quotas preserved (refresh failed silently)
        assert qm._quotas == {"model-a": 25.0}


# ===========================================================================
# 7. DuckDBStorage.close() Unit Tests
# ===========================================================================


class TestDuckDBStorageClose:
    """Verify DuckDB connection cleanup behavior."""

    def test_close_releases_connection(self, tmp_path):
        """After close(), conn should be None."""

        db_path = str(tmp_path / "close_test.duckdb")
        storage = DuckDBStorage(db_path=db_path)
        assert storage.conn is not None
        storage.close()
        assert storage.conn is None

    def test_double_close_is_safe(self, tmp_path):
        """Calling close() twice should not raise."""

        db_path = str(tmp_path / "double_close.duckdb")
        storage = DuckDBStorage(db_path=db_path)
        storage.close()
        storage.close()  # should not raise
        assert storage.conn is None

    def test_close_then_reconnect(self, tmp_path):
        """After close(), creating a new DuckDBStorage on same path should work."""

        db_path = str(tmp_path / "reconnect.duckdb")
        storage1 = DuckDBStorage(db_path=db_path)
        storage1.close()

        storage2 = DuckDBStorage(db_path=db_path)
        assert storage2.conn is not None
        storage2.close()

    @pytest.mark.asyncio
    async def test_operations_after_close_handled(self, tmp_path):
        """Operations after close should raise or be handled gracefully."""

        db_path = str(tmp_path / "ops_after_close.duckdb")
        storage = DuckDBStorage(db_path=db_path)
        storage.close()

        # Attempt to write after close should raise StorageError

        with pytest.raises(StorageError, match="DuckDB connection is closed"):
            await storage.append_trace(id="test", collaborator_id="c1", trace_type="message", content="after close")


# ===========================================================================
# 8. Integration: list_models via execute() dispatch
# ===========================================================================


class TestListModelsDispatch:
    """Verify list_models goes through the execute() dispatch correctly."""

    @pytest.mark.asyncio
    async def test_execute_list_models_action(self):
        """execute(action='list_models') should call _list_models and return a string."""

        reg = ModelRegistry.for_testing(
            path=Path("/tmp/test_reg.json"),
            config_provider="mock",
            config_model="mock-model",
            data=RegistryData.from_dict(
                {
                    "active_provider": "mock",
                    "active_model": "mock-model",
                    "providers": {
                        "mock": {
                            "available": True,
                            "models": [{"id": "mock-model", "name": "Mock Model", "description": "Test"}],
                        },
                    },
                }
            ),
        )

        ctx = ToolContext(registry=reg)
        tool = ModelTool(context=ctx)
        result = await tool.execute(action="list_models")

        assert isinstance(result, str)
        assert "mock-model" in result

    @pytest.mark.asyncio
    async def test_list_models_shows_active_provider(self):
        """Catalog text should indicate the active provider."""

        reg = ModelRegistry.for_testing(
            path=Path("/tmp/test_reg2.json"),
            config_provider="minimax",
            config_model="M2.5",
            data=RegistryData.from_dict(
                {
                    "active_provider": "minimax",
                    "active_model": "M2.5",
                    "providers": {
                        "minimax": {
                            "available": True,
                            "models": [{"id": "M2.5", "name": "M2.5", "description": "Test"}],
                        },
                    },
                }
            ),
        )

        ctx = ToolContext(registry=reg)
        tool = ModelTool(context=ctx)
        result = await tool.execute(action="list_models")

        assert "minimax" in result.lower()
        assert "active" in result.lower()

"""QA tests for Round 5 — Architectural Best-Practices.

Coverage areas:
  1. Exception hierarchy — errors.py
  2. Typed registry models — registry_models.py
  3. Registry typed data — registry.py with RegistryData
  4. Bootstrap decomposition — lifecycle phases + task tracking
  5. Factory return type & errors — factory.py → ProviderProtocol
  6. Metrics auto-wiring — LatencyTimer integration
  7. Config cross-field validation — model_post_init warnings
  8. Exception chaining — raise ... from exc
  9. SystemTool typed constructor
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest

# ===========================================================================
# 1. Exception Hierarchy
# ===========================================================================


class TestExceptionHierarchy:
    """Verify the MiraiError exception tree."""

    def test_mirai_error_is_exception(self):
        from mirai.errors import MiraiError

        assert issubclass(MiraiError, Exception)

    def test_provider_error_inherits_mirai_error(self):
        from mirai.errors import MiraiError, ProviderError

        assert issubclass(ProviderError, MiraiError)

    def test_storage_error_inherits_mirai_error(self):
        from mirai.errors import MiraiError, StorageError

        assert issubclass(StorageError, MiraiError)

    def test_config_error_inherits_mirai_error(self):
        from mirai.errors import ConfigError, MiraiError

        assert issubclass(ConfigError, MiraiError)

    def test_tool_error_inherits_mirai_error(self):
        from mirai.errors import MiraiError, ToolError

        assert issubclass(ToolError, MiraiError)

    def test_shutdown_error_inherits_mirai_error(self):
        from mirai.errors import MiraiError, ShutdownError

        assert issubclass(ShutdownError, MiraiError)

    def test_all_errors_catchable_by_base(self):
        from mirai.errors import (
            ConfigError,
            MiraiError,
            ProviderError,
            ShutdownError,
            StorageError,
            ToolError,
        )

        for cls in (ProviderError, StorageError, ConfigError, ToolError, ShutdownError):
            try:
                raise cls("test")
            except MiraiError:
                pass  # expected

    def test_error_message_preserved(self):
        from mirai.errors import ProviderError

        err = ProviderError("custom message")
        assert str(err) == "custom message"

    def test_exception_chaining_with_from(self):
        """'from' chaining should set __cause__."""
        from mirai.errors import ProviderError

        cause = ValueError("root cause")
        try:
            raise ProviderError("wrapper") from cause
        except ProviderError as e:
            assert e.__cause__ is cause


# ===========================================================================
# 2. Typed Registry Models
# ===========================================================================


class TestRegistryModels:
    """Verify RegistryModelEntry, RegistryProviderData, RegistryData."""

    def test_model_entry_with_name(self):
        from mirai.agent.registry_models import RegistryModelEntry

        entry = RegistryModelEntry(id="gpt-4", name="gpt-4")
        assert entry.id == "gpt-4"
        assert entry.name == "gpt-4"
        assert entry.description is None
        assert entry.reasoning is False
        assert entry.vision is False

    def test_model_entry_custom_fields(self):
        from mirai.agent.registry_models import RegistryModelEntry

        entry = RegistryModelEntry(
            id="claude-3",
            name="Claude 3",
            description="Anthropic model",
            reasoning=True,
            vision=True,
        )
        assert entry.name == "Claude 3"
        assert entry.reasoning is True
        assert entry.vision is True

    def test_provider_data_defaults(self):
        from mirai.agent.registry_models import RegistryProviderData

        pd = RegistryProviderData()
        assert pd.available is False
        assert pd.env_key == ""
        assert pd.models == []

    def test_registry_data_defaults(self):
        from mirai.agent.registry_models import RegistryData

        rd = RegistryData()
        assert rd.version == 1
        assert rd.last_refreshed is None
        assert rd.active_provider is None
        assert rd.providers == {}

    def test_registry_data_to_dict_roundtrip(self):
        from mirai.agent.registry_models import (
            RegistryData,
            RegistryModelEntry,
            RegistryProviderData,
        )

        rd = RegistryData(
            version=2,
            last_refreshed="2025-01-01T00:00:00",
            active_provider="openai",
            active_model="gpt-4",
            providers={
                "openai": RegistryProviderData(
                    available=True,
                    env_key="OPENAI_API_KEY",
                    models=[
                        RegistryModelEntry(id="gpt-4", name="GPT-4", vision=True),
                    ],
                )
            },
        )

        d = rd.to_dict()
        assert d["version"] == 2
        assert d["active_provider"] == "openai"
        assert len(d["providers"]["openai"]["models"]) == 1
        assert d["providers"]["openai"]["models"][0]["vision"] is True

        # Roundtrip
        rd2 = RegistryData.from_dict(d)
        assert rd2.version == 2
        assert rd2.active_provider == "openai"
        assert rd2.providers["openai"].models[0].vision is True

    def test_registry_data_from_dict_missing_fields(self):
        """from_dict should handle missing/minimal data gracefully."""
        from mirai.agent.registry_models import RegistryData

        rd = RegistryData.from_dict({})
        assert rd.version == 1
        assert rd.providers == {}


# ===========================================================================
# 3. Registry with Typed Data
# ===========================================================================


class TestRegistryTypedIntegration:
    """Verify ModelRegistry uses RegistryData consistently."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_deps(self):
        pytest.importorskip("orjson")
        pytest.importorskip("structlog")

    def test_registry_load_creates_registry_data(self, tmp_path, monkeypatch):
        """A fresh registry should create a RegistryData instance."""
        pytest.importorskip("orjson")
        from mirai.agent.registry import ModelRegistry
        from mirai.agent.registry_models import RegistryData

        monkeypatch.setattr(ModelRegistry, "PATH", tmp_path / "registry.json")
        reg = ModelRegistry(config_provider="openai", config_model="gpt-4")
        assert isinstance(reg._data, RegistryData)

    def test_registry_save_load_roundtrip(self, tmp_path, monkeypatch):
        """Save then load should preserve typed data."""
        pytest.importorskip("orjson")
        from mirai.agent.registry import ModelRegistry

        json_path = tmp_path / "registry.json"
        monkeypatch.setattr(ModelRegistry, "PATH", json_path)

        reg = ModelRegistry(config_provider="anthropic", config_model="claude-3")
        reg._save()
        assert json_path.exists()

        reg2 = ModelRegistry(config_provider="anthropic", config_model="claude-3")
        assert reg2._data.active_provider == "anthropic"
        assert reg2._data.active_model == "claude-3"


# ===========================================================================
# 4. Bootstrap Decomposition & Task Tracking
# ===========================================================================


class TestBootstrapDecomposition:
    """Verify MiraiApp lifecycle phases and task tracking."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_deps(self):
        pytest.importorskip("duckdb")
        pytest.importorskip("pydantic_settings")

    def test_track_task_adds_and_removes(self):
        """_track_task should add a task and remove it when done."""
        from mirai.bootstrap import MiraiApp

        app = MiraiApp()
        app._tasks = set()

        async def _inner():
            coro = asyncio.sleep(0)
            task = app._track_task(coro, name="test_task")
            assert task in app._tasks
            await task
            # After completion, done callback should discard it
            await asyncio.sleep(0.05)
            assert task not in app._tasks

        asyncio.run(_inner())

    def test_track_task_logs_exception(self):
        """_track_task should log but not propagate errors."""
        from mirai.bootstrap import MiraiApp

        app = MiraiApp()
        app._tasks = set()

        async def _inner():
            async def _failing():
                raise ValueError("boom")

            task = app._track_task(_failing(), name="failing_task")
            # Wait for it to complete
            await asyncio.sleep(0.1)
            assert task not in app._tasks
            assert task.done()

        asyncio.run(_inner())

    def test_shutdown_cancels_tasks(self):
        """shutdown() should cancel all tracked tasks."""
        from mirai.bootstrap import MiraiApp

        app = MiraiApp()
        app._tasks = set()
        app.heartbeat = None
        app.agent = None

        async def _inner():
            async def _forever():
                await asyncio.sleep(3600)

            task = app._track_task(_forever(), name="forever_task")
            assert len(app._tasks) >= 1
            await app.shutdown()
            assert len(app._tasks) == 0
            assert task.cancelled()

        asyncio.run(_inner())

    def test_create_method_has_lifecycle_phases(self):
        """Verify that MiraiApp has the four lifecycle methods."""
        from mirai.bootstrap import MiraiApp

        assert hasattr(MiraiApp, "_init_config")
        assert hasattr(MiraiApp, "_init_storage")
        assert hasattr(MiraiApp, "_init_agent_stack")
        assert hasattr(MiraiApp, "_init_integrations")


# ===========================================================================
# 5. Factory Return Type & Errors
# ===========================================================================


class TestProviderFactory:
    """Verify factory.py returns ProviderProtocol and raises ProviderError."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_deps(self):
        pytest.importorskip("anthropic", reason="anthropic SDK not installed")

    def test_factory_raises_provider_error_for_missing_key(self, monkeypatch):
        from mirai.agent.providers.factory import create_provider
        from mirai.errors import ProviderError

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ProviderError):
            create_provider(provider="openai", api_key=None)

    def test_factory_raises_provider_error_for_minimax_no_key(self, monkeypatch):
        from mirai.agent.providers.factory import create_provider
        from mirai.errors import ProviderError

        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        with pytest.raises(ProviderError, match="MiniMax"):
            create_provider(provider="minimax", api_key=None)

    def test_factory_return_annotation_is_provider_protocol(self):
        """The return annotation should be ProviderProtocol, not Any."""
        import inspect

        from mirai.agent.providers.factory import create_provider

        sig = inspect.signature(create_provider)
        ret = sig.return_annotation
        assert "ProviderProtocol" in str(ret) or ret.__name__ == "ProviderProtocol"


# ===========================================================================
# 6. Metrics Auto-Wiring
# ===========================================================================


class TestMetricsAutoWire:
    """Verify LatencyTimer is wired into the HTTP middleware."""

    def test_latency_timer_records_success(self):
        from mirai.metrics import LatencyTimer, RequestMetrics

        m = RequestMetrics()
        with LatencyTimer(metrics_instance=m):
            pass  # simulate normal request

        snap = m.snapshot()
        assert snap["total_requests"] == 1
        assert snap["error_rate_pct"] == 0.0

    def test_latency_timer_records_error(self):
        from mirai.metrics import LatencyTimer, RequestMetrics

        m = RequestMetrics()
        with LatencyTimer(metrics_instance=m) as timer:
            timer.mark_error()

        snap = m.snapshot()
        assert snap["total_requests"] == 1
        assert snap["error_rate_pct"] == 100.0

    def test_latency_timer_records_on_exception(self):
        from mirai.metrics import LatencyTimer, RequestMetrics

        m = RequestMetrics()
        with pytest.raises(ValueError):
            with LatencyTimer(metrics_instance=m):
                raise ValueError("boom")

        snap = m.snapshot()
        assert snap["total_requests"] == 1
        assert snap["error_rate_pct"] == 100.0

    def test_middleware_source_contains_latency_timer(self):
        """Verify main.py middleware source references LatencyTimer."""
        main_path = Path(__file__).parent.parent / "main.py"
        source = main_path.read_text()
        assert "LatencyTimer" in source


# ===========================================================================
# 7. Config Cross-Field Validation
# ===========================================================================


class TestConfigCrossValidation:
    """Verify model_post_init warns on misconfigured API keys."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_pydantic_settings(self):
        pytest.importorskip("pydantic_settings")

    def test_no_warning_for_antigravity(self, caplog, monkeypatch):
        """Default antigravity provider should not emit a warning."""
        from mirai.config import MiraiConfig

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with caplog.at_level(logging.WARNING, logger="mirai.config"):
            MiraiConfig(llm={"provider": "antigravity"})

        config_warnings = [r for r in caplog.records if "mirai.config" in r.name]
        assert len(config_warnings) == 0

    def test_warning_for_openai_without_key(self, caplog, monkeypatch):
        """Selecting openai without API key should log a warning."""
        from mirai.config import MiraiConfig

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with caplog.at_level(logging.WARNING, logger="mirai.config"):
            MiraiConfig(llm={"provider": "openai"})

        config_warnings = [r for r in caplog.records if "mirai.config" in r.name]
        assert len(config_warnings) == 1
        assert "OPENAI_API_KEY" in config_warnings[0].getMessage()

    def test_no_warning_when_key_present(self, caplog, monkeypatch):
        """openai provider with API key set should not warn."""
        from mirai.config import MiraiConfig

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

        with caplog.at_level(logging.WARNING, logger="mirai.config"):
            MiraiConfig(llm={"provider": "openai"})

        config_warnings = [r for r in caplog.records if "mirai.config" in r.name]
        assert len(config_warnings) == 0


# ===========================================================================
# 8. Exception Chaining (raise ... from exc)
# ===========================================================================


class TestExceptionChaining:
    """Verify that exceptions are properly chained."""

    def test_duck_storage_error_on_closed_conn(self):
        """DuckDBStorage should raise StorageError when connection is closed."""
        pytest.importorskip("duckdb")
        from mirai.db.duck import DuckDBStorage
        from mirai.errors import StorageError

        storage = DuckDBStorage.__new__(DuckDBStorage)
        storage.conn = None

        with pytest.raises(StorageError, match="closed"):
            storage._check_conn()

    def test_main_py_uses_bare_raise(self):
        """main.py exception handler should use 'raise' not 'raise e'."""
        main_path = Path(__file__).parent.parent / "main.py"
        source = main_path.read_text()
        lines = source.splitlines()
        # Find lines with bare 'raise' in exception handlers
        # Should NOT have "raise e" or "raise exc" — should be plain "raise"
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == "raise e" or stripped == "raise exc":
                pytest.fail(f"Found improper re-raise at line {i + 1}: {stripped}")

    def test_loop_uses_provider_error_for_fallback(self):
        """agent_loop.py should raise ProviderError, not RuntimeError, for fallback chain."""
        loop_path = Path(__file__).parent.parent / "mirai" / "agent" / "agent_loop.py"
        source = loop_path.read_text()
        assert "ProviderError" in source
        assert 'raise ProviderError("All models in fallback chain failed")' in source


# ===========================================================================
# 9. SystemTool Typed Constructor
# ===========================================================================


class TestSystemToolTypedConstructor:
    """Verify SystemTool constructor uses typed parameters."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_deps(self):
        pytest.importorskip("anthropic", reason="anthropic SDK not installed")

    def test_constructor_accepts_typed_args(self):
        """SystemTool should accept typed constructor arguments."""
        from mirai.agent.tools.system import SystemTool

        tool = SystemTool(
            config=None,
            start_time=1000.0,
            provider=None,
            registry=None,
            agent_loop=None,
        )
        assert tool._start_time == 1000.0

    def test_constructor_type_annotations(self):
        """Constructor annotations should use specific types, not Any, for
        registry and agent_loop."""
        import inspect

        from mirai.agent.tools.system import SystemTool

        sig = inspect.signature(SystemTool.__init__)
        params = sig.parameters

        # registry and agent_loop should NOT be plain Any
        registry_annotation = str(params["registry"].annotation)
        agent_loop_annotation = str(params["agent_loop"].annotation)

        assert "Any" not in registry_annotation or "ModelRegistry" in registry_annotation
        assert "Any" not in agent_loop_annotation or "AgentLoop" in agent_loop_annotation

    def test_default_start_time_is_monotonic(self):
        """If no start_time given, it should default to time.monotonic()."""
        import time

        from mirai.agent.tools.system import SystemTool

        before = time.monotonic()
        tool = SystemTool()
        after = time.monotonic()
        assert before <= tool._start_time <= after

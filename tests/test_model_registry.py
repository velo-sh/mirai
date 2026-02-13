"""Tests for ModelRegistry — T1 through T6 as specified in the implementation plan."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirai.agent.providers.base import ModelInfo


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_registry_path(tmp_path: Path) -> Path:
    """Return a temp path for model_registry.json."""
    return tmp_path / "model_registry.json"


@pytest.fixture
def sample_registry_data() -> dict[str, Any]:
    """A valid registry JSON structure."""
    return {
        "version": 1,
        "last_refreshed": "2026-02-13T23:00:00+00:00",
        "active_provider": "minimax",
        "active_model": "MiniMax-M2.5",
        "providers": {
            "minimax": {
                "available": True,
                "env_key": "MINIMAX_API_KEY",
                "models": [
                    {"id": "MiniMax-M2.5", "name": "MiniMax M2.5", "description": "Advanced reasoning", "reasoning": True, "vision": False},
                    {"id": "MiniMax-VL-01", "name": "MiniMax VL 01", "description": "Vision-language model", "reasoning": False, "vision": True},
                ],
            },
            "anthropic": {
                "available": False,
                "env_key": "ANTHROPIC_API_KEY",
                "models": [],
            },
        },
    }


def _make_registry(path: Path, **kwargs):
    """Create a ModelRegistry with a custom path."""
    from mirai.agent.registry import ModelRegistry
    registry = ModelRegistry.__new__(ModelRegistry)
    registry.PATH = path  # Override class var on instance
    registry._config_provider = kwargs.get("config_provider")
    registry._config_model = kwargs.get("config_model")
    # Manually assign PATH before _load so _load reads from correct path
    ModelRegistry.PATH = path
    registry._data = registry._load()
    ModelRegistry.PATH = Path.home() / ".mirai" / "model_registry.json"  # restore
    return registry


def _make_registry_from_data(path: Path, data: dict, **kwargs):
    """Create a ModelRegistry pre-loaded with data."""
    from mirai.agent.registry import ModelRegistry
    path.write_text(json.dumps(data), encoding="utf-8")
    registry = ModelRegistry.__new__(ModelRegistry)
    registry._config_provider = kwargs.get("config_provider")
    registry._config_model = kwargs.get("config_model")
    registry.PATH = path
    registry._data = data
    return registry


# ===========================================================================
# T1: Registry Lifecycle
# ===========================================================================

class TestRegistryLifecycle:
    """T1: Load, save, and error handling."""

    def test_t1_1_first_run_no_file(self, tmp_registry_path: Path):
        """T1.1: First run — no JSON file exists."""
        registry = _make_registry(
            tmp_registry_path,
            config_provider="minimax",
            config_model="MiniMax-M2.5",
        )
        assert registry.active_provider == "minimax"
        assert registry.active_model == "MiniMax-M2.5"
        assert registry._data["providers"] == {}

    def test_t1_2_normal_load(self, tmp_registry_path: Path, sample_registry_data: dict):
        """T1.2: Normal load — valid JSON exists."""
        registry = _make_registry_from_data(tmp_registry_path, sample_registry_data)
        assert registry.active_provider == "minimax"
        assert registry.active_model == "MiniMax-M2.5"
        providers = registry._data["providers"]
        assert "minimax" in providers
        assert providers["minimax"]["available"] is True
        assert len(providers["minimax"]["models"]) == 2

    def test_t1_3_corrupted_json(self, tmp_registry_path: Path):
        """T1.3: Corrupted JSON → falls back to empty state."""
        tmp_registry_path.write_text("{invalid json!!", encoding="utf-8")
        registry = _make_registry(
            tmp_registry_path,
            config_provider="minimax",
            config_model="MiniMax-M2.5",
        )
        # Should not crash, should fall back
        assert registry.active_provider == "minimax"
        assert registry._data["providers"] == {}

    @pytest.mark.asyncio
    async def test_t1_4_save_permission_error(self, tmp_registry_path: Path, sample_registry_data: dict):
        """T1.4: Permission denied on save → graceful failure."""
        registry = _make_registry_from_data(tmp_registry_path, sample_registry_data)
        # Point to a non-writable path
        registry.PATH = Path("/root/no_access/registry.json")
        # Should not raise
        await registry._save()
        # In-memory state should still be valid
        assert registry.active_provider == "minimax"


# ===========================================================================
# T2: Config Priority Layering
# ===========================================================================

class TestConfigPriorityLayering:
    """T2: registry (runtime) > config.toml (default) > code default."""

    def test_t2_1_no_registry_active_uses_config(self, tmp_registry_path: Path):
        """T2.1: Registry has no active_model, config.toml has one."""
        data = {"version": 1, "active_provider": None, "active_model": None, "providers": {}}
        registry = _make_registry_from_data(
            tmp_registry_path, data,
            config_provider="anthropic",
            config_model="claude-sonnet-4",
        )
        assert registry.active_provider == "anthropic"
        assert registry.active_model == "claude-sonnet-4"

    def test_t2_2_registry_overrides_config(self, tmp_registry_path: Path):
        """T2.2: Registry has active_model, config.toml has different → registry wins."""
        data = {
            "version": 1,
            "active_provider": "minimax",
            "active_model": "MiniMax-M2.5",
            "providers": {},
        }
        registry = _make_registry_from_data(
            tmp_registry_path, data,
            config_provider="anthropic",
            config_model="claude-sonnet-4",
        )
        assert registry.active_provider == "minimax"
        assert registry.active_model == "MiniMax-M2.5"

    def test_t2_3_neither_has_active(self, tmp_registry_path: Path):
        """T2.3: Neither has active_model → falls back to 'unknown'."""
        data = {"version": 1, "active_provider": None, "active_model": None, "providers": {}}
        registry = _make_registry_from_data(tmp_registry_path, data)
        assert registry.active_provider == "unknown"
        assert registry.active_model == "unknown"

    @pytest.mark.asyncio
    async def test_t2_4_agent_updates_active(self, tmp_registry_path: Path, sample_registry_data: dict):
        """T2.4: Agent updates active_model → written to registry."""
        registry = _make_registry_from_data(tmp_registry_path, sample_registry_data)
        await registry.set_active("anthropic", "claude-opus-4")
        assert registry.active_provider == "anthropic"
        assert registry.active_model == "claude-opus-4"
        # Check it was saved to disk
        saved = json.loads(registry.PATH.read_text())
        assert saved["active_provider"] == "anthropic"
        assert saved["active_model"] == "claude-opus-4"

    @pytest.mark.asyncio
    async def test_t2_5_survives_restart(self, tmp_registry_path: Path, sample_registry_data: dict):
        """T2.5: Simulated restart — registry value persists."""
        registry = _make_registry_from_data(tmp_registry_path, sample_registry_data)
        await registry.set_active("anthropic", "claude-opus-4")

        # "Restart" — create new registry from same file
        registry2 = _make_registry_from_data(
            tmp_registry_path,
            json.loads(tmp_registry_path.read_text()),
            config_provider="minimax",
            config_model="MiniMax-M2.5",
        )
        # Runtime override should still win
        assert registry2.active_provider == "anthropic"
        assert registry2.active_model == "claude-opus-4"


# ===========================================================================
# T3: Refresh (Remote Discovery)
# ===========================================================================

class TestRefresh:
    """T3: Refresh via provider.list_models()."""

    @pytest.mark.asyncio
    async def test_t3_1_valid_api_key(self, tmp_registry_path: Path):
        """T3.1: Provider with valid API key → list_models() called."""
        data = {"version": 1, "active_provider": "minimax", "active_model": "MiniMax-M2.5", "providers": {}}
        registry = _make_registry_from_data(tmp_registry_path, data)

        mock_models = [
            ModelInfo(id="MiniMax-M2.5", name="MiniMax M2.5", description="Advanced reasoning", reasoning=True),
        ]
        mock_provider = MagicMock()
        mock_provider.list_models = AsyncMock(return_value=mock_models)

        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key", "ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": ""}):
            with patch("mirai.agent.registry._import_provider_class") as mock_import:
                mock_import.return_value = MagicMock(return_value=mock_provider)
                await registry.refresh()

        assert registry._data["providers"]["minimax"]["available"] is True
        assert len(registry._data["providers"]["minimax"]["models"]) == 1
        assert registry._data["providers"]["minimax"]["models"][0]["id"] == "MiniMax-M2.5"

    @pytest.mark.asyncio
    async def test_t3_2_no_api_key(self, tmp_registry_path: Path):
        """T3.2: Provider with no API key → marked unavailable."""
        data = {"version": 1, "active_provider": "minimax", "active_model": "MiniMax-M2.5", "providers": {}}
        registry = _make_registry_from_data(tmp_registry_path, data)

        with patch.dict(os.environ, {"MINIMAX_API_KEY": "", "ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": ""}, clear=False):
            # Remove keys if they exist
            for key in ["MINIMAX_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"]:
                os.environ.pop(key, None)
            await registry.refresh()

        for pdata in registry._data["providers"].values():
            assert pdata["available"] is False

    @pytest.mark.asyncio
    async def test_t3_3_api_error_keeps_last_state(self, tmp_registry_path: Path, sample_registry_data: dict):
        """T3.3: Provider API error → keep last known models."""
        registry = _make_registry_from_data(tmp_registry_path, sample_registry_data)
        original_models = sample_registry_data["providers"]["minimax"]["models"]

        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key", "ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": ""}):
            with patch("mirai.agent.registry._import_provider_class") as mock_import:
                mock_cls = MagicMock()
                mock_cls.return_value.list_models = AsyncMock(side_effect=Exception("API timeout"))
                mock_import.return_value = mock_cls
                await registry.refresh()

        # Should have kept original models
        minimax = registry._data["providers"].get("minimax", {})
        assert len(minimax.get("models", [])) == len(original_models)

    @pytest.mark.asyncio
    async def test_t3_6_partial_failure(self, tmp_registry_path: Path):
        """T3.6: Multiple providers, one fails → others still updated."""
        data = {"version": 1, "active_provider": "minimax", "active_model": "MiniMax-M2.5", "providers": {}}
        registry = _make_registry_from_data(tmp_registry_path, data)

        mock_models = [ModelInfo(id="claude-sonnet-4", name="Claude Sonnet 4")]

        def side_effect(path):
            if "minimax" in path:
                cls = MagicMock()
                cls.return_value.list_models = AsyncMock(side_effect=Exception("fail"))
                return cls
            else:
                cls = MagicMock()
                cls.return_value.list_models = AsyncMock(return_value=mock_models)
                return cls

        with patch.dict(os.environ, {"MINIMAX_API_KEY": "key1", "ANTHROPIC_API_KEY": "key2", "OPENAI_API_KEY": ""}):
            with patch("mirai.agent.registry._import_provider_class", side_effect=side_effect):
                await registry.refresh()

        # Anthropic should succeed
        anthropic = registry._data["providers"].get("anthropic", {})
        assert anthropic.get("available") is True
        assert len(anthropic.get("models", [])) == 1


# ===========================================================================
# T4: Concurrency & Performance
# ===========================================================================

class TestConcurrency:
    """T4: Read consistency and performance."""

    def test_t4_3_get_catalog_text_performance(self, tmp_registry_path: Path, sample_registry_data: dict):
        """T4.3: get_catalog_text() should be < 1ms (pure in-memory)."""
        import time
        registry = _make_registry_from_data(tmp_registry_path, sample_registry_data)

        start = time.perf_counter()
        for _ in range(1000):
            registry.get_catalog_text()
        elapsed = (time.perf_counter() - start) / 1000

        assert elapsed < 0.001, f"get_catalog_text() took {elapsed*1000:.3f}ms, expected < 1ms"


# ===========================================================================
# T5: Tool Integration
# ===========================================================================

class TestToolIntegration:
    """T5: list_models action via SystemTool."""

    def test_t5_1_list_models_returns_catalog(self, tmp_registry_path: Path, sample_registry_data: dict):
        """T5.1: mirai_system(action='list_models') returns formatted table."""
        from mirai.agent.tools.system import SystemTool
        registry = _make_registry_from_data(tmp_registry_path, sample_registry_data)
        tool = SystemTool(registry=registry)
        result = tool._list_models()
        assert "minimax" in result.lower()
        assert "MiniMax-M2.5" in result

    def test_t5_2_marks_current_model(self, tmp_registry_path: Path, sample_registry_data: dict):
        """T5.2: Response marks current model with ← current."""
        registry = _make_registry_from_data(tmp_registry_path, sample_registry_data)
        text = registry.get_catalog_text()
        assert "← current" in text

    def test_t5_3_unavailable_not_listed(self, tmp_registry_path: Path, sample_registry_data: dict):
        """T5.3: Unavailable providers not shown in available list."""
        registry = _make_registry_from_data(tmp_registry_path, sample_registry_data)
        text = registry.get_catalog_text()
        # Anthropic is unavailable, should NOT appear as a section header
        assert "ANTHROPIC" not in text

    def test_t5_no_registry(self):
        """SystemTool with no registry returns error."""
        from mirai.agent.tools.system import SystemTool
        tool = SystemTool()
        result = tool._list_models()
        assert "Error" in result


# ===========================================================================
# T6: Regression
# ===========================================================================

class TestRegression:
    """T6: No regression in existing functionality."""

    def test_t6_3_system_prompt_no_full_catalog(self, tmp_registry_path: Path):
        """T6.3: RUNTIME INFO should be lightweight, not full model list."""
        # This test verifies the loop.py change — just check the string
        # The actual loop test is in the existing test suite
        from mirai.agent.loop import AgentLoop
        # Verify AgentLoop doesn't reference MODEL_CATALOG anymore
        import inspect
        source = inspect.getsource(AgentLoop._build_system_prompt)
        assert "MODEL_CATALOG" not in source

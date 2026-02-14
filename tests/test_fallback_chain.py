"""Tests for the fallback chain in AgentLoop._execute_cycle.

Verifies:
 - Primary model failure → fallback model succeeds
 - All models fail → RuntimeError propagated
 - No fallback attempted when primary succeeds
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from mirai.agent.agent_loop import AgentLoop, RefinedStep
from mirai.agent.models import ProviderResponse, TextBlock
from mirai.errors import ProviderError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_loop(provider, fallback_models: list[str] | None = None) -> AgentLoop:
    """Construct an AgentLoop without async init (identity is faked)."""
    loop = AgentLoop.__new__(AgentLoop)
    loop.provider = provider
    loop.tools = {}
    loop.collaborator_id = "test-collab"
    loop.fallback_models = fallback_models or []

    # Fake identity fields normally set by _initialize()
    loop.name = "TestAgent"
    loop.role = "tester"
    loop.base_system_prompt = "You are a test agent."
    loop.soul_content = ""

    # Stub storage
    from mirai.agent.providers import MockEmbeddingProvider
    from mirai.db.duck import DuckDBStorage
    from mirai.memory.vector_db import VectorStore

    loop.l3_storage = MagicMock(spec=DuckDBStorage)
    loop.l3_storage.append_trace = AsyncMock()
    loop.l2_storage = MagicMock(spec=VectorStore)
    loop.l2_storage.query = AsyncMock(return_value=[])
    loop.embedder = MockEmbeddingProvider()

    return loop


def _ok_response(text: str = "Hello", model: str = "test-model") -> ProviderResponse:
    """Build a minimal successful ProviderResponse."""
    return ProviderResponse(
        content=[TextBlock(text=text)],
        stop_reason="end_turn",
        model_id=model,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFallbackChain:
    @pytest.mark.asyncio
    async def test_fallback_on_primary_failure(self):
        """Primary model raises → fallback model succeeds."""
        provider = MagicMock()
        call_count = 0

        async def _gen(model, system, messages, tools):
            nonlocal call_count
            call_count += 1
            if model == "primary-model":
                raise RuntimeError("Primary model quota exhausted")
            return _ok_response(text=f"Fallback says hi via {model}", model=model)

        provider.generate_response = _gen
        provider.model = "primary-model"

        loop = _make_loop(provider, fallback_models=["fallback-model-a", "fallback-model-b"])

        # Patch _build_system_prompt to avoid real I/O
        loop._build_system_prompt = AsyncMock(return_value="system prompt")

        final_text = ""
        async for step in loop._execute_cycle("Hello", model="primary-model"):
            if isinstance(step, RefinedStep):
                final_text = step.text

        assert "Fallback says hi" in final_text
        assert "fallback-model-a" in final_text
        assert call_count == 2  # primary failed, first fallback succeeded

    @pytest.mark.asyncio
    async def test_all_models_fail(self):
        """All models (primary + fallbacks) fail → RuntimeError raised."""
        provider = MagicMock()

        async def _gen(model, system, messages, tools):
            raise RuntimeError(f"Model {model} unavailable")

        provider.generate_response = _gen
        provider.model = "primary"

        loop = _make_loop(provider, fallback_models=["fb-1", "fb-2"])
        loop._build_system_prompt = AsyncMock(return_value="system prompt")

        with pytest.raises(ProviderError, match="Model fb-2 unavailable"):
            async for _ in loop._execute_cycle("Hello", model="primary"):
                pass

    @pytest.mark.asyncio
    async def test_no_fallback_when_primary_succeeds(self):
        """Primary model works → no fallback attempted."""
        provider = MagicMock()
        models_tried: list[str] = []

        async def _gen(model, system, messages, tools):
            models_tried.append(model)
            return _ok_response(text="Primary works", model=model)

        provider.generate_response = _gen
        provider.model = "primary"

        loop = _make_loop(provider, fallback_models=["fb-1", "fb-2"])
        loop._build_system_prompt = AsyncMock(return_value="system prompt")

        async for _step in loop._execute_cycle("Hello", model="primary"):
            pass

        assert models_tried == ["primary"]

    @pytest.mark.asyncio
    async def test_fallback_skips_duplicate_primary(self):
        """If primary is also in fallback list, it is not retried."""
        provider = MagicMock()
        models_tried: list[str] = []

        async def _gen(model, system, messages, tools):
            models_tried.append(model)
            if model == "primary":
                raise RuntimeError("fail")
            return _ok_response(text="ok", model=model)

        provider.generate_response = _gen
        provider.model = "primary"

        # primary is duplicated in fallback list — should be deduped
        loop = _make_loop(provider, fallback_models=["primary", "fb-1"])
        loop._build_system_prompt = AsyncMock(return_value="system prompt")

        async for _step in loop._execute_cycle("Hello", model="primary"):
            pass

        # primary tried once, fb-1 tried once (no duplicate retry of primary)
        assert models_tried == ["primary", "fb-1"]

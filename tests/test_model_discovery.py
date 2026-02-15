"""QA tests for the provider model discovery & usage interface.

Test categories:
  1. Data model contracts — ModelInfo and UsageSnapshot dataclass defaults
  2. ProviderProtocol compliance — all providers expose the new methods
  3. list_models() — correctness for each provider
  4. get_usage() — correctness and error handling
  5. MiniMax helpers — _pick_number / _pick_string
  6. MiniMax get_usage() — API response parsing, HTTP errors, network failures
  7. HTTP endpoints — GET /models and GET /usage responses
  8. Edge cases — empty catalogs, provider without methods, etc.
"""

import time
from dataclasses import asdict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirai.agent.providers.base import ModelInfo, ProviderProtocol, UsageSnapshot


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_provider():
    from tests.mocks.providers import MockProvider

    return MockProvider()


@pytest.fixture
def openai_provider():
    from mirai.agent.providers.openai import OpenAIProvider

    return OpenAIProvider(api_key="test-key", model="gpt-4o")


@pytest.fixture
def anthropic_provider():
    from mirai.agent.providers.anthropic import AnthropicProvider

    return AnthropicProvider(api_key="test-key")


@pytest.fixture
def minimax_provider():
    from mirai.agent.providers.minimax import MiniMaxProvider

    return MiniMaxProvider(api_key="test-key")


# ---------------------------------------------------------------------------
# 1. Data model contracts
# ---------------------------------------------------------------------------
class TestModelInfo:
    """Verify ModelInfo dataclass has correct defaults and serialisation."""

    def test_minimal_construction(self):
        m = ModelInfo(id="gpt-4o", name="GPT-4o")
        assert m.id == "gpt-4o"
        assert m.name == "GPT-4o"
        assert m.description is None
        assert m.context_window is None
        assert m.max_output_tokens is None
        assert m.reasoning is False
        assert m.input_modalities == ["text"]
        assert m.output_modalities == ["text"]

    def test_minimal_defaults_for_capabilities(self):
        m = ModelInfo(id="x", name="X")
        assert m.supports_tool_use is True
        assert m.supports_streaming is True
        assert m.supports_json_mode is False
        assert m.supports_vision is False

    def test_minimal_defaults_for_pricing_and_lifecycle(self):
        m = ModelInfo(id="x", name="X")
        assert m.input_price is None
        assert m.output_price is None
        assert m.knowledge_cutoff is None
        assert m.deprecation_date is None

    def test_full_construction(self):
        m = ModelInfo(
            id="MiniMax-M2.5",
            name="MiniMax M2.5",
            description="Advanced reasoning model",
            context_window=200_000,
            max_output_tokens=8192,
            reasoning=True,
            supports_tool_use=True,
            supports_vision=True,
            supports_json_mode=True,
            input_modalities=["text", "image"],
            input_price=1.1,
            output_price=4.4,
            knowledge_cutoff="2025-06",
        )
        assert m.context_window == 200_000
        assert m.max_output_tokens == 8192
        assert m.reasoning is True
        assert m.description == "Advanced reasoning model"
        assert m.supports_vision is True
        assert m.input_price == 1.1
        assert m.knowledge_cutoff == "2025-06"

    def test_asdict_serialisation(self):
        m = ModelInfo(id="test", name="Test Model")
        d = asdict(m)
        assert isinstance(d, dict)
        assert d["id"] == "test"
        assert d["input_modalities"] == ["text"]
        assert d["output_modalities"] == ["text"]
        assert "supports_tool_use" in d
        assert "input_price" in d

    def test_default_modalities_are_independent(self):
        """Each instance should get its own list (mutable default gotcha)."""
        m1 = ModelInfo(id="a", name="A")
        m2 = ModelInfo(id="b", name="B")
        m1.input_modalities.append("image")
        assert "image" not in m2.input_modalities

    def test_output_modalities_are_independent(self):
        m1 = ModelInfo(id="a", name="A")
        m2 = ModelInfo(id="b", name="B")
        m1.output_modalities.append("image")
        assert "image" not in m2.output_modalities

    def test_equality(self):
        m1 = ModelInfo(id="x", name="X", context_window=100)
        m2 = ModelInfo(id="x", name="X", context_window=100)
        assert m1 == m2


class TestUsageSnapshot:
    """Verify UsageSnapshot dataclass defaults and serialisation."""

    def test_defaults(self):
        u = UsageSnapshot()
        assert u.provider == ""
        assert u.used_percent is None
        assert u.plan is None
        assert u.reset_at is None
        assert u.error is None

    def test_error_snapshot(self):
        u = UsageSnapshot(provider="test", error="not supported")
        assert u.error == "not supported"
        assert u.used_percent is None

    def test_usage_snapshot(self):
        u = UsageSnapshot(provider="minimax", used_percent=42.5, plan="pro")
        assert u.used_percent == 42.5
        assert u.plan == "pro"

    def test_asdict_serialisation(self):
        u = UsageSnapshot(provider="minimax", used_percent=10.0)
        d = asdict(u)
        assert d["provider"] == "minimax"
        assert d["used_percent"] == 10.0
        assert d["error"] is None


# ---------------------------------------------------------------------------
# 2. ProviderProtocol compliance (extended)
# ---------------------------------------------------------------------------
class TestProviderProtocolExtended:
    """All providers satisfy the extended ProviderProtocol."""

    def test_mock_provider_has_new_methods(self, mock_provider):
        assert isinstance(mock_provider, ProviderProtocol)
        assert hasattr(mock_provider, "provider_name")
        assert hasattr(mock_provider, "list_models")
        assert hasattr(mock_provider, "get_usage")

    def test_openai_provider_has_new_methods(self, openai_provider):
        assert isinstance(openai_provider, ProviderProtocol)
        assert hasattr(openai_provider, "provider_name")
        assert hasattr(openai_provider, "list_models")
        assert hasattr(openai_provider, "get_usage")

    def test_anthropic_provider_has_new_methods(self, anthropic_provider):
        assert isinstance(anthropic_provider, ProviderProtocol)

    def test_minimax_provider_has_new_methods(self, minimax_provider):
        assert isinstance(minimax_provider, ProviderProtocol)

    def test_antigravity_provider_has_new_methods(self):
        """AntigravityProvider should also satisfy the protocol."""
        with patch(
            "mirai.agent.providers.antigravity.load_credentials", return_value={"token": "t", "expires": 9999999999}
        ):
            from mirai.agent.providers.antigravity import AntigravityProvider

            p = AntigravityProvider()
            assert hasattr(p, "provider_name")
            assert hasattr(p, "list_models")
            assert hasattr(p, "get_usage")


# ---------------------------------------------------------------------------
# 3. provider_name
# ---------------------------------------------------------------------------
class TestProviderName:
    """Each provider returns its canonical name."""

    def test_mock(self, mock_provider):
        assert mock_provider.provider_name == "mock"

    def test_openai(self, openai_provider):
        assert openai_provider.provider_name == "openai"

    def test_anthropic(self, anthropic_provider):
        assert anthropic_provider.provider_name == "anthropic"

    def test_minimax(self, minimax_provider):
        assert minimax_provider.provider_name == "minimax"

    def test_antigravity(self):
        with patch(
            "mirai.agent.providers.antigravity.load_credentials", return_value={"token": "t", "expires": 9999999999}
        ):
            from mirai.agent.providers.antigravity import AntigravityProvider

            assert AntigravityProvider().provider_name == "antigravity"

    def test_provider_name_is_string(self, mock_provider, openai_provider, minimax_provider):
        """provider_name must always be a plain str, not a MagicMock or other type."""
        for p in [mock_provider, openai_provider, minimax_provider]:
            assert isinstance(p.provider_name, str)
            assert len(p.provider_name) > 0


# ---------------------------------------------------------------------------
# 4. list_models()
# ---------------------------------------------------------------------------
class TestListModels:
    """Test list_models() returns correct results for each provider."""

    @pytest.mark.asyncio
    async def test_mock_returns_one_model(self, mock_provider):
        models = await mock_provider.list_models()
        assert len(models) == 1
        assert models[0].id == "mock-model"

    @pytest.mark.asyncio
    async def test_minimax_catalog_has_five_models(self, minimax_provider):
        models = await minimax_provider.list_models()
        assert len(models) == 5

    @pytest.mark.asyncio
    async def test_minimax_catalog_model_ids(self, minimax_provider):
        models = await minimax_provider.list_models()
        ids = {m.id for m in models}
        assert "MiniMax-M2.1" in ids
        assert "MiniMax-M2.1-lightning" in ids
        assert "MiniMax-M2.5" in ids
        assert "MiniMax-M2.5-Lightning" in ids
        assert "MiniMax-VL-01" in ids

    @pytest.mark.asyncio
    async def test_minimax_vl_model_has_image_modality(self, minimax_provider):
        models = await minimax_provider.list_models()
        vl = [m for m in models if m.id == "MiniMax-VL-01"][0]
        assert "image" in vl.input_modalities
        assert vl.supports_vision is True

    @pytest.mark.asyncio
    async def test_minimax_models_have_descriptions(self, minimax_provider):
        models = await minimax_provider.list_models()
        for m in models:
            assert m.description is not None
            assert len(m.description) > 0

    @pytest.mark.asyncio
    async def test_minimax_models_have_pricing(self, minimax_provider):
        models = await minimax_provider.list_models()
        for m in models:
            assert m.input_price is not None
            assert m.output_price is not None
            assert m.input_price > 0

    @pytest.mark.asyncio
    async def test_minimax_models_have_knowledge_cutoff(self, minimax_provider):
        models = await minimax_provider.list_models()
        for m in models:
            assert m.knowledge_cutoff is not None

    @pytest.mark.asyncio
    async def test_minimax_m25_is_reasoning(self, minimax_provider):
        models = await minimax_provider.list_models()
        m25 = [m for m in models if m.id == "MiniMax-M2.5"][0]
        assert m25.reasoning is True

    @pytest.mark.asyncio
    async def test_minimax_m21_is_not_reasoning(self, minimax_provider):
        models = await minimax_provider.list_models()
        m21 = [m for m in models if m.id == "MiniMax-M2.1"][0]
        assert m21.reasoning is False

    @pytest.mark.asyncio
    async def test_anthropic_catalog_has_models(self, anthropic_provider):
        models = await anthropic_provider.list_models()
        assert len(models) >= 3
        ids = {m.id for m in models}
        assert "claude-sonnet-4-20250514" in ids

    @pytest.mark.asyncio
    async def test_anthropic_models_have_vision(self, anthropic_provider):
        models = await anthropic_provider.list_models()
        for m in models:
            assert m.supports_vision is True
            assert "image" in m.input_modalities

    @pytest.mark.asyncio
    async def test_anthropic_models_have_pricing(self, anthropic_provider):
        models = await anthropic_provider.list_models()
        for m in models:
            assert m.input_price is not None
            assert m.output_price is not None

    @pytest.mark.asyncio
    async def test_openai_with_catalog_returns_catalog(self):
        """If MODEL_CATALOG is set, list_models() returns it without API call."""
        from mirai.agent.providers.openai import OpenAIProvider

        class CustomProvider(OpenAIProvider):
            MODEL_CATALOG = [ModelInfo(id="test-model", name="Test")]

        p = CustomProvider(api_key="test")
        models = await p.list_models()
        assert len(models) == 1
        assert models[0].id == "test-model"

    @pytest.mark.asyncio
    async def test_openai_without_catalog_calls_api(self, openai_provider):
        """Without MODEL_CATALOG, list_models() queries the API."""
        mock_model = MagicMock()
        mock_model.id = "gpt-4o"

        # Mock the async iterator returned by client.models.list()
        openai_provider.client.models.list = AsyncMock(return_value=[mock_model])
        models = await openai_provider.list_models()
        assert len(models) == 1
        assert models[0].id == "gpt-4o"

    @pytest.mark.asyncio
    async def test_openai_api_failure_returns_fallback(self, openai_provider):
        """On API error, should return the current model as fallback."""
        openai_provider.client.models.list = AsyncMock(side_effect=Exception("connection error"))
        models = await openai_provider.list_models()
        assert len(models) == 1
        assert models[0].id == "gpt-4o"

    @pytest.mark.asyncio
    async def test_list_models_returns_new_list(self, minimax_provider):
        """Each call should return a new list (not a reference to the catalog)."""
        models1 = await minimax_provider.list_models()
        models2 = await minimax_provider.list_models()
        assert models1 is not models2

    @pytest.mark.asyncio
    async def test_all_models_have_required_fields(self, minimax_provider, anthropic_provider, mock_provider):
        """Every model from every provider must have id and name."""
        for provider in [minimax_provider, anthropic_provider, mock_provider]:
            models = await provider.list_models()
            for m in models:
                assert isinstance(m.id, str) and len(m.id) > 0
                assert isinstance(m.name, str) and len(m.name) > 0


# ---------------------------------------------------------------------------
# 5. get_usage()
# ---------------------------------------------------------------------------
class TestGetUsage:
    """Test get_usage() for each provider."""

    @pytest.mark.asyncio
    async def test_mock_returns_not_supported(self, mock_provider):
        usage = await mock_provider.get_usage()
        assert usage.provider == "mock"
        assert usage.error == "not supported"

    @pytest.mark.asyncio
    async def test_openai_returns_not_supported(self, openai_provider):
        usage = await openai_provider.get_usage()
        assert usage.provider == "openai"
        assert usage.error == "not supported"

    @pytest.mark.asyncio
    async def test_anthropic_returns_not_supported(self, anthropic_provider):
        usage = await anthropic_provider.get_usage()
        assert usage.provider == "anthropic"
        assert usage.error == "not supported"

    @pytest.mark.asyncio
    async def test_usage_snapshot_is_serialisable(self, openai_provider):
        usage = await openai_provider.get_usage()
        d = asdict(usage)
        assert isinstance(d, dict)
        assert "provider" in d
        assert "error" in d


# ---------------------------------------------------------------------------
# 6. MiniMax helpers — _pick_number / _pick_string
# ---------------------------------------------------------------------------
class TestMiniMaxHelpers:
    """Test the _pick_number and _pick_string utility functions."""

    def test_pick_number_first_key(self):
        from mirai.agent.providers.minimax import _pick_number

        assert _pick_number({"used": 42, "total": 100}, ["used", "usage"]) == 42.0

    def test_pick_number_second_key(self):
        from mirai.agent.providers.minimax import _pick_number

        assert _pick_number({"usage": 99}, ["used", "usage"]) == 99.0

    def test_pick_number_string_value(self):
        from mirai.agent.providers.minimax import _pick_number

        assert _pick_number({"used": "42.5"}, ["used"]) == 42.5

    def test_pick_number_invalid_string(self):
        from mirai.agent.providers.minimax import _pick_number

        assert _pick_number({"used": "not-a-number"}, ["used"]) is None

    def test_pick_number_missing_keys(self):
        from mirai.agent.providers.minimax import _pick_number

        assert _pick_number({"foo": 1}, ["used", "usage"]) is None

    def test_pick_number_empty_record(self):
        from mirai.agent.providers.minimax import _pick_number

        assert _pick_number({}, ["used"]) is None

    def test_pick_number_zero(self):
        from mirai.agent.providers.minimax import _pick_number

        assert _pick_number({"used": 0}, ["used"]) == 0.0

    def test_pick_number_float(self):
        from mirai.agent.providers.minimax import _pick_number

        assert _pick_number({"used": 3.14}, ["used"]) == 3.14

    def test_pick_string_first_key(self):
        from mirai.agent.providers.minimax import _pick_string

        assert _pick_string({"plan": "pro"}, ["plan", "tier"]) == "pro"

    def test_pick_string_second_key(self):
        from mirai.agent.providers.minimax import _pick_string

        assert _pick_string({"tier": "enterprise"}, ["plan", "tier"]) == "enterprise"

    def test_pick_string_empty_value(self):
        from mirai.agent.providers.minimax import _pick_string

        assert _pick_string({"plan": "  "}, ["plan"]) is None

    def test_pick_string_missing_key(self):
        from mirai.agent.providers.minimax import _pick_string

        assert _pick_string({"foo": "bar"}, ["plan"]) is None

    def test_pick_string_non_string_value(self):
        from mirai.agent.providers.minimax import _pick_string

        assert _pick_string({"plan": 123}, ["plan"]) is None

    def test_pick_string_strips_whitespace(self):
        from mirai.agent.providers.minimax import _pick_string

        assert _pick_string({"plan": "  pro  "}, ["plan"]) == "pro"


# ---------------------------------------------------------------------------
# 7. MiniMax get_usage() — API response parsing
# ---------------------------------------------------------------------------
class TestMiniMaxGetUsage:
    """Test MiniMaxProvider.get_usage() with various mocked HTTP responses."""

    def _make_provider(self):
        from mirai.agent.providers.minimax import MiniMaxProvider

        return MiniMaxProvider(api_key="test-key")

    @pytest.mark.asyncio
    async def test_successful_usage_with_model_remains(self):
        """Real MiniMax response uses model_remains array.

        current_interval_usage_count is the REMAINING count.
        """
        provider = self._make_provider()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "base_resp": {"status_code": 0, "status_msg": "success"},
            "model_remains": [
                {
                    "start_time": 1770984000000,
                    "end_time": 1770998400000,
                    "remains_time": 11128818,
                    "current_interval_total_count": 1500,
                    "current_interval_usage_count": 1200,  # 1200 remaining
                    "model_name": "MiniMax-M2",
                }
            ],
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            usage = await provider.get_usage()

        assert usage.provider == "minimax"
        # consumed = 1500 - 1200 = 300, used_pct = 300/1500 * 100 = 20.0
        assert usage.used_percent == 20.0
        assert usage.plan == "MiniMax-M2"
        assert usage.reset_at is not None
        assert usage.error is None

    @pytest.mark.asyncio
    async def test_successful_usage_with_remaining(self):
        """Response with remaining/total should compute used_percent from remainder."""
        provider = self._make_provider()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "base_resp": {"status_code": 0},
            "data": {"remaining": 70, "total": 100},
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            usage = await provider.get_usage()

        assert usage.used_percent == 30.0

    @pytest.mark.asyncio
    async def test_http_error(self):
        """Non-200 status should return error snapshot."""
        provider = self._make_provider()
        mock_resp = MagicMock()
        mock_resp.status_code = 403

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            usage = await provider.get_usage()

        assert usage.error == "HTTP 403"
        assert usage.used_percent is None

    @pytest.mark.asyncio
    async def test_api_error_in_base_resp(self):
        """MiniMax error in base_resp should surface as error message."""
        provider = self._make_provider()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "base_resp": {"status_code": 1001, "status_msg": "insufficient balance"},
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            usage = await provider.get_usage()

        assert usage.error == "insufficient balance"

    @pytest.mark.asyncio
    async def test_network_exception(self):
        """Network errors should be caught and returned as error snapshot."""
        provider = self._make_provider()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(side_effect=ConnectionError("timeout"))
            mock_client_cls.return_value = mock_client

            usage = await provider.get_usage()

        assert usage.provider == "minimax"
        assert "timeout" in usage.error

    @pytest.mark.asyncio
    async def test_missing_data_key(self):
        """Response without 'data' key should still work (fallback to top-level)."""
        provider = self._make_provider()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "base_resp": {"status_code": 0},
            "used": 50,
            "total": 200,
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            usage = await provider.get_usage()

        assert usage.used_percent == 25.0

    @pytest.mark.asyncio
    async def test_zero_total_avoids_division_by_zero(self):
        """Zero total should not crash — used_percent stays None."""
        provider = self._make_provider()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "base_resp": {"status_code": 0},
            "data": {"used": 0, "total": 0},
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            usage = await provider.get_usage()

        assert usage.used_percent is None  # no division by zero


# ---------------------------------------------------------------------------
# 8. HTTP endpoints — GET /models and GET /usage
# ---------------------------------------------------------------------------
class TestModelEndpoint:
    """Test GET /models HTTP endpoint."""

    @pytest.fixture
    def endpoint_client(self):
        from fastapi import FastAPI
        from starlette.testclient import TestClient

        import main as main_module

        saved = main_module._mirai

        # Build a MiraiApp with real MockProvider
        from mirai.bootstrap import MiraiApp
        from tests.mocks.providers import MockProvider

        mirai_app = MiraiApp()
        agent = MagicMock()
        agent.provider = MockProvider()
        mirai_app.agent = agent
        mirai_app.start_time = time.monotonic()
        main_module._mirai = mirai_app

        test_app = FastAPI()
        test_app.add_api_route("/models", main_module.list_models, methods=["GET"])
        test_app.add_api_route("/usage", main_module.get_usage, methods=["GET"])
        test_app.add_api_route("/health", main_module.health_check, methods=["GET"])

        with TestClient(test_app, raise_server_exceptions=False) as c:
            yield c
        main_module._mirai = saved

    @pytest.fixture
    def no_provider_client(self):
        from fastapi import FastAPI
        from starlette.testclient import TestClient

        import main as main_module

        saved = main_module._mirai

        from mirai.bootstrap import MiraiApp

        mirai_app = MiraiApp()
        mirai_app.agent = None
        mirai_app.start_time = time.monotonic()
        main_module._mirai = mirai_app

        test_app = FastAPI()
        test_app.add_api_route("/models", main_module.list_models, methods=["GET"])
        test_app.add_api_route("/usage", main_module.get_usage, methods=["GET"])

        with TestClient(test_app, raise_server_exceptions=False) as c:
            yield c
        main_module._mirai = saved

    def test_models_returns_list(self, endpoint_client):
        resp = endpoint_client.get("/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert isinstance(data["models"], list)
        assert len(data["models"]) >= 1

    def test_models_contains_provider(self, endpoint_client):
        resp = endpoint_client.get("/models")
        data = resp.json()
        assert data["provider"] == "mock"

    def test_models_contains_current_model(self, endpoint_client):
        resp = endpoint_client.get("/models")
        data = resp.json()
        assert data["current_model"] == "mock-model"

    def test_models_entries_have_id_and_name(self, endpoint_client):
        resp = endpoint_client.get("/models")
        for m in resp.json()["models"]:
            assert "id" in m
            assert "name" in m

    def test_usage_returns_snapshot(self, endpoint_client):
        resp = endpoint_client.get("/usage")
        assert resp.status_code == 200
        data = resp.json()
        assert data["provider"] == "mock"
        assert data["error"] == "not supported"

    def test_models_without_agent_returns_error(self, no_provider_client):
        resp = no_provider_client.get("/models")
        data = resp.json()
        assert "error" in data

    def test_usage_without_agent_returns_error(self, no_provider_client):
        resp = no_provider_client.get("/usage")
        data = resp.json()
        assert "error" in data

    def test_health_includes_provider_name(self, endpoint_client):
        resp = endpoint_client.get("/health")
        data = resp.json()
        assert data["provider"] == "mock"
        assert data["model"] == "mock-model"


# ---------------------------------------------------------------------------
# 9. MiniMax provider defaults
# ---------------------------------------------------------------------------
class TestMiniMaxDefaults:
    """Verify MiniMax provider configuration."""

    def test_default_model(self):
        from mirai.agent.providers.minimax import MiniMaxProvider

        assert MiniMaxProvider.DEFAULT_MODEL == "MiniMax-M2.5"

    def test_default_base_url(self):
        from mirai.agent.providers.minimax import MiniMaxProvider

        assert "minimaxi.com" in MiniMaxProvider.DEFAULT_BASE_URL

    def test_catalog_is_class_level(self):
        from mirai.agent.providers.minimax import MiniMaxProvider

        assert len(MiniMaxProvider.MODEL_CATALOG) == 5

    def test_all_catalog_models_have_context_window(self):
        from mirai.agent.providers.minimax import MiniMaxProvider

        for m in MiniMaxProvider.MODEL_CATALOG:
            assert m.context_window is not None
            assert m.context_window > 0

    def test_all_catalog_models_have_max_output_tokens(self):
        from mirai.agent.providers.minimax import MiniMaxProvider

        for m in MiniMaxProvider.MODEL_CATALOG:
            assert m.max_output_tokens is not None
            assert m.max_output_tokens > 0

    def test_all_catalog_models_have_description(self):
        from mirai.agent.providers.minimax import MiniMaxProvider

        for m in MiniMaxProvider.MODEL_CATALOG:
            assert m.description is not None

    def test_all_catalog_models_have_pricing(self):
        from mirai.agent.providers.minimax import MiniMaxProvider

        for m in MiniMaxProvider.MODEL_CATALOG:
            assert m.input_price is not None
            assert m.output_price is not None

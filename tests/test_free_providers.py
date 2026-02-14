"""Integration tests for the free LLM provider system.

Tests cover:
  - FP1: FreeProviderSpec catalog integrity
  - FP2: FreeProviderSource caching
  - FP3: Registry integration (scanning, catalog display)
  - FP4: Factory dispatch (data-driven creation)
  - FP5: OpenAIProvider dynamic provider_name
  - FP6: Live smoke tests (require real API keys, marked @smoke)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirai.agent.free_providers import (
    FREE_PROVIDER_SPECS,
    FreeModelEntry,
    FreeProviderSource,
    get_free_provider_spec,
)
from mirai.agent.providers.base import ModelInfo

# ===========================================================================
# FP1: Catalog integrity
# ===========================================================================


class TestFreeProviderCatalog:
    """FP1: Verify the static catalog is well-formed."""

    def test_fp1_1_all_specs_have_required_fields(self):
        """Every spec should have name, env_key, base_url, signup_url."""
        for spec in FREE_PROVIDER_SPECS:
            assert spec.name, f"spec missing name: {spec}"
            assert spec.env_key, f"{spec.name} missing env_key"
            assert spec.base_url, f"{spec.name} missing base_url"
            assert spec.base_url.startswith("https://"), f"{spec.name} base_url not https"
            assert spec.signup_url, f"{spec.name} missing signup_url"
            assert spec.display_name, f"{spec.name} missing display_name"

    def test_fp1_2_no_duplicate_names(self):
        """Provider names should be unique."""
        names = [s.name for s in FREE_PROVIDER_SPECS]
        assert len(names) == len(set(names))

    def test_fp1_3_no_duplicate_env_keys(self):
        """Env keys should be unique across providers."""
        keys = [s.env_key for s in FREE_PROVIDER_SPECS]
        assert len(keys) == len(set(keys))

    def test_fp1_4_lookup_by_name(self):
        """get_free_provider_spec() should return exact matches."""
        for spec in FREE_PROVIDER_SPECS:
            found = get_free_provider_spec(spec.name)
            assert found is spec

    def test_fp1_5_lookup_unknown(self):
        """Unknown name should return None."""
        assert get_free_provider_spec("nonexistent_provider") is None

    def test_fp1_6_known_providers_present(self):
        """Verify the expected free providers are in the catalog."""
        names = {s.name for s in FREE_PROVIDER_SPECS}
        expected = {"groq", "openrouter", "cerebras", "mistral", "sambanova"}
        assert expected.issubset(names), f"Missing: {expected - names}"


# ===========================================================================
# FP2: FreeProviderSource caching
# ===========================================================================


class TestFreeProviderSourceCache:
    """FP2: Verify cache load/save/expiry logic."""

    @pytest.fixture
    def tmp_cache_path(self, tmp_path: Path) -> Path:
        return tmp_path / "free_providers_cache.json"

    def _make_source(self, cache_path: Path) -> FreeProviderSource:
        src = FreeProviderSource()
        src.CACHE_PATH = cache_path
        return src

    def test_fp2_1_no_cache_returns_none(self, tmp_cache_path: Path):
        """No cache file → _load_cache returns None."""
        src = self._make_source(tmp_cache_path)
        assert src._load_cache() is None

    def test_fp2_2_fresh_cache_returns_data(self, tmp_cache_path: Path):
        """Fresh cache → _load_cache returns data."""
        data = {
            "timestamp": time.time(),
            "providers": {
                "openrouter": {
                    "fetched_at": time.time(),
                    "models": [
                        {
                            "id": "deepseek/deepseek-r1:free",
                            "name": "DeepSeek R1",
                            "provider": "openrouter",
                            "limits": {"requests/day": 50},
                        }
                    ],
                }
            },
        }
        tmp_cache_path.write_text(json.dumps(data), encoding="utf-8")
        src = self._make_source(tmp_cache_path)
        result = src._load_cache()
        assert result is not None
        assert "openrouter" in result
        assert len(result["openrouter"].models) == 1
        assert result["openrouter"].models[0].id == "deepseek/deepseek-r1:free"

    def test_fp2_3_stale_cache_returns_none(self, tmp_cache_path: Path):
        """Stale cache (> TTL) → _load_cache returns None."""
        data = {
            "timestamp": time.time() - 7200,  # 2 hours ago
            "providers": {"openrouter": {"fetched_at": 0, "models": []}},
        }
        tmp_cache_path.write_text(json.dumps(data), encoding="utf-8")
        src = self._make_source(tmp_cache_path)
        assert src._load_cache() is None

    def test_fp2_4_corrupted_cache_returns_none(self, tmp_cache_path: Path):
        """Corrupted JSON → _load_cache returns None (fail-open)."""
        tmp_cache_path.write_text("{bad json!!!", encoding="utf-8")
        src = self._make_source(tmp_cache_path)
        assert src._load_cache() is None

    @pytest.mark.asyncio
    async def test_fp2_5_fetch_saves_cache(self, tmp_cache_path: Path):
        """After fetch(), cache file should be written."""
        src = self._make_source(tmp_cache_path)
        mock_models = [FreeModelEntry(id="test/model:free", name="Test Model", provider="openrouter")]
        with patch.object(src, "_fetch_openrouter", new_callable=AsyncMock, return_value=mock_models):
            with patch.object(src, "_fetch_sambanova", new_callable=AsyncMock, return_value=[]):
                await src.fetch()
        assert tmp_cache_path.exists()
        cached = json.loads(tmp_cache_path.read_text())
        assert "openrouter" in cached["providers"]

    @pytest.mark.asyncio
    async def test_fp2_6_fetch_uses_cache_when_fresh(self, tmp_cache_path: Path):
        """fetch() should use cache when fresh, not call APIs."""
        data = {
            "timestamp": time.time(),
            "providers": {
                "openrouter": {
                    "fetched_at": time.time(),
                    "models": [
                        {
                            "id": "cached/model:free",
                            "name": "Cached",
                            "provider": "openrouter",
                        }
                    ],
                }
            },
        }
        tmp_cache_path.write_text(json.dumps(data), encoding="utf-8")
        src = self._make_source(tmp_cache_path)
        mock_fetch_or = AsyncMock(return_value=[])
        with patch.object(src, "_fetch_openrouter", mock_fetch_or):
            result = await src.fetch()
        # Should NOT have called the API
        mock_fetch_or.assert_not_called()
        assert "openrouter" in result

    @pytest.mark.asyncio
    async def test_fp2_7_fetch_error_returns_empty_models(self, tmp_cache_path: Path):
        """API error → provider entry exists but with empty model list."""
        src = self._make_source(tmp_cache_path)
        with patch.object(src, "_fetch_openrouter", new_callable=AsyncMock, side_effect=Exception("Network error")):
            with patch.object(src, "_fetch_sambanova", new_callable=AsyncMock, return_value=[]):
                result = await src.fetch()
        assert "openrouter" in result
        assert result["openrouter"].models == []


# ===========================================================================
# FP3: Registry integration
# ===========================================================================


def _make_registry_from_data(path: Path, data: dict, **kwargs):
    """Create a ModelRegistry pre-loaded with data."""
    from mirai.agent.registry import ModelRegistry

    path.write_text(json.dumps(data), encoding="utf-8")
    registry = ModelRegistry.__new__(ModelRegistry)
    registry._config_provider = kwargs.get("config_provider")
    registry._config_model = kwargs.get("config_model")
    registry._enrichment_source = None
    registry.PATH = path
    from mirai.agent.registry_models import RegistryData

    registry._data = RegistryData.from_dict(data)
    return registry


class TestRegistryFreeProviderIntegration:
    """FP3: Free providers in registry scanning and catalog display."""

    @pytest.fixture
    def tmp_registry_path(self, tmp_path: Path) -> Path:
        return tmp_path / "model_registry.json"

    @pytest.mark.asyncio
    async def test_fp3_1_free_provider_with_key_scans_models(self, tmp_registry_path: Path):
        """Free provider with configured key → models discovered via list_models()."""
        data = {
            "version": 1,
            "active_provider": "minimax",
            "active_model": "MiniMax-M2.5",
            "providers": {},
        }
        registry = _make_registry_from_data(tmp_registry_path, data)

        mock_models = [
            ModelInfo(id="llama-3.3-70b-versatile", name="Llama 3.3 70B"),
            ModelInfo(id="gemma2-9b-it", name="Gemma 2 9B"),
        ]
        mock_provider = MagicMock()
        mock_provider.list_models = AsyncMock(return_value=mock_models)

        # Clear all env vars, set only GROQ_API_KEY
        env = {
            "GROQ_API_KEY": "test-groq-key",
            "MINIMAX_API_KEY": "",
            "ANTHROPIC_API_KEY": "",
            "OPENAI_API_KEY": "",
        }
        # Also clear the free provider keys
        for spec in FREE_PROVIDER_SPECS:
            if spec.name != "groq":
                env[spec.env_key] = ""

        with patch.dict(os.environ, env, clear=False):
            for key, val in env.items():
                if not val:
                    os.environ.pop(key, None)
            with patch("mirai.agent.registry._import_provider_class") as mock_import:
                mock_import.return_value = MagicMock(return_value=mock_provider)
                await registry.refresh()

        groq_data = registry._data.providers.get("groq")
        assert groq_data is not None
        assert groq_data.available is True
        assert len(groq_data.models) == 2
        assert groq_data.models[0].id == "llama-3.3-70b-versatile"

    @pytest.mark.asyncio
    async def test_fp3_2_free_provider_without_key_marked_unavailable(self, tmp_registry_path: Path):
        """Free provider without API key → available=False."""
        data = {
            "version": 1,
            "active_provider": "minimax",
            "active_model": "MiniMax-M2.5",
            "providers": {},
        }
        registry = _make_registry_from_data(tmp_registry_path, data)

        # Ensure no free provider keys are set
        env_clear = {}
        for spec in FREE_PROVIDER_SPECS:
            env_clear[spec.env_key] = ""
        env_clear.update({"MINIMAX_API_KEY": "", "ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": ""})

        with patch.dict(os.environ, env_clear, clear=False):
            for key in env_clear:
                os.environ.pop(key, None)
            await registry.refresh()

        for spec in FREE_PROVIDER_SPECS:
            pdata = registry._data.providers.get(spec.name)
            assert pdata is not None, f"Missing provider entry: {spec.name}"
            assert pdata.available is False

    def test_fp3_3_catalog_text_shows_unconfigured_free_providers(self, tmp_registry_path: Path):
        """get_catalog_text() should list unconfigured free providers with signup links."""
        data = {
            "version": 1,
            "active_provider": "minimax",
            "active_model": "MiniMax-M2.5",
            "providers": {
                "minimax": {
                    "available": True,
                    "env_key": "MINIMAX_API_KEY",
                    "models": [{"id": "MiniMax-M2.5", "name": "M2.5"}],
                },
            },
        }
        registry = _make_registry_from_data(tmp_registry_path, data)
        text = registry.get_catalog_text()

        # Should show the configured provider
        assert "MINIMAX" in text
        assert "MiniMax-M2.5" in text

        # Should show unconfigured free providers
        assert "Free providers (not configured)" in text
        assert "groq" in text.lower()
        assert "GROQ_API_KEY" in text
        assert "console.groq.com" in text

    def test_fp3_4_catalog_text_hides_configured_free_provider(self, tmp_registry_path: Path):
        """Configured free provider should NOT appear in 'not configured' section."""
        data = {
            "version": 1,
            "active_provider": "groq",
            "active_model": "llama-3.3-70b-versatile",
            "providers": {
                "groq": {
                    "available": True,
                    "env_key": "GROQ_API_KEY",
                    "models": [{"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B"}],
                },
            },
        }
        registry = _make_registry_from_data(tmp_registry_path, data)
        text = registry.get_catalog_text()

        # Groq should appear in the available section
        assert "GROQ (active)" in text

        # Groq should NOT appear in the unconfigured section
        lines = text.split("\n")
        unconfigured_section = False
        for line in lines:
            if "Free providers (not configured)" in line:
                unconfigured_section = True
            if unconfigured_section and "groq" in line.lower():
                pytest.fail("Configured provider 'groq' should not appear in unconfigured section")


# ===========================================================================
# FP4: Factory dispatch
# ===========================================================================


class TestFactoryFreeProviderDispatch:
    """FP4: Factory creates correct provider instances for free providers."""

    def test_fp4_1_groq_creates_openai_provider(self):
        """provider='groq' → OpenAIProvider with groq base_url."""
        from mirai.agent.providers.factory import create_provider

        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            provider = create_provider(provider="groq", model="llama-3.3-70b-versatile", api_key="test-key")

        from mirai.agent.providers.openai import OpenAIProvider

        assert isinstance(provider, OpenAIProvider)
        assert provider.provider_name == "groq"
        assert provider.model == "llama-3.3-70b-versatile"
        assert "groq.com" in str(provider.client.base_url)

    def test_fp4_2_openrouter_creates_openai_provider(self):
        """provider='openrouter' → OpenAIProvider with openrouter base_url."""
        from mirai.agent.providers.factory import create_provider

        provider = create_provider(provider="openrouter", model="deepseek/deepseek-r1:free", api_key="test-key")

        from mirai.agent.providers.openai import OpenAIProvider

        assert isinstance(provider, OpenAIProvider)
        assert provider.provider_name == "openrouter"
        assert "openrouter.ai" in str(provider.client.base_url)

    def test_fp4_3_cerebras_creates_openai_provider(self):
        """provider='cerebras' → OpenAIProvider with cerebras base_url."""
        from mirai.agent.providers.factory import create_provider

        provider = create_provider(provider="cerebras", model="llama-3.3-70b", api_key="test-key")

        from mirai.agent.providers.openai import OpenAIProvider

        assert isinstance(provider, OpenAIProvider)
        assert provider.provider_name == "cerebras"
        assert "cerebras.ai" in str(provider.client.base_url)

    def test_fp4_4_missing_key_raises_with_signup_url(self):
        """Free provider without key → ProviderError with signup URL."""
        from mirai.agent.providers.factory import create_provider
        from mirai.errors import ProviderError

        # Clear all possible keys
        env = {s.env_key: "" for s in FREE_PROVIDER_SPECS}
        with patch.dict(os.environ, env, clear=False):
            for key in env:
                os.environ.pop(key, None)
            with pytest.raises(ProviderError, match="Signup"):
                create_provider(provider="groq", model="llama-3.3-70b-versatile")

    def test_fp4_5_custom_base_url_overrides_default(self):
        """Explicit base_url in config takes precedence over catalog base_url."""
        from mirai.agent.providers.factory import create_provider

        custom_url = "https://my-proxy.example.com/v1"
        provider = create_provider(
            provider="groq",
            model="llama-3.3-70b-versatile",
            api_key="test-key",
            base_url=custom_url,
        )
        assert custom_url in str(provider.client.base_url)

    def test_fp4_6_unknown_provider_falls_through_to_openai(self):
        """Unknown provider (not in free specs) → falls to generic OpenAI."""
        from mirai.agent.providers.factory import create_provider

        provider = create_provider(
            provider="my-custom-provider",
            model="some-model",
            api_key="test-key",
            base_url="https://custom.example.com/v1",
        )

        from mirai.agent.providers.openai import OpenAIProvider

        assert isinstance(provider, OpenAIProvider)
        assert provider.provider_name == "my-custom-provider"

    def test_fp4_7_builtin_providers_not_routed_to_free(self):
        """Built-in providers (minimax, anthropic) should use their own classes."""
        from mirai.agent.providers.factory import create_provider
        from mirai.agent.providers.minimax import MiniMaxProvider

        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            provider = create_provider(provider="minimax", model="MiniMax-M2.5", api_key="test-key")

        assert isinstance(provider, MiniMaxProvider)
        assert provider.provider_name == "minimax"


# ===========================================================================
# FP5: OpenAIProvider dynamic provider_name
# ===========================================================================


class TestOpenAIProviderDynamicName:
    """FP5: OpenAIProvider respects the provider_name parameter."""

    def test_fp5_1_default_name(self):
        """No provider_name → defaults to 'openai'."""
        from mirai.agent.providers.openai import OpenAIProvider

        p = OpenAIProvider(api_key="test")
        assert p.provider_name == "openai"

    def test_fp5_2_custom_name(self):
        """Explicit provider_name → stored and returned."""
        from mirai.agent.providers.openai import OpenAIProvider

        p = OpenAIProvider(api_key="test", provider_name="groq")
        assert p.provider_name == "groq"

    def test_fp5_3_subclass_not_affected(self):
        """MiniMaxProvider (subclass) should still return 'minimax'."""
        from mirai.agent.providers.minimax import MiniMaxProvider

        p = MiniMaxProvider(api_key="test")
        assert p.provider_name == "minimax"


# ===========================================================================
# FP6: Build provider specs
# ===========================================================================


class TestBuildProviderSpecs:
    """FP6: _build_provider_specs merges built-in and free providers."""

    def test_fp6_1_includes_builtins(self):
        """Should include minimax, anthropic, openai."""
        from mirai.agent.registry import _build_provider_specs

        specs = _build_provider_specs()
        names = {s.name for s in specs}
        assert {"minimax", "anthropic", "openai"}.issubset(names)

    def test_fp6_2_includes_free_providers(self):
        """Should include groq, openrouter, etc."""
        from mirai.agent.registry import _build_provider_specs

        specs = _build_provider_specs()
        names = {s.name for s in specs}
        assert {"groq", "openrouter", "cerebras"}.issubset(names)

    def test_fp6_3_no_duplicates(self):
        """No duplicate provider names."""
        from mirai.agent.registry import _build_provider_specs

        specs = _build_provider_specs()
        names = [s.name for s in specs]
        assert len(names) == len(set(names))

    def test_fp6_4_free_providers_use_openai_class(self):
        """Free providers should use OpenAIProvider import path."""
        from mirai.agent.registry import _build_provider_specs

        specs = _build_provider_specs()
        for s in specs:
            if s.name in {"groq", "openrouter", "cerebras", "mistral", "sambanova"}:
                assert "OpenAIProvider" in s.import_path

    def test_fp6_5_free_providers_have_base_url(self):
        """Free providers should have base_url set."""
        from mirai.agent.registry import _build_provider_specs

        specs = _build_provider_specs()
        for s in specs:
            if s.name in {"groq", "openrouter", "cerebras", "mistral", "sambanova"}:
                assert s.base_url is not None
                assert s.base_url.startswith("https://")


# ===========================================================================
# FP7: Live smoke tests (require real API keys)
# ===========================================================================


def _skip_unless_key(env_var: str) -> str:
    """Return the key or skip the test if the env var is unset."""
    key = os.environ.get(env_var, "")
    if not key or key.startswith("xxx"):
        pytest.skip(f"{env_var} not configured")
    return key


@pytest.mark.smoke
class TestFreeProviderLiveSmoke:
    """FP7: Live smoke tests — real API calls to free providers.

    Run with: pytest tests/test_free_providers.py -m smoke -v
    """

    @pytest.mark.asyncio
    async def test_fp7_1_openrouter_public_api(self):
        """OpenRouter /models is public — should return free models without a key."""
        src = FreeProviderSource()
        models = await src._fetch_openrouter()
        assert len(models) > 0
        # All should have :free suffix
        for m in models:
            assert ":free" in m.id
            assert m.provider == "openrouter"

    @pytest.mark.asyncio
    async def test_fp7_2_sambanova_public_api(self):
        """SambaNova /api/pricing is public — should return models without a key."""
        src = FreeProviderSource()
        models = await src._fetch_sambanova()
        assert len(models) > 0
        for m in models:
            assert m.provider == "sambanova"
            assert m.id  # non-empty ID

    @pytest.mark.asyncio
    async def test_fp7_3_groq_chat_completion(self):
        """Groq — send a trivial prompt and verify response."""
        api_key = _skip_unless_key("GROQ_API_KEY")
        from mirai.agent.providers.factory import create_provider

        provider = create_provider(provider="groq", model="llama-3.3-70b-versatile", api_key=api_key)
        response = await provider.generate_response(
            model="llama-3.3-70b-versatile",
            system="Reply in one sentence.",
            messages=[{"role": "user", "content": "Say hello."}],
            tools=[],
        )
        assert response is not None
        from mirai.agent.models import TextBlock

        text_blocks = [b for b in response.content if isinstance(b, TextBlock)]
        assert len(text_blocks) >= 1
        assert len(text_blocks[0].text.strip()) > 0

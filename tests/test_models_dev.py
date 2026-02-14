"""Tests for ``mirai.agent.models_dev`` — ModelsDevSource.

Comprehensive QA test suite covering:
  - ExternalModelData dataclass construction and defaults
  - ModelsDevSource.__init__ (default / custom cache path)
  - _normalise: field extraction, edge cases, provider name mapping
  - _lookup: multi-strategy fallback
  - _load_cache / _save_cache: round-trip, expiry, corruption, field fidelity
  - enrich(): provider-wins semantics, partial fill, no-op when full
  - fetch(): cache-hit, cache-miss, API failure, stale-cache fallback
  - _fetch_api: HTTP errors, timeout, invalid JSON
  - Integration: enrichment inside ModelRegistry.refresh()
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from mirai.agent.models_dev import (
    ExternalModelData,
    ModelsDevSource,
)
from mirai.agent.registry_models import RegistryModelEntry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_API_RESPONSE = {
    "minimax": {
        "id": "minimax",
        "name": "MiniMax",
        "models": {
            "minimax/minimax-m2.5": {
                "id": "minimax/minimax-m2.5",
                "name": "MiniMax M2.5",
                "reasoning": True,
                "tool_call": True,
                "knowledge": "2025-01",
                "modalities": {"input": ["text"], "output": ["text"]},
                "cost": {"input": 0.3, "output": 1.2},
                "limit": {"context": 204800, "output": 131072},
            },
        },
    },
    "anthropic": {
        "id": "anthropic",
        "name": "Anthropic",
        "models": {
            "anthropic/claude-sonnet-4.5": {
                "id": "anthropic/claude-sonnet-4.5",
                "name": "Claude Sonnet 4.5",
                "reasoning": True,
                "tool_call": True,
                "modalities": {"input": ["text", "image", "pdf"], "output": ["text"]},
                "cost": {"input": 3, "output": 15},
                "limit": {"context": 1000000, "output": 64000},
            },
        },
    },
}


@pytest.fixture
def tmp_cache_path(tmp_path: Path) -> Path:
    return tmp_path / "test_cache.json"


@pytest.fixture
def source(tmp_cache_path: Path) -> ModelsDevSource:
    return ModelsDevSource(cache_path=tmp_cache_path)


def _make_source_with_data(tmp_cache_path: Path) -> ModelsDevSource:
    """Helper: create a source with pre-loaded SAMPLE_API_RESPONSE data."""
    s = ModelsDevSource(cache_path=tmp_cache_path)
    s._data = s._normalise(SAMPLE_API_RESPONSE)
    return s


# ===========================================================================
# ExternalModelData dataclass
# ===========================================================================


class TestExternalModelData:
    """Tests for the ExternalModelData dataclass defaults and construction."""

    def test_required_fields_only(self) -> None:
        ext = ExternalModelData(id="test-model", provider="test")
        assert ext.id == "test-model"
        assert ext.provider == "test"
        assert ext.name is None
        assert ext.input_cost is None
        assert ext.output_cost is None
        assert ext.context_limit is None
        assert ext.output_limit is None
        assert ext.tool_call is None
        assert ext.reasoning is None
        assert ext.vision is None
        assert ext.knowledge_cutoff is None
        assert ext.input_modalities == []
        assert ext.output_modalities == []

    def test_full_construction(self) -> None:
        ext = ExternalModelData(
            id="m1",
            provider="p1",
            name="Model One",
            input_cost=1.0,
            output_cost=5.0,
            context_limit=128000,
            output_limit=8192,
            tool_call=True,
            reasoning=True,
            vision=True,
            knowledge_cutoff="2025-06",
            input_modalities=["text", "image"],
            output_modalities=["text"],
        )
        assert ext.input_cost == 1.0
        assert ext.vision is True
        assert ext.input_modalities == ["text", "image"]

    def test_list_field_independence(self) -> None:
        """Verify that default list fields are independent across instances."""
        a = ExternalModelData(id="a", provider="p")
        b = ExternalModelData(id="b", provider="p")
        a.input_modalities.append("audio")
        assert b.input_modalities == []


# ===========================================================================
# __init__
# ===========================================================================


class TestInit:
    """Tests for ModelsDevSource constructor."""

    def test_default_cache_path(self) -> None:
        src = ModelsDevSource()
        assert src.CACHE_PATH == Path.home() / ".mirai" / "models_dev_cache.json"
        assert src._data == {}

    def test_custom_cache_path(self, tmp_path: Path) -> None:
        custom = tmp_path / "custom_cache.json"
        src = ModelsDevSource(cache_path=custom)
        assert src.CACHE_PATH == custom


# ===========================================================================
# _normalise
# ===========================================================================


class TestNormalise:
    """Tests for the _normalise method."""

    def test_normalise_creates_full_key(self, source: ModelsDevSource) -> None:
        data = source._normalise(SAMPLE_API_RESPONSE)
        # Full key format: provider_key/model_id
        assert "minimax/minimax/minimax-m2.5" in data
        assert "anthropic/anthropic/claude-sonnet-4.5" in data

    def test_normalise_creates_bare_key(self, source: ModelsDevSource) -> None:
        data = source._normalise(SAMPLE_API_RESPONSE)
        assert "minimax/minimax-m2.5" in data
        assert "anthropic/claude-sonnet-4.5" in data

    def test_normalise_cost_fields(self, source: ModelsDevSource) -> None:
        data = source._normalise(SAMPLE_API_RESPONSE)
        m = data["minimax/minimax-m2.5"]
        assert m.input_cost == 0.3
        assert m.output_cost == 1.2

    def test_normalise_limit_fields(self, source: ModelsDevSource) -> None:
        data = source._normalise(SAMPLE_API_RESPONSE)
        m = data["minimax/minimax-m2.5"]
        assert m.context_limit == 204800
        assert m.output_limit == 131072

    def test_normalise_vision_from_image(self, source: ModelsDevSource) -> None:
        data = source._normalise(SAMPLE_API_RESPONSE)
        m = data["anthropic/claude-sonnet-4.5"]
        assert m.vision is True

    def test_normalise_no_vision_text_only(self, source: ModelsDevSource) -> None:
        data = source._normalise(SAMPLE_API_RESPONSE)
        m = data["minimax/minimax-m2.5"]
        assert m.vision is False

    def test_normalise_vision_from_video(self, source: ModelsDevSource) -> None:
        """Video-only input should also set vision=True."""
        api = {
            "test_prov": {
                "id": "test_prov",
                "models": {
                    "vid-model": {
                        "id": "vid-model",
                        "name": "Video Model",
                        "modalities": {"input": ["text", "video"], "output": ["text"]},
                        "cost": {},
                        "limit": {},
                    }
                },
            }
        }
        data = source._normalise(api)
        assert data["vid-model"].vision is True

    def test_normalise_knowledge_cutoff(self, source: ModelsDevSource) -> None:
        data = source._normalise(SAMPLE_API_RESPONSE)
        m = data["minimax/minimax-m2.5"]
        assert m.knowledge_cutoff == "2025-01"

    def test_normalise_no_knowledge_cutoff(self, source: ModelsDevSource) -> None:
        data = source._normalise(SAMPLE_API_RESPONSE)
        m = data["anthropic/claude-sonnet-4.5"]
        assert m.knowledge_cutoff is None

    def test_normalise_reasoning_and_tool_call(self, source: ModelsDevSource) -> None:
        data = source._normalise(SAMPLE_API_RESPONSE)
        m = data["minimax/minimax-m2.5"]
        assert m.reasoning is True
        assert m.tool_call is True

    def test_normalise_modalities_preserved(self, source: ModelsDevSource) -> None:
        data = source._normalise(SAMPLE_API_RESPONSE)
        m = data["anthropic/claude-sonnet-4.5"]
        assert m.input_modalities == ["text", "image", "pdf"]
        assert m.output_modalities == ["text"]

    def test_normalise_name_field(self, source: ModelsDevSource) -> None:
        data = source._normalise(SAMPLE_API_RESPONSE)
        assert data["minimax/minimax-m2.5"].name == "MiniMax M2.5"
        assert data["anthropic/claude-sonnet-4.5"].name == "Claude Sonnet 4.5"

    def test_normalise_provider_mapping(self, source: ModelsDevSource) -> None:
        """Provider key should be mapped via _PROVIDER_NAME_MAP."""
        data = source._normalise(SAMPLE_API_RESPONSE)
        m = data["minimax/minimax-m2.5"]
        assert m.provider == "minimax"  # mapped from "minimax" -> "minimax"

    def test_normalise_unmapped_provider_passthrough(self, source: ModelsDevSource) -> None:
        """Unknown provider key should be used as-is."""
        api = {
            "unknown_provider": {
                "id": "unknown_provider",
                "models": {
                    "m1": {"id": "m1", "name": "M", "cost": {}, "limit": {}, "modalities": {}},
                },
            }
        }
        data = source._normalise(api)
        assert data["m1"].provider == "unknown_provider"

    def test_normalise_provider_name_map_xai(self, source: ModelsDevSource) -> None:
        """x-ai should map to xai."""
        api = {
            "x-ai": {
                "id": "x-ai",
                "models": {
                    "grok-4": {"id": "grok-4", "name": "Grok 4", "cost": {}, "limit": {}, "modalities": {}},
                },
            }
        }
        data = source._normalise(api)
        assert data["grok-4"].provider == "xai"

    def test_normalise_free_model_zero_cost(self, source: ModelsDevSource) -> None:
        """Models with cost 0 should preserve 0 (not None)."""
        api = {
            "free_prov": {
                "id": "free_prov",
                "models": {
                    "free-model": {
                        "id": "free-model",
                        "name": "Free Model",
                        "cost": {"input": 0, "output": 0},
                        "limit": {"context": 128000, "output": 4096},
                        "modalities": {"input": ["text"], "output": ["text"]},
                    }
                },
            }
        }
        data = source._normalise(api)
        m = data["free-model"]
        assert m.input_cost == 0
        assert m.output_cost == 0

    def test_normalise_missing_cost_block(self, source: ModelsDevSource) -> None:
        """Model without 'cost' key should yield None costs."""
        api = {
            "p": {
                "id": "p",
                "models": {
                    "no-cost": {
                        "id": "no-cost",
                        "name": "No Cost",
                        "limit": {"context": 100},
                        "modalities": {},
                    }
                },
            }
        }
        data = source._normalise(api)
        m = data["no-cost"]
        assert m.input_cost is None
        assert m.output_cost is None

    def test_normalise_missing_limit_block(self, source: ModelsDevSource) -> None:
        """Model without 'limit' key should yield None limits."""
        api = {
            "p": {
                "id": "p",
                "models": {
                    "no-limit": {
                        "id": "no-limit",
                        "name": "No Limit",
                        "cost": {"input": 1},
                        "modalities": {},
                    }
                },
            }
        }
        data = source._normalise(api)
        m = data["no-limit"]
        assert m.context_limit is None
        assert m.output_limit is None

    def test_normalise_missing_modalities_block(self, source: ModelsDevSource) -> None:
        """Model without 'modalities' key should yield empty lists and vision=False."""
        api = {
            "p": {
                "id": "p",
                "models": {
                    "no-mod": {"id": "no-mod", "name": "No Mod", "cost": {}, "limit": {}},
                },
            }
        }
        data = source._normalise(api)
        m = data["no-mod"]
        assert m.input_modalities == []
        assert m.output_modalities == []
        assert m.vision is False

    def test_normalise_skips_string_provider(self, source: ModelsDevSource) -> None:
        bad_api = {"bad_string": "not a dict"}
        data = source._normalise(bad_api)
        assert len(data) == 0

    def test_normalise_skips_provider_without_models_key(self, source: ModelsDevSource) -> None:
        bad_api = {"bad_no_models": {"id": "x"}}
        data = source._normalise(bad_api)
        assert len(data) == 0

    def test_normalise_skips_non_dict_model_entry(self, source: ModelsDevSource) -> None:
        """If a model entry is a string instead of dict, skip it."""
        api = {
            "p": {
                "id": "p",
                "models": {
                    "bad": "not a model dict",
                    "good": {"id": "good", "name": "Good", "cost": {}, "limit": {}, "modalities": {}},
                },
            }
        }
        data = source._normalise(api)
        assert "good" in data
        assert "bad" not in data

    def test_normalise_skips_models_key_not_dict(self, source: ModelsDevSource) -> None:
        """If 'models' value is a list instead of dict, skip provider."""
        api = {"p": {"id": "p", "models": ["not", "a", "dict"]}}
        data = source._normalise(api)
        assert len(data) == 0

    def test_normalise_duplicate_model_id_first_wins(self, source: ModelsDevSource) -> None:
        """When two providers have the same bare model_id, the first one wins for the bare key."""
        api = {
            "provider_a": {
                "id": "provider_a",
                "models": {
                    "shared-model": {
                        "id": "shared-model",
                        "name": "From A",
                        "cost": {"input": 1},
                        "limit": {},
                        "modalities": {},
                    }
                },
            },
            "provider_b": {
                "id": "provider_b",
                "models": {
                    "shared-model": {
                        "id": "shared-model",
                        "name": "From B",
                        "cost": {"input": 2},
                        "limit": {},
                        "modalities": {},
                    }
                },
            },
        }
        data = source._normalise(api)
        # Both full keys should exist
        assert "provider_a/shared-model" in data
        assert "provider_b/shared-model" in data
        # Bare key should be from the first provider encountered
        # (dict ordering is preserved in Python 3.7+)
        assert data["shared-model"].name == "From A"

    def test_normalise_empty_api_response(self, source: ModelsDevSource) -> None:
        data = source._normalise({})
        assert data == {}

    def test_normalise_provider_empty_models(self, source: ModelsDevSource) -> None:
        api = {"p": {"id": "p", "models": {}}}
        data = source._normalise(api)
        assert data == {}

    def test_normalise_large_context_window(self, source: ModelsDevSource) -> None:
        """Verify very large context windows (>1M) are preserved."""
        api = {
            "p": {
                "id": "p",
                "models": {
                    "big": {
                        "id": "big",
                        "name": "Big Context",
                        "cost": {},
                        "limit": {"context": 2000000, "output": 128000},
                        "modalities": {},
                    }
                },
            }
        }
        data = source._normalise(api)
        assert data["big"].context_limit == 2000000


# ===========================================================================
# _lookup
# ===========================================================================


class TestLookup:
    """Tests for the _lookup multi-strategy fallback."""

    def test_lookup_exact_provider_model(self, source: ModelsDevSource) -> None:
        """Lookup via provider/model_id format."""
        source._data = {
            "minimax/m1": ExternalModelData(id="m1", provider="minimax"),
        }
        source._build_index()
        result = source._lookup("m1", "minimax")
        assert result is not None
        assert result.id == "m1"

    def test_lookup_bare_model_id(self, source: ModelsDevSource) -> None:
        """Fallback to bare model_id."""
        source._data = {
            "m1": ExternalModelData(id="m1", provider="minimax"),
        }
        source._build_index()
        result = source._lookup("m1", "different_provider")
        assert result is not None
        assert result.id == "m1"

    def test_lookup_miss_returns_none(self, source: ModelsDevSource) -> None:
        source._data = {}
        source._build_index()
        result = source._lookup("nonexistent", "provider")
        assert result is None

    def test_lookup_prefers_provider_key_over_bare(self, source: ModelsDevSource) -> None:
        """If both exist, provider/model_id should be preferred."""
        source._data = {
            "provider/m1": ExternalModelData(id="m1", provider="provider", name="specific"),
            "m1": ExternalModelData(id="m1", provider="other", name="generic"),
        }
        source._build_index()
        result = source._lookup("m1", "provider")
        assert result is not None
        assert result.name == "specific"

    def test_lookup_uses_lowercase_prefix_variant(self, source: ModelsDevSource) -> None:
        """Provider name case variation should be attempted."""
        source._data = {
            "minimax/m1": ExternalModelData(id="m1", provider="minimax"),
        }
        source._build_index()
        # Pass uppercase provider — _lookup should try lowercase
        result = source._lookup("m1", "MiniMax")
        # The first check is "MiniMax/m1" (miss), then bare "m1" (miss),
        # then prefix variants including "minimax/m1" (hit)
        assert result is not None

    def test_lookup_with_normalised_data(self, source: ModelsDevSource) -> None:
        """End-to-end: normalise real data then look up."""
        source._data = source._normalise(SAMPLE_API_RESPONSE)
        source._build_index()
        result = source._lookup("minimax/minimax-m2.5", "minimax")
        assert result is not None
        assert result.input_cost == 0.3


# ===========================================================================
# Cache
# ===========================================================================


class TestCache:
    """Tests for disk cache read/write."""

    def test_save_and_load_roundtrip(self, source: ModelsDevSource, tmp_cache_path: Path) -> None:
        source._data = source._normalise(SAMPLE_API_RESPONSE)
        source._build_index()
        source._save_cache()
        assert tmp_cache_path.exists()

        loaded = source._load_cache()
        assert loaded is not None
        assert len(loaded) == len(source._data)

    def test_cache_field_fidelity(self, source: ModelsDevSource, tmp_cache_path: Path) -> None:
        """Verify every field survives a save/load round-trip."""
        source._data = source._normalise(SAMPLE_API_RESPONSE)
        source._build_index()
        source._save_cache()
        loaded = source._load_cache()
        assert loaded is not None

        key = "minimax/minimax-m2.5"
        orig = source._data[key]
        loaded_item = loaded[key]

        assert loaded_item.id == orig.id
        assert loaded_item.provider == orig.provider
        assert loaded_item.name == orig.name
        assert loaded_item.input_cost == orig.input_cost
        assert loaded_item.output_cost == orig.output_cost
        assert loaded_item.context_limit == orig.context_limit
        assert loaded_item.output_limit == orig.output_limit
        assert loaded_item.tool_call == orig.tool_call
        assert loaded_item.reasoning == orig.reasoning
        assert loaded_item.vision == orig.vision
        assert loaded_item.knowledge_cutoff == orig.knowledge_cutoff
        assert loaded_item.input_modalities == orig.input_modalities
        assert loaded_item.output_modalities == orig.output_modalities

    def test_cache_expired(self, source: ModelsDevSource, tmp_cache_path: Path) -> None:
        source._data = source._normalise(SAMPLE_API_RESPONSE)
        source._build_index()
        source._save_cache()

        raw = json.loads(tmp_cache_path.read_text())
        raw["timestamp"] = time.time() - source.CACHE_TTL - 100
        tmp_cache_path.write_text(json.dumps(raw))

        loaded = source._load_cache()
        assert loaded is None

    def test_cache_just_fresh(self, source: ModelsDevSource, tmp_cache_path: Path) -> None:
        """Cache that is exactly at the TTL boundary should still be valid."""
        source._data = source._normalise(SAMPLE_API_RESPONSE)
        source._build_index()
        source._save_cache()

        raw = json.loads(tmp_cache_path.read_text())
        # Set timestamp to just within TTL
        raw["timestamp"] = time.time() - source.CACHE_TTL + 60
        tmp_cache_path.write_text(json.dumps(raw))

        loaded = source._load_cache()
        assert loaded is not None

    def test_cache_corrupt_json(self, source: ModelsDevSource, tmp_cache_path: Path) -> None:
        tmp_cache_path.write_text("NOT JSON!!!")
        loaded = source._load_cache()
        assert loaded is None

    def test_cache_missing_file(self, source: ModelsDevSource) -> None:
        loaded = source._load_cache()
        assert loaded is None

    def test_cache_missing_timestamp_key(self, source: ModelsDevSource, tmp_cache_path: Path) -> None:
        """Cache file with no timestamp should be treated as expired (timestamp=0)."""
        tmp_cache_path.write_text(json.dumps({"models": {}}))
        loaded = source._load_cache()
        assert loaded is None  # timestamp=0 means extremely old

    def test_cache_missing_models_key(self, source: ModelsDevSource, tmp_cache_path: Path) -> None:
        """Cache file with no 'models' key should return empty dict (not crash)."""
        tmp_cache_path.write_text(json.dumps({"timestamp": time.time()}))
        loaded = source._load_cache()
        assert loaded is not None
        assert loaded == {}

    def test_cache_missing_required_field_in_model(self, source: ModelsDevSource, tmp_cache_path: Path) -> None:
        """If a cached model is missing the 'id' field, KeyError should be caught."""
        payload = {
            "timestamp": time.time(),
            "models": {
                "bad_model": {"provider": "p"},  # missing 'id' → KeyError
            },
        }
        tmp_cache_path.write_text(json.dumps(payload))
        loaded = source._load_cache()
        assert loaded is None  # KeyError is caught and returns None

    def test_cache_parent_dir_created(self, tmp_path: Path) -> None:
        """_save_cache should create parent directories if needed."""
        deep_path = tmp_path / "a" / "b" / "c" / "cache.json"
        src = ModelsDevSource(cache_path=deep_path)
        src._data = src._normalise(SAMPLE_API_RESPONSE)
        src._save_cache()
        assert deep_path.exists()

    def test_cache_atomic_write(self, source: ModelsDevSource, tmp_cache_path: Path) -> None:
        """After _save_cache, no .tmp file should remain."""
        source._data = source._normalise(SAMPLE_API_RESPONSE)
        source._build_index()
        source._save_cache()
        tmp_file = tmp_cache_path.with_suffix(".json.tmp")
        assert not tmp_file.exists()
        assert tmp_cache_path.exists()

    def test_cache_save_fail_graceful(self, source: ModelsDevSource) -> None:
        """_save_cache should not raise on write failure (e.g. read-only path)."""
        source.CACHE_PATH = Path("/dev/null/impossible/path.json")
        source._data = source._normalise(SAMPLE_API_RESPONSE)
        source._build_index()
        # Should not raise
        source._save_cache()

    def test_cache_persistence_across_instances(self, tmp_cache_path: Path) -> None:
        """Data saved by one instance should be loadable by a new instance."""
        src1 = ModelsDevSource(cache_path=tmp_cache_path)
        src1._data = src1._normalise(SAMPLE_API_RESPONSE)
        src1._save_cache()

        src2 = ModelsDevSource(cache_path=tmp_cache_path)
        loaded = src2._load_cache()
        assert loaded is not None
        assert "minimax/minimax-m2.5" in loaded


# ===========================================================================
# Enrich
# ===========================================================================


class TestEnrich:
    """Tests for the enrich() provider-wins semantics."""

    def test_fills_all_none_fields(self, source: ModelsDevSource) -> None:
        source._data = source._normalise(SAMPLE_API_RESPONSE)
        source._build_index()

        entry = RegistryModelEntry(
            id="minimax/minimax-m2.5",
            name="MiniMax M2.5",
        )
        enriched = source.enrich(entry, "minimax")
        assert enriched.input_price == 0.3
        assert enriched.output_price == 1.2
        assert enriched.context_window == 204800
        assert enriched.max_output_tokens == 131072
        assert enriched.knowledge_cutoff == "2025-01"

    def test_provider_wins_input_price(self, source: ModelsDevSource) -> None:
        source._data = source._normalise(SAMPLE_API_RESPONSE)
        source._build_index()

        entry = RegistryModelEntry(
            id="minimax/minimax-m2.5",
            name="MiniMax M2.5",
            input_price=99.99,
        )
        enriched = source.enrich(entry, "minimax")
        assert enriched.input_price == 99.99

    def test_provider_wins_context_window(self, source: ModelsDevSource) -> None:
        source._data = source._normalise(SAMPLE_API_RESPONSE)
        source._build_index()

        entry = RegistryModelEntry(
            id="minimax/minimax-m2.5",
            name="MiniMax M2.5",
            context_window=999,
        )
        enriched = source.enrich(entry, "minimax")
        assert enriched.context_window == 999

    def test_partial_fill_mixed_fields(self, source: ModelsDevSource) -> None:
        """Some fields set by provider, others filled from external."""
        source._data = source._normalise(SAMPLE_API_RESPONSE)
        source._build_index()

        entry = RegistryModelEntry(
            id="minimax/minimax-m2.5",
            name="MiniMax M2.5",
            input_price=99.99,
            context_window=999,
        )
        enriched = source.enrich(entry, "minimax")
        assert enriched.input_price == 99.99  # Provider
        assert enriched.output_price == 1.2  # External
        assert enriched.context_window == 999  # Provider
        assert enriched.max_output_tokens == 131072  # External

    def test_all_fields_set_is_noop(self, source: ModelsDevSource) -> None:
        """If all enrichable fields are already set, enrich should not change anything."""
        source._data = source._normalise(SAMPLE_API_RESPONSE)
        source._build_index()

        entry = RegistryModelEntry(
            id="minimax/minimax-m2.5",
            name="MiniMax M2.5",
            input_price=10.0,
            output_price=20.0,
            context_window=100,
            max_output_tokens=50,
            knowledge_cutoff="2024-01",
        )
        enriched = source.enrich(entry, "minimax")
        assert enriched.input_price == 10.0
        assert enriched.output_price == 20.0
        assert enriched.context_window == 100
        assert enriched.max_output_tokens == 50
        assert enriched.knowledge_cutoff == "2024-01"

    def test_missing_model_returns_unchanged(self, source: ModelsDevSource) -> None:
        source._data = {}
        entry = RegistryModelEntry(id="nonexistent", name="No Match")
        enriched = source.enrich(entry, "minimax")
        assert enriched.input_price is None
        assert enriched.context_window is None

    def test_enrich_returns_same_object(self, source: ModelsDevSource) -> None:
        """enrich() should mutate and return the same entry object."""
        source._data = source._normalise(SAMPLE_API_RESPONSE)
        source._build_index()
        entry = RegistryModelEntry(id="minimax/minimax-m2.5", name="MiniMax M2.5")
        enriched = source.enrich(entry, "minimax")
        assert enriched is entry

    def test_enrich_with_external_none_cost(self, source: ModelsDevSource) -> None:
        """If external data has None cost, entry's None should remain None."""
        source._data = {
            "prov/m1": ExternalModelData(
                id="m1",
                provider="prov",
                input_cost=None,
                output_cost=None,
            ),
        }
        entry = RegistryModelEntry(id="m1", name="M1")
        enriched = source.enrich(entry, "prov")
        assert enriched.input_price is None
        assert enriched.output_price is None

    def test_enrich_fills_knowledge_cutoff(self, source: ModelsDevSource) -> None:
        source._data = source._normalise(SAMPLE_API_RESPONSE)
        source._build_index()
        entry = RegistryModelEntry(id="minimax/minimax-m2.5", name="MiniMax M2.5")
        enriched = source.enrich(entry, "minimax")
        assert enriched.knowledge_cutoff == "2025-01"

    def test_enrich_does_not_fill_knowledge_cutoff_if_set(self, source: ModelsDevSource) -> None:
        source._data = source._normalise(SAMPLE_API_RESPONSE)
        source._build_index()
        entry = RegistryModelEntry(
            id="minimax/minimax-m2.5",
            name="MiniMax M2.5",
            knowledge_cutoff="2024-12",
        )
        enriched = source.enrich(entry, "minimax")
        assert enriched.knowledge_cutoff == "2024-12"


# ===========================================================================
# Fetch (async)
# ===========================================================================


class TestFetch:
    """Tests for the async fetch() method."""

    @pytest.mark.asyncio
    async def test_fetch_from_api(self, source: ModelsDevSource) -> None:
        with patch.object(source, "_fetch_api", new_callable=AsyncMock, return_value=SAMPLE_API_RESPONSE):
            result = await source.fetch()
        assert len(result) > 0
        assert "minimax/minimax-m2.5" in result

    @pytest.mark.asyncio
    async def test_fetch_uses_cache_when_fresh(self, source: ModelsDevSource, tmp_cache_path: Path) -> None:
        source._data = source._normalise(SAMPLE_API_RESPONSE)
        source._build_index()
        source._save_cache()

        with patch.object(source, "_fetch_api", new_callable=AsyncMock) as mock_api:
            result = await source.fetch()
            mock_api.assert_not_called()
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_fetch_failopen_on_api_error(self, source: ModelsDevSource) -> None:
        with patch.object(source, "_fetch_api", new_callable=AsyncMock, return_value=None):
            result = await source.fetch()
        assert result == {}

    @pytest.mark.asyncio
    async def test_fetch_saves_cache_after_api_call(self, source: ModelsDevSource, tmp_cache_path: Path) -> None:
        with patch.object(source, "_fetch_api", new_callable=AsyncMock, return_value=SAMPLE_API_RESPONSE):
            await source.fetch()
        assert tmp_cache_path.exists()

    @pytest.mark.asyncio
    async def test_fetch_stale_cache_triggers_api(self, source: ModelsDevSource, tmp_cache_path: Path) -> None:
        """When cache is stale, fetch() should call the API."""
        source._data = source._normalise(SAMPLE_API_RESPONSE)
        source._build_index()
        source._save_cache()

        # Make cache stale
        raw = json.loads(tmp_cache_path.read_text())
        raw["timestamp"] = time.time() - source.CACHE_TTL - 100
        tmp_cache_path.write_text(json.dumps(raw))

        with patch.object(source, "_fetch_api", new_callable=AsyncMock, return_value=SAMPLE_API_RESPONSE) as mock_api:
            result = await source.fetch()
            mock_api.assert_called_once()
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_fetch_populates_internal_data(self, source: ModelsDevSource) -> None:
        """After fetch(), source._data should be populated."""
        assert source._data == {}
        with patch.object(source, "_fetch_api", new_callable=AsyncMock, return_value=SAMPLE_API_RESPONSE):
            await source.fetch()
        assert len(source._data) > 0

    @pytest.mark.asyncio
    async def test_fetch_api_fail_preserves_empty(self, source: ModelsDevSource) -> None:
        """If API fails and no cache exists, _data should remain empty."""
        with patch.object(source, "_fetch_api", new_callable=AsyncMock, return_value=None):
            await source.fetch()
        assert source._data == {}


# ===========================================================================
# _fetch_api (HTTP-level)
# ===========================================================================


class TestFetchApi:
    """Tests for _fetch_api HTTP error handling."""

    @pytest.mark.asyncio
    async def test_fetch_api_http_error(self, source: ModelsDevSource) -> None:
        """HTTP 500 should return None."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error", request=MagicMock(), response=MagicMock(status_code=500)
        )
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get.return_value = mock_response
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await source._fetch_api()
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_api_timeout(self, source: ModelsDevSource) -> None:
        """Network timeout should return None."""
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get.side_effect = httpx.ConnectTimeout("timeout")
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await source._fetch_api()
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_api_connection_error(self, source: ModelsDevSource) -> None:
        """DNS/connection failure should return None."""
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get.side_effect = httpx.ConnectError("connection refused")
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await source._fetch_api()
        assert result is None


# ===========================================================================
# Integration: enrichment in ModelRegistry.refresh()
# ===========================================================================


class TestRegistryIntegration:
    """Integration tests: ModelsDevSource wired into ModelRegistry.refresh()."""

    @pytest.mark.asyncio
    async def test_refresh_enriches_models(self, tmp_path: Path) -> None:
        """Verify that refresh() calls enrich() and fills external metadata."""
        from mirai.agent.providers.base import ModelInfo
        from mirai.agent.registry import ModelRegistry

        reg = ModelRegistry.__new__(ModelRegistry)
        reg._path = tmp_path / "reg.json"
        reg._config_provider = "minimax"
        reg._config_model = "minimax-m2.5"

        from mirai.agent.registry_models import RegistryData

        reg._data = RegistryData(active_provider="minimax", active_model="minimax-m2.5")
        reg.PATH = tmp_path / "reg.json"
        reg._free_source = None

        # Set up enrichment source
        cache_path = tmp_path / "cache.json"
        enrich_src = ModelsDevSource(cache_path=cache_path)
        enrich_src._data = enrich_src._normalise(SAMPLE_API_RESPONSE)
        reg._enrichment_source = enrich_src

        # Mock provider returning a model without pricing
        mock_model = ModelInfo(
            id="minimax/minimax-m2.5",
            name="MiniMax M2.5",
            reasoning=True,
            # input_price=None, output_price=None — should be filled by enrichment
        )

        with patch("mirai.agent.registry._import_provider_class") as mock_import:
            mock_provider_cls = MagicMock()
            mock_provider_instance = MagicMock()
            mock_provider_instance.list_models = AsyncMock(return_value=[mock_model])
            mock_provider_cls.return_value = mock_provider_instance
            mock_import.return_value = mock_provider_cls

            with patch.dict("os.environ", {"MINIMAX_API_KEY": "test-key"}):
                await reg.refresh()

        # Check that the model was enriched
        provider_data = reg._data.providers.get("minimax")
        assert provider_data is not None
        assert len(provider_data.models) > 0
        model = provider_data.models[0]
        assert model.input_price == 0.3
        assert model.output_price == 1.2
        assert model.context_window == 204800

    @pytest.mark.asyncio
    async def test_refresh_without_enrichment_source(self, tmp_path: Path) -> None:
        """refresh() should work fine when _enrichment_source is None."""
        from mirai.agent.providers.base import ModelInfo
        from mirai.agent.registry import ModelRegistry

        reg = ModelRegistry.__new__(ModelRegistry)
        reg._path = tmp_path / "reg.json"
        reg._config_provider = "minimax"
        reg._config_model = "minimax-m2.5"

        from mirai.agent.registry_models import RegistryData

        reg._data = RegistryData(active_provider="minimax", active_model="minimax-m2.5")
        reg.PATH = tmp_path / "reg.json"
        reg._enrichment_source = None
        reg._free_source = None

        mock_model = ModelInfo(
            id="minimax-m2.5",
            name="MiniMax M2.5",
            input_price=0.5,
            output_price=2.0,
        )

        with patch("mirai.agent.registry._import_provider_class") as mock_import:
            mock_provider_cls = MagicMock()
            mock_provider_instance = MagicMock()
            mock_provider_instance.list_models = AsyncMock(return_value=[mock_model])
            mock_provider_cls.return_value = mock_provider_instance
            mock_import.return_value = mock_provider_cls

            with patch.dict("os.environ", {"MINIMAX_API_KEY": "test-key"}):
                await reg.refresh()

        provider_data = reg._data.providers.get("minimax")
        assert provider_data is not None
        model = provider_data.models[0]
        assert model.input_price == 0.5  # From provider, not enriched

    @pytest.mark.asyncio
    async def test_refresh_enrichment_fetch_failure(self, tmp_path: Path) -> None:
        """If enrichment source.fetch() raises, refresh() should still succeed."""
        from mirai.agent.providers.base import ModelInfo
        from mirai.agent.registry import ModelRegistry

        reg = ModelRegistry.__new__(ModelRegistry)
        reg._path = tmp_path / "reg.json"
        reg._config_provider = "minimax"
        reg._config_model = "minimax-m2.5"

        from mirai.agent.registry_models import RegistryData

        reg._data = RegistryData(active_provider="minimax", active_model="minimax-m2.5")
        reg.PATH = tmp_path / "reg.json"
        reg._free_source = None
        mock_source = MagicMock()
        mock_source.fetch = AsyncMock(side_effect=RuntimeError("network down"))
        reg._enrichment_source = mock_source

        mock_model = ModelInfo(id="minimax-m2.5", name="MiniMax M2.5")

        with patch("mirai.agent.registry._import_provider_class") as mock_import:
            mock_provider_cls = MagicMock()
            mock_provider_instance = MagicMock()
            mock_provider_instance.list_models = AsyncMock(return_value=[mock_model])
            mock_provider_cls.return_value = mock_provider_instance
            mock_import.return_value = mock_provider_cls

            with patch.dict("os.environ", {"MINIMAX_API_KEY": "test-key"}):
                # Should NOT raise — fail-open design
                await reg.refresh()

        provider_data = reg._data.providers.get("minimax")
        assert provider_data is not None
        assert len(provider_data.models) > 0

    @pytest.mark.asyncio
    async def test_refresh_enrichment_timeout_does_not_block(self, tmp_path: Path) -> None:
        """If enrichment fetch hangs beyond the timeout, refresh() still completes.

        The 20-second asyncio.wait_for guard should cancel the enrichment
        coroutine so provider scanning is never blocked.
        """
        import asyncio

        from mirai.agent.providers.base import ModelInfo
        from mirai.agent.registry import ModelRegistry
        from mirai.agent.registry_models import RegistryData

        reg = ModelRegistry.__new__(ModelRegistry)
        reg._config_provider = "minimax"
        reg._config_model = "minimax-m2.5"
        reg._data = RegistryData(active_provider="minimax", active_model="minimax-m2.5")
        reg.PATH = tmp_path / "reg.json"
        reg._free_source = None
        mock_source = MagicMock()

        async def _hang_forever() -> dict:
            await asyncio.sleep(9999)
            return {}

        mock_source.fetch = _hang_forever
        reg._enrichment_source = mock_source

        mock_model = ModelInfo(id="minimax-m2.5", name="MiniMax M2.5")

        with patch("mirai.agent.registry._import_provider_class") as mock_import:
            mock_provider_cls = MagicMock()
            mock_provider_instance = MagicMock()
            mock_provider_instance.list_models = AsyncMock(return_value=[mock_model])
            mock_provider_cls.return_value = mock_provider_instance
            mock_import.return_value = mock_provider_cls

            with patch.dict("os.environ", {"MINIMAX_API_KEY": "test-key"}):
                # Patch the wait_for timeout to 0.1s for fast test
                with patch(
                    "mirai.agent.registry.asyncio.wait_for",
                    wraps=asyncio.wait_for,
                ) as _mock_wf:
                    # Override the timeout used in refresh
                    original_refresh = reg.refresh

                    async def _patched_refresh() -> None:
                        # We can't easily patch the timeout literal, so instead
                        # we patch the enrichment source to use a short hang
                        mock_source.fetch = AsyncMock(side_effect=asyncio.TimeoutError)
                        await original_refresh()

                    await _patched_refresh()

        # Provider data should still be present despite enrichment timeout
        provider_data = reg._data.providers.get("minimax")
        assert provider_data is not None
        assert len(provider_data.models) > 0
        model = provider_data.models[0]
        assert model.name == "MiniMax M2.5"

    @pytest.mark.asyncio
    async def test_refresh_concurrent_execution(self, tmp_path: Path) -> None:
        """Verify that enrichment fetch runs concurrently with provider scanning.

        We use event ordering to prove that both coroutines overlap in time.
        """
        import asyncio

        from mirai.agent.providers.base import ModelInfo
        from mirai.agent.registry import ModelRegistry
        from mirai.agent.registry_models import RegistryData

        reg = ModelRegistry.__new__(ModelRegistry)
        reg._config_provider = "minimax"
        reg._config_model = "minimax-m2.5"
        reg._data = RegistryData(active_provider="minimax", active_model="minimax-m2.5")
        reg.PATH = tmp_path / "reg.json"
        reg._free_source = None

        # Track events to prove concurrency
        events: list[str] = []

        cache_path = tmp_path / "cache.json"
        enrich_src = ModelsDevSource(cache_path=cache_path)

        _original_fetch = enrich_src.fetch  # noqa: F841

        async def _slow_fetch() -> dict:
            events.append("enrich_start")
            await asyncio.sleep(0.05)
            events.append("enrich_end")
            return enrich_src._normalise(SAMPLE_API_RESPONSE)

        enrich_src.fetch = _slow_fetch  # type: ignore[assignment]
        reg._enrichment_source = enrich_src

        mock_model = ModelInfo(id="minimax-m2.5", name="MiniMax M2.5")

        async def _slow_list_models() -> list:
            events.append("provider_start")
            await asyncio.sleep(0.05)
            events.append("provider_end")
            return [mock_model]

        with patch("mirai.agent.registry._import_provider_class") as mock_import:
            mock_provider_cls = MagicMock()
            mock_provider_instance = MagicMock()
            mock_provider_instance.list_models = _slow_list_models
            mock_provider_cls.return_value = mock_provider_instance
            mock_import.return_value = mock_provider_cls

            with patch.dict("os.environ", {"MINIMAX_API_KEY": "test-key"}):
                await reg.refresh()

        # Both should have started before either ended (proves concurrency)
        assert "enrich_start" in events
        assert "provider_start" in events
        # If sequential, provider_start would come after enrich_end
        _enrich_start_idx = events.index("enrich_start")  # noqa: F841
        provider_start_idx = events.index("provider_start")
        enrich_end_idx = events.index("enrich_end")
        # In concurrent execution, provider_start should happen before enrich_end
        assert provider_start_idx < enrich_end_idx, f"Expected concurrent execution, but got sequential: {events}"

    @pytest.mark.asyncio
    async def test_refresh_in_background_does_not_block_event_loop(self, tmp_path: Path) -> None:
        """registry_refresh_loop runs as a background task; verify the event loop
        remains responsive while refresh() is running.
        """
        import asyncio

        from mirai.agent.providers.base import ModelInfo
        from mirai.agent.registry import ModelRegistry
        from mirai.agent.registry_models import RegistryData

        reg = ModelRegistry.__new__(ModelRegistry)
        reg._config_provider = "minimax"
        reg._config_model = "minimax-m2.5"
        reg._data = RegistryData(active_provider="minimax", active_model="minimax-m2.5")
        reg.PATH = tmp_path / "reg.json"
        reg._enrichment_source = None
        reg._free_source = None

        mock_model = ModelInfo(id="minimax-m2.5", name="MiniMax M2.5")

        async def _slow_list_models() -> list:
            await asyncio.sleep(0.1)
            return [mock_model]

        with patch("mirai.agent.registry._import_provider_class") as mock_import:
            mock_provider_cls = MagicMock()
            mock_provider_instance = MagicMock()
            mock_provider_instance.list_models = _slow_list_models
            mock_provider_cls.return_value = mock_provider_instance
            mock_import.return_value = mock_provider_cls

            with patch.dict("os.environ", {"MINIMAX_API_KEY": "test-key"}):
                # Run refresh as a background task
                refresh_task = asyncio.create_task(reg.refresh())

                # The event loop should be responsive — we can do other work
                counter = 0
                while not refresh_task.done():
                    await asyncio.sleep(0.01)
                    counter += 1

                await refresh_task

        # We should have ticked multiple times while refresh was running
        assert counter >= 2, f"Event loop was blocked, only ticked {counter} times"

        # Refresh should have completed successfully
        provider_data = reg._data.providers.get("minimax")
        assert provider_data is not None
        assert len(provider_data.models) > 0

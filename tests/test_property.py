"""Property-based tests using Hypothesis for config parsing and cron scheduling."""

from hypothesis import given, settings
from hypothesis import strategies as st

from mirai.config import (
    AgentConfig,
    DatabaseConfig,
    DreamerConfig,
    FeishuConfig,
    HeartbeatConfig,
    LLMConfig,
    RegistryConfig,
    ServerConfig,
    TracingConfig,
)
from mirai.cron import _dt_to_ms, _error_backoff_ms, _ms_to_dt, _now_ms, compute_next_run

# ---------------------------------------------------------------------------
# Config model properties
# ---------------------------------------------------------------------------


class TestLLMConfigProperties:
    """Property-based tests for LLMConfig parsing."""

    @given(
        provider=st.sampled_from(["antigravity", "anthropic", "openai", "minimax"]),
        model=st.text(min_size=1, max_size=50).filter(lambda s: s.strip()),
        max_tokens=st.integers(min_value=1, max_value=100000),
        max_retries=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=50)
    def test_llm_config_accepts_valid_values(self, provider, model, max_tokens, max_retries):
        """LLMConfig should accept any valid string/int combination."""
        config = LLMConfig(
            provider=provider,
            default_model=model,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )
        assert config.provider == provider
        assert config.default_model == model
        assert config.max_tokens == max_tokens
        assert config.max_retries == max_retries

    @given(
        api_key=st.one_of(st.none(), st.text(min_size=10, max_size=100)),
        base_url=st.one_of(st.none(), st.from_regex(r"https?://[a-z]+\.example\.com", fullmatch=True)),
    )
    @settings(max_examples=30)
    def test_optional_fields_accept_none_or_string(self, api_key, base_url):
        """api_key and base_url can be None or any string."""
        config = LLMConfig(api_key=api_key, base_url=base_url)
        assert config.api_key == api_key
        assert config.base_url == base_url


class TestFeishuConfigProperties:
    """Property tests for FeishuConfig."""

    @given(enabled=st.booleans())
    @settings(max_examples=10)
    def test_enabled_accepts_bool(self, enabled):
        config = FeishuConfig(enabled=enabled)
        assert config.enabled is enabled


class TestServerConfigProperties:
    """Property tests for ServerConfig."""

    @given(
        port=st.integers(min_value=1, max_value=65535),
        log_level=st.sampled_from(["DEBUG", "INFO", "WARNING", "ERROR"]),
        log_format=st.sampled_from(["console", "json"]),
    )
    @settings(max_examples=30)
    def test_server_config_port_range(self, port, log_level, log_format):
        config = ServerConfig(port=port, log_level=log_level, log_format=log_format)
        assert 1 <= config.port <= 65535
        assert config.log_level == log_level
        assert config.log_format == log_format


class TestConfigDefaults:
    """Property: default configs should always be constructable."""

    @given(st.just(None))
    def test_all_configs_have_defaults(self, _):
        """Every config model should be constructable with no arguments."""
        configs = [
            LLMConfig(),
            FeishuConfig(),
            HeartbeatConfig(),
            ServerConfig(),
            DatabaseConfig(),
            AgentConfig(),
            RegistryConfig(),
            DreamerConfig(),
            TracingConfig(),
        ]
        for config in configs:
            assert config is not None


# ---------------------------------------------------------------------------
# Cron scheduling properties
# ---------------------------------------------------------------------------


class TestComputeNextRunProperties:
    """Property-based tests for cron schedule computation."""

    @given(
        interval_ms=st.integers(min_value=1000, max_value=86400000),
        after_ms=st.integers(min_value=1000000000000, max_value=2000000000000),
    )
    @settings(max_examples=50)
    def test_every_schedule_is_monotonic(self, interval_ms, after_ms):
        """For 'every' schedule, next_run is always after_ms + interval_ms."""
        schedule = {"kind": "every", "everyMs": interval_ms}
        result = compute_next_run(schedule, after_ms=after_ms)
        assert result is not None
        assert result == after_ms + interval_ms
        assert result > after_ms

    @given(
        interval_ms=st.integers(min_value=1, max_value=1000000),
    )
    @settings(max_examples=20)
    def test_every_schedule_produces_future(self, interval_ms):
        """'every' schedule always produces a time in the future relative to base."""
        now = _now_ms()
        schedule = {"kind": "every", "everyMs": interval_ms}
        result = compute_next_run(schedule, after_ms=now)
        assert result is not None
        assert result > now

    @given(
        after_ms=st.integers(min_value=1000000000000, max_value=2000000000000),
    )
    def test_cron_every_minute_is_within_60s(self, after_ms):
        """'* * * * *' schedule runs within 60 seconds of any base time."""
        schedule = {"kind": "cron", "expr": "* * * * *"}
        result = compute_next_run(schedule, after_ms=after_ms)
        assert result is not None
        assert result > after_ms
        assert result - after_ms <= 60 * 1000  # at most 60s ahead

    def test_at_schedule_past_returns_none(self):
        """'at' schedule in the past returns None."""
        schedule = {"kind": "at", "at": "2020-01-01T00:00:00+00:00"}
        result = compute_next_run(schedule, after_ms=_now_ms())
        assert result is None

    def test_at_schedule_future_returns_timestamp(self):
        """'at' schedule in the future returns the timestamp."""
        schedule = {"kind": "at", "at": "2030-01-01T00:00:00+00:00"}
        result = compute_next_run(schedule, after_ms=_now_ms())
        assert result is not None
        assert result > _now_ms()

    def test_unknown_kind_returns_none(self):
        """Unknown schedule kind returns None."""
        schedule = {"kind": "unknown_kind"}
        result = compute_next_run(schedule, after_ms=_now_ms())
        assert result is None


class TestErrorBackoffProperties:
    """Property-based tests for error backoff calculation."""

    @given(errors=st.integers(min_value=0, max_value=100))
    @settings(max_examples=50)
    def test_backoff_is_non_negative(self, errors):
        """Backoff should always be non-negative."""
        result = _error_backoff_ms(errors)
        assert result >= 0

    @given(errors=st.integers(min_value=5, max_value=100))
    @settings(max_examples=20)
    def test_backoff_caps_at_max(self, errors):
        """Backoff is capped at the maximum value for high error counts."""
        result = _error_backoff_ms(errors)
        # The max in ERROR_BACKOFF_MS is 60 * 60_000 = 3_600_000 ms
        assert result <= 60 * 60_000


class TestTimeConversionProperties:
    """Property-based tests for time conversion round-trips."""

    @given(ms=st.integers(min_value=0, max_value=4102444800000))
    @settings(max_examples=50)
    def test_ms_to_dt_and_back(self, ms):
        """Round-trip: ms → datetime → ms should preserve value."""
        dt = _ms_to_dt(ms)
        back = _dt_to_ms(dt)
        assert abs(back - ms) < 1000  # within 1 second due to potential rounding

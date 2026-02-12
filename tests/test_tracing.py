"""Tests for OpenTelemetry tracing module (mirai/tracing.py).

Covers: setup_tracing(), get_tracer(), @traced() decorator with spans,
error recording, custom attributes, and default span naming.
"""

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from mirai.tracing import get_tracer, traced


class SpanCollector(SpanExporter):
    """Simple in-memory span collector for testing."""

    def __init__(self):
        self.spans = []

    def export(self, spans):
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass

    def get_finished_spans(self):
        return list(self.spans)


@pytest.fixture(autouse=True)
def _reset_tracer():
    """Reset global tracer state between tests."""
    import mirai.tracing as tracing_mod

    tracing_mod._tracer = None
    yield
    tracing_mod._tracer = None


def _make_collector():
    """Create a TracerProvider + SpanCollector, assign to mirai._tracer directly.

    We avoid trace.set_tracer_provider() because OTel only allows it once
    per process. Instead, we create a tracer from a fresh provider and
    inject it directly into mirai.tracing._tracer.
    """
    import mirai.tracing as tracing_mod

    collector = SpanCollector()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(collector))
    tracing_mod._tracer = provider.get_tracer("test")
    return collector


# ---------------------------------------------------------------------------
# setup_tracing
# ---------------------------------------------------------------------------


class TestSetupTracing:
    def test_setup_creates_tracer(self):
        from mirai.tracing import setup_tracing

        setup_tracing(service_name="test-service", console=True)
        tracer = get_tracer()
        assert tracer is not None
        with tracer.start_as_current_span("test") as span:
            assert span is not None

    def test_setup_console_mode(self, monkeypatch):
        from mirai.tracing import setup_tracing

        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        monkeypatch.delenv("OTEL_TRACES_CONSOLE", raising=False)
        setup_tracing(console=True)
        tracer = get_tracer()
        assert tracer is not None

    def test_setup_via_env_var(self, monkeypatch):
        from mirai.tracing import setup_tracing

        monkeypatch.setenv("OTEL_TRACES_CONSOLE", "1")
        setup_tracing()
        tracer = get_tracer()
        assert tracer is not None


# ---------------------------------------------------------------------------
# get_tracer
# ---------------------------------------------------------------------------


class TestGetTracer:
    def test_returns_noop_tracer_before_setup(self):
        """get_tracer() returns a usable tracer even before setup_tracing()."""
        tracer = get_tracer()
        assert tracer is not None
        with tracer.start_as_current_span("noop"):
            pass

    def test_returns_configured_tracer_after_setup(self):
        _make_collector()  # side-effect: sets mirai.tracing._tracer
        tracer = get_tracer()
        assert tracer is not None


# ---------------------------------------------------------------------------
# @traced decorator
# ---------------------------------------------------------------------------


class TestTracedDecorator:
    @pytest.mark.asyncio
    async def test_creates_span_with_custom_name(self):
        collector = _make_collector()

        @traced("my.custom.span")
        async def my_func():
            return 42

        result = await my_func()
        assert result == 42

        spans = collector.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "my.custom.span"

    @pytest.mark.asyncio
    async def test_default_span_name_from_qualname(self):
        collector = _make_collector()

        @traced()
        async def some_function():
            return "ok"

        await some_function()

        spans = collector.get_finished_spans()
        assert len(spans) == 1
        assert "some_function" in spans[0].name

    @pytest.mark.asyncio
    async def test_custom_attributes(self):
        collector = _make_collector()

        @traced("attr.test", attributes={"component": "test", "version": "1.0"})
        async def func_with_attrs():
            return True

        await func_with_attrs()

        spans = collector.get_finished_spans()
        assert len(spans) == 1
        attrs = dict(spans[0].attributes)
        assert attrs["component"] == "test"
        assert attrs["version"] == "1.0"

    @pytest.mark.asyncio
    async def test_records_error_on_exception(self):
        collector = _make_collector()

        @traced("error.test")
        async def failing_func():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            await failing_func()

        spans = collector.get_finished_spans()
        assert len(spans) == 1
        attrs = dict(spans[0].attributes)
        assert attrs["error"] is True
        assert "boom" in attrs["error.message"]

    @pytest.mark.asyncio
    async def test_preserves_function_metadata(self):
        @traced("meta.test")
        async def documented_func():
            """This is a docstring."""
            pass

        assert documented_func.__name__ == "documented_func"
        assert documented_func.__doc__ == "This is a docstring."

    @pytest.mark.asyncio
    async def test_nested_spans(self):
        collector = _make_collector()

        @traced("outer")
        async def outer():
            return await inner()

        @traced("inner")
        async def inner():
            return "nested"

        result = await outer()
        assert result == "nested"

        spans = collector.get_finished_spans()
        span_names = {s.name for s in spans}
        assert "outer" in span_names
        assert "inner" in span_names

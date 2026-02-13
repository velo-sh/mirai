"""OpenTelemetry tracing setup for Mirai.

Provides distributed tracing across the Agent pipeline:
  Agent.run() → provider.generate_response() → tool.execute()

In production, export spans via OTLP to Jaeger/Tempo/etc.
In development, spans are printed to the console.
"""

from __future__ import annotations

import atexit
import os
from functools import wraps
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

_tracer: trace.Tracer | None = None


def setup_tracing(service_name: str = "mirai", console: bool = False) -> None:
    """Initialize OpenTelemetry tracing.

    Args:
        service_name: Service name for the resource attribute.
        console: If True, export spans to console (dev mode).
                 If False, try OTLP exporter or fall back to no-op.
    """
    global _tracer

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    if console or os.getenv("OTEL_TRACES_CONSOLE", "").lower() in ("1", "true"):
        # SimpleSpanProcessor exports synchronously — avoids
        # "I/O operation on closed file" when stdout is closed before
        # BatchSpanProcessor's background thread flushes.
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    else:
        # Try OTLP exporter if endpoint is configured
        otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        if otlp_endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

                exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
                provider.add_span_processor(BatchSpanProcessor(exporter))
            except ImportError:
                # OTLP exporter not installed, fall back to console
                provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(service_name)

    # Ensure spans are flushed before process exit
    atexit.register(provider.shutdown)


# Suppress the harmless "Failed to detach context" error that occurs
# when async frameworks (granian, uvloop) dispatch callbacks in different
# contextvars.Context copies, causing ContextVar.reset() to fail.
# See: https://github.com/open-telemetry/opentelemetry-python/issues/2606
#
# We patch _RUNTIME_CONTEXT.detach at the lowest level, because the
# module-level detach() function catches exceptions and logs them via
# logger.exception(), producing noisy tracebacks even though the error
# is harmless.
def _install_context_detach_fix():
    from opentelemetry.context import _RUNTIME_CONTEXT

    _original = _RUNTIME_CONTEXT.__class__.detach

    def _safe_detach(self, token):
        try:
            _original(self, token)
        except ValueError:
            pass

    _RUNTIME_CONTEXT.__class__.detach = _safe_detach


_install_context_detach_fix()


def get_tracer() -> trace.Tracer:
    """Return the configured tracer (or a no-op tracer if not initialized)."""
    return _tracer or trace.get_tracer("mirai")


def traced(
    name: str | None = None,
    attributes: dict[str, Any] | None = None,
):
    """Decorator to wrap async functions with OTel spans.

    Args:
        name: Span name. Defaults to the function's qualified name.
        attributes: Static span attributes to set.

    Usage:
        @traced("agent.think")
        async def think(self, ...):
            ...

        @traced(attributes={"component": "provider"})
        async def generate_response(self, ...):
            ...
    """

    def decorator(func):
        span_name = name or f"{func.__module__}.{func.__qualname__}"

        @wraps(func)
        async def wrapper(*args, **kwargs):
            tracer = get_tracer()
            with tracer.start_as_current_span(span_name) as span:
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:
                    span.set_attribute("error", True)
                    span.set_attribute("error.message", str(exc))
                    raise

        return wrapper

    return decorator

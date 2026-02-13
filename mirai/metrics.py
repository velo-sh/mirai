"""Lightweight request-level metrics for the /health endpoint.

Thread-safe counters that track total requests, error count, and
cumulative latency.  No external dependencies.
"""

import threading
import time


class RequestMetrics:
    """Accumulates request counts, errors, and latency."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total: int = 0
        self._errors: int = 0
        self._total_latency: float = 0.0

    def record(self, latency: float, *, error: bool = False) -> None:
        """Record a completed request."""
        with self._lock:
            self._total += 1
            self._total_latency += latency
            if error:
                self._errors += 1

    def snapshot(self) -> dict[str, float]:
        """Return a point-in-time snapshot for the health endpoint."""
        with self._lock:
            total = self._total
            errors = self._errors
            avg = round(self._total_latency / total * 1000, 1) if total else 0.0
            rate = round(errors / total * 100, 1) if total else 0.0
        return {
            "total_requests": total,
            "avg_latency_ms": avg,
            "error_rate_pct": rate,
        }


# Module-level singleton
metrics = RequestMetrics()


class LatencyTimer:
    """Context manager that records latency on exit.

    Usage::

        with LatencyTimer():
            ...  # handle request
    """

    def __init__(self, *, metrics_instance: RequestMetrics | None = None) -> None:
        self._metrics = metrics_instance or metrics
        self._start: float = 0.0
        self._error = False

    def mark_error(self) -> None:
        """Call inside the block to flag this request as an error."""
        self._error = True

    def __enter__(self) -> "LatencyTimer":
        self._start = time.monotonic()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
        elapsed = time.monotonic() - self._start
        self._metrics.record(elapsed, error=self._error or exc_type is not None)

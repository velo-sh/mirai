"""Tests for mirai.metrics — RequestMetrics and LatencyTimer."""

import threading
import time

from mirai.metrics import LatencyTimer, RequestMetrics


class TestRequestMetrics:
    """Tests for the RequestMetrics accumulator."""

    def test_initial_snapshot_is_zero(self):
        m = RequestMetrics()
        snap = m.snapshot()
        assert snap["total_requests"] == 0
        assert snap["avg_latency_ms"] == 0.0
        assert snap["error_rate_pct"] == 0.0

    def test_record_single_success(self):
        m = RequestMetrics()
        m.record(0.1)
        snap = m.snapshot()
        assert snap["total_requests"] == 1
        assert snap["error_rate_pct"] == 0.0
        assert snap["avg_latency_ms"] == 100.0  # 0.1s = 100ms

    def test_record_single_error(self):
        m = RequestMetrics()
        m.record(0.05, error=True)
        snap = m.snapshot()
        assert snap["total_requests"] == 1
        assert snap["error_rate_pct"] == 100.0
        assert snap["avg_latency_ms"] == 50.0

    def test_record_multiple_mixed(self):
        m = RequestMetrics()
        m.record(0.1)  # success
        m.record(0.2, error=True)  # error
        m.record(0.3)  # success
        snap = m.snapshot()
        assert snap["total_requests"] == 3
        assert snap["error_rate_pct"] == round(1 / 3 * 100, 1)
        assert snap["avg_latency_ms"] == round(0.2 * 1000, 1)

    def test_thread_safety(self):
        """Multiple threads recording concurrently should not corrupt state."""
        m = RequestMetrics()
        n = 100

        def worker():
            for _ in range(n):
                m.record(0.001)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        snap = m.snapshot()
        assert snap["total_requests"] == 400


class TestLatencyTimer:
    """Tests for the LatencyTimer context manager."""

    def test_records_latency_on_exit(self):
        m = RequestMetrics()
        with LatencyTimer(metrics_instance=m):
            time.sleep(0.01)
        snap = m.snapshot()
        assert snap["total_requests"] == 1
        assert snap["avg_latency_ms"] >= 5  # at least 5ms
        assert snap["error_rate_pct"] == 0.0

    def test_mark_error_flags_request(self):
        m = RequestMetrics()
        with LatencyTimer(metrics_instance=m) as timer:
            timer.mark_error()
        snap = m.snapshot()
        assert snap["total_requests"] == 1
        assert snap["error_rate_pct"] == 100.0

    def test_exception_marks_error(self):
        m = RequestMetrics()
        try:
            with LatencyTimer(metrics_instance=m):
                raise ValueError("boom")
        except ValueError:
            pass
        snap = m.snapshot()
        assert snap["total_requests"] == 1
        assert snap["error_rate_pct"] == 100.0

    def test_uses_module_singleton_by_default(self):
        """When no metrics_instance is passed, uses the module singleton."""
        timer = LatencyTimer()
        from mirai.metrics import metrics

        assert timer._metrics is metrics

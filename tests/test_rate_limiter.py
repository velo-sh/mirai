"""Tests for rate limiter internals (main._check_rate_limit).

Covers: window expiry, concurrent IPs, edge cases, bucket filling.
"""

import time

import main as main_module


class TestCheckRateLimit:
    """Unit tests for the sliding window rate limiter."""

    def setup_method(self):
        main_module._rate_limits.clear()

    def teardown_method(self):
        main_module._rate_limits.clear()

    def test_first_request_allowed(self):
        assert main_module._check_rate_limit("192.168.1.1") is True

    def test_within_limit(self):
        """Multiple requests within the limit should all pass."""
        for _ in range(main_module._RATE_MAX - 1):
            assert main_module._check_rate_limit("192.168.1.1") is True

    def test_at_limit_blocked(self):
        """21st request should be blocked (limit=20)."""
        for _ in range(main_module._RATE_MAX):
            main_module._check_rate_limit("192.168.1.1")

        assert main_module._check_rate_limit("192.168.1.1") is False

    def test_different_ips_independent(self):
        """Rate limits are per-IP — filling one doesn't affect another."""
        for _ in range(main_module._RATE_MAX):
            main_module._check_rate_limit("10.0.0.1")

        assert main_module._check_rate_limit("10.0.0.1") is False
        assert main_module._check_rate_limit("10.0.0.2") is True

    def test_expired_timestamps_pruned(self):
        """Old timestamps outside the window should be purged."""
        # Add timestamps from 2 minutes ago (outside the 60s window)
        old_time = time.monotonic() - 120
        main_module._rate_limits["old_ip"] = [old_time] * main_module._RATE_MAX

        # Should pass because all old entries are purged
        assert main_module._check_rate_limit("old_ip") is True
        # Only the new timestamp should remain
        assert len(main_module._rate_limits["old_ip"]) == 1

    def test_mixed_old_and_new_timestamps(self):
        """Mix of expired and current timestamps."""
        now = time.monotonic()
        old = now - 120  # expired
        # 15 expired + 5 fresh = 5 active (under limit)
        main_module._rate_limits["mixed"] = [old] * 15 + [now] * 5

        assert main_module._check_rate_limit("mixed") is True

    def test_exactly_at_boundary(self):
        """Exactly at max should block the next request."""
        for i in range(main_module._RATE_MAX):
            result = main_module._check_rate_limit("boundary_ip")
            assert result is True, f"Request {i + 1} should be allowed"

        # This should be blocked
        assert main_module._check_rate_limit("boundary_ip") is False

    def test_unknown_ip_starts_fresh(self):
        """Unknown IP has no history and should be allowed."""
        assert "never_seen_ip" not in main_module._rate_limits
        assert main_module._check_rate_limit("never_seen_ip") is True

    def test_cleanup_reduces_memory(self):
        """After expiry window, old entries should be cleaned up."""
        old_time = time.monotonic() - 200
        main_module._rate_limits["cleanup"] = [old_time] * 100

        main_module._check_rate_limit("cleanup")
        # Only the new timestamp should remain
        assert len(main_module._rate_limits["cleanup"]) == 1

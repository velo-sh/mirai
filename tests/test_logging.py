"""Unit tests for mirai.logging — structlog setup and logger factory."""

import logging

from mirai.logging import _orjson_renderer, get_logger, setup_logging


class TestOrjsonRenderer:
    """Tests for the orjson JSON renderer."""

    def test_renders_dict_to_json_string(self):
        result = _orjson_renderer(None, "test", {"event": "hello", "count": 42})
        assert isinstance(result, str)
        assert '"event"' in result
        assert '"hello"' in result

    def test_handles_non_string_keys(self):
        """orjson OPT_NON_STR_KEYS should handle int keys."""
        result = _orjson_renderer(None, "test", {1: "value"})
        assert isinstance(result, str)


class TestSetupLogging:
    def setup_method(self):
        """Reset root logger level before each test."""
        logging.getLogger().setLevel(logging.DEBUG)

    def test_console_mode_runs_without_error(self):
        """setup_logging(json_output=False) should configure without errors."""
        setup_logging(json_output=False, level="DEBUG")
        log = get_logger("test.console")
        log.info("test_event", key="value")

    def test_json_mode_runs_without_error(self):
        """setup_logging(json_output=True) should configure without errors."""
        setup_logging(json_output=True, level="INFO")
        log = get_logger("test.json")
        log.info("test_event", key="value")

    def test_setup_does_not_raise(self):
        """setup_logging should not raise for any valid level."""
        for level in ("DEBUG", "INFO", "WARNING", "ERROR"):
            setup_logging(json_output=False, level=level)


class TestGetLogger:
    def test_returns_bound_logger(self):
        logger = get_logger("test.module")
        assert logger is not None
        # structlog loggers should have standard methods
        assert hasattr(logger, "info")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "error")
        assert hasattr(logger, "debug")

    def test_different_names_return_loggers(self):
        log1 = get_logger("module.a")
        log2 = get_logger("module.b")
        # Both should be usable
        log1.info("test_a")
        log2.info("test_b")

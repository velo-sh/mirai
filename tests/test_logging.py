"""Unit tests for mirai.logging — structlog setup and logger factory."""

import logging

from mirai.logging import get_logger, setup_logging


class TestSetupLogging:
    def test_console_mode(self):
        """setup_logging(json_output=False) should configure console rendering."""
        setup_logging(json_output=False, level="DEBUG")
        log = get_logger("test.console")
        # Logger should be usable without errors
        log.info("test_event", key="value")

    def test_json_mode(self):
        """setup_logging(json_output=True) should configure JSON rendering."""
        setup_logging(json_output=True, level="INFO")
        log = get_logger("test.json")
        log.info("test_event", key="value")

    def test_log_level_respected(self):
        """Log level should be applied to the root logger."""
        setup_logging(json_output=False, level="WARNING")
        root = logging.getLogger()
        assert root.level == logging.WARNING


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

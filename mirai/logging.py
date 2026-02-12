"""
Structured logging configuration for Mirai.

Uses structlog with orjson serialization for high-performance JSON output
in production, and colored console output for development.
"""

import sys
from typing import Any

import orjson
import structlog


def _orjson_renderer(logger: object, name: str, event_dict: dict[str, object]) -> str:
    """Render log events as JSON using orjson."""
    return orjson.dumps(event_dict, option=orjson.OPT_NON_STR_KEYS).decode("utf-8")


def setup_logging(*, json_output: bool = False, level: str = "INFO") -> None:
    """Configure structlog for the application.

    Args:
        json_output: If True, emit JSON lines (for production).
                     If False, emit colored console output (for development).
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR).
    """
    import logging

    # Set stdlib root logger level
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=getattr(logging, level.upper(), logging.INFO),
    )

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    renderer: Any
    if json_output:
        # Production: JSON lines via orjson
        renderer = _orjson_renderer
    else:
        # Development: colored console
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure the formatter for stdlib logging integration
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    # Apply to root handler
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.setFormatter(formatter)


def get_logger(name: str) -> Any:
    """Get a named structlog logger."""
    return structlog.get_logger(name)

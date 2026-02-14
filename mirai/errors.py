"""Mirai application exception hierarchy.

Provides structured, catchable exceptions to replace bare ``Exception``
and ``ValueError`` throughout the codebase.  All application errors
derive from :class:`MiraiError` so callers can use a single ``except``
clause when needed.
"""


class MiraiError(Exception):
    """Base class for all Mirai application errors."""


class ProviderError(MiraiError, ValueError):
    """An LLM provider failed (auth, rate-limit, model not found, etc.)."""


class StorageError(MiraiError):
    """DuckDB / SQLite operation failed."""


class ConfigError(MiraiError):
    """Invalid or missing configuration."""


class ToolError(MiraiError):
    """A tool execution failed."""


class ShutdownError(MiraiError):
    """Error during graceful shutdown."""

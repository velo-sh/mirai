"""
Mirai Configuration System.

Uses pydantic-settings for type-safe configuration with TOML file support
and environment variable overrides.

Resolution priority (highest wins):
  1. Environment variables (MIRAI_ prefix, e.g. MIRAI_LLM_DEFAULT_MODEL)
  2. TOML config file (~/.mirai/config.toml)
  3. Built-in defaults
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

# Default config file location
CONFIG_DIR = Path.home() / ".mirai"
CONFIG_PATH = CONFIG_DIR / "config.toml"


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    default_model: str = "gemini-3-flash"
    max_tokens: int = 4096
    max_retries: int = 3


class FeishuConfig(BaseModel):
    """Feishu/Lark integration configuration."""

    app_id: str | None = None
    app_secret: str | None = None
    webhook_url: str | None = None
    enabled: bool = True


class HeartbeatConfig(BaseModel):
    """Heartbeat (proactive pulse) configuration."""

    interval: float = 3600.0
    enabled: bool = True


class ServerConfig(BaseModel):
    """FastAPI server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"


class DatabaseConfig(BaseModel):
    """Database configuration."""

    sqlite_url: str = "sqlite+aiosqlite:///./mirai.db"
    duckdb_path: str = "mirai_hdd.duckdb"


class AgentConfig(BaseModel):
    """Agent identity configuration."""

    collaborator_id: str = "01AN4Z048W7N7DF3SQ5G16CYAJ"


class MiraiConfig(BaseSettings):
    """Root configuration for the Mirai system.

    Load order (highest priority wins):
    1. Environment variables (MIRAI_ prefix)
    2. TOML config file (~/.mirai/config.toml)
    3. Built-in defaults

    Legacy Feishu env vars (FEISHU_APP_ID, etc.) are also honored.
    """

    model_config = SettingsConfigDict(
        env_prefix="MIRAI_",
        env_nested_delimiter="__",
        toml_file=str(CONFIG_PATH) if CONFIG_PATH.exists() else None,
        extra="ignore",
    )

    llm: LLMConfig = LLMConfig()
    feishu: FeishuConfig = FeishuConfig()
    heartbeat: HeartbeatConfig = HeartbeatConfig()
    server: ServerConfig = ServerConfig()
    database: DatabaseConfig = DatabaseConfig()
    agent: AgentConfig = AgentConfig()

    def model_post_init(self, __context: Any) -> None:
        """Apply legacy Feishu env var overrides after normal loading."""
        legacy_map = {
            "app_id": "FEISHU_APP_ID",
            "app_secret": "FEISHU_APP_SECRET",
            "webhook_url": "FEISHU_WEBHOOK_URL",
        }
        for field_name, env_key in legacy_map.items():
            val = os.getenv(env_key)
            if val and not getattr(self.feishu, field_name):
                object.__setattr__(self.feishu, field_name, val)

    @classmethod
    def load(cls, config_path: Path | None = None) -> MiraiConfig:
        """Load configuration (backward-compatible entry point).

        Args:
            config_path: Optional override for TOML file location.
        """
        if config_path and config_path.exists():
            import tomllib

            with open(config_path, "rb") as f:
                toml_data = tomllib.load(f)
            # Build nested models from TOML sections, then overlay on defaults
            kwargs: dict[str, Any] = {}
            section_map = {
                "llm": LLMConfig,
                "feishu": FeishuConfig,
                "heartbeat": HeartbeatConfig,
                "server": ServerConfig,
                "database": DatabaseConfig,
                "agent": AgentConfig,
            }
            for key, model_cls in section_map.items():
                if key in toml_data:
                    kwargs[key] = model_cls(**toml_data[key])
            return cls(**kwargs)  # type: ignore[arg-type]
        return cls()

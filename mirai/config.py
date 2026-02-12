"""
Mirai Configuration System.

3-tier resolution: TOML config file → environment variable → default value.
Config file path: ~/.mirai/config.toml
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

# Default config file location (same directory as credentials)
CONFIG_DIR = Path.home() / ".mirai"
CONFIG_PATH = CONFIG_DIR / "config.toml"


@dataclass
class LLMConfig:
    """LLM provider configuration."""

    default_model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    max_retries: int = 3


@dataclass
class FeishuConfig:
    """Feishu/Lark integration configuration."""

    app_id: str | None = None
    app_secret: str | None = None
    webhook_url: str | None = None
    enabled: bool = True


@dataclass
class HeartbeatConfig:
    """Heartbeat (proactive pulse) configuration."""

    interval: float = 3600.0
    enabled: bool = True


@dataclass
class ServerConfig:
    """FastAPI server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class DatabaseConfig:
    """Database configuration."""

    sqlite_url: str = "sqlite+aiosqlite:///./mirai.db"
    duckdb_path: str = "mirai_hdd.duckdb"


@dataclass
class AgentConfig:
    """Agent identity configuration."""

    collaborator_id: str = "01AN4Z048W7N7DF3SQ5G16CYAJ"


@dataclass
class MiraiConfig:
    """Root configuration for the Mirai system.

    Load order (highest priority wins):
    1. Environment variables (MIRAI_LLM_DEFAULT_MODEL, FEISHU_APP_ID, etc.)
    2. TOML config file (~/.mirai/config.toml)
    3. Dataclass defaults
    """

    llm: LLMConfig = field(default_factory=LLMConfig)
    feishu: FeishuConfig = field(default_factory=FeishuConfig)
    heartbeat: HeartbeatConfig = field(default_factory=HeartbeatConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    @classmethod
    def load(cls, config_path: Path | None = None) -> MiraiConfig:
        """Load configuration with 3-tier resolution.

        Priority: env vars > TOML file > defaults.
        """
        path = config_path or CONFIG_PATH
        toml_data: dict[str, Any] = {}

        # Load TOML file if it exists
        if path.exists():
            with open(path, "rb") as f:
                toml_data = tomllib.load(f)

        # Build sub-configs with TOML + env var overrides
        config = cls(
            llm=_build_section(LLMConfig, toml_data.get("llm", {}), "MIRAI_LLM"),
            feishu=_build_feishu(toml_data.get("feishu", {})),
            heartbeat=_build_section(HeartbeatConfig, toml_data.get("heartbeat", {}), "MIRAI_HEARTBEAT"),
            server=_build_section(ServerConfig, toml_data.get("server", {}), "MIRAI_SERVER"),
            database=_build_section(DatabaseConfig, toml_data.get("database", {}), "MIRAI_DB"),
            agent=_build_section(AgentConfig, toml_data.get("agent", {}), "MIRAI_AGENT"),
        )

        return config

    def __str__(self) -> str:
        lines = ["MiraiConfig:"]
        for f in fields(self):
            section = getattr(self, f.name)
            lines.append(f"  [{f.name}]")
            for sf in fields(section):
                val = getattr(section, sf.name)
                # Mask secrets
                if "secret" in sf.name.lower() or "key" in sf.name.lower():
                    display = f"{str(val)[:8]}...{str(val)[-4:]}" if val else "None"
                else:
                    display = val
                lines.append(f"    {sf.name} = {display}")
        return "\n".join(lines)


def _coerce(value: Any, target_type: Any) -> Any:
    """Coerce a value to a target type."""
    if value is None:
        return None

    # Handle string type annotations
    if isinstance(target_type, str):
        if "bool" in target_type:
            if isinstance(value, str):
                return value.lower() in ("true", "1", "yes")
            return bool(value)
        if "int" in target_type:
            return int(value)
        if "float" in target_type:
            return float(value)
        return value

    if target_type is bool:
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes")
        return bool(value)

    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)

    return value


def _build_section(cls: type, toml_section: dict[str, Any], env_prefix: str) -> Any:
    """Build a config section from TOML data + env var overrides."""
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        env_key = f"{env_prefix}_{f.name.upper()}"
        env_val = os.getenv(env_key)
        if env_val is not None:
            # Env var wins
            kwargs[f.name] = _coerce(env_val, f.type)
        elif f.name in toml_section:
            # TOML file value
            kwargs[f.name] = _coerce(toml_section[f.name], f.type)
        # else: use dataclass default
    return cls(**kwargs)


def _build_feishu(toml_section: dict[str, Any]) -> FeishuConfig:
    """Build Feishu config with backward-compatible env var names.

    Supports both legacy names (FEISHU_APP_ID) and new names (MIRAI_FEISHU_APP_ID).
    """
    kwargs: dict[str, Any] = {}

    # Map of field -> (legacy_env, new_env)
    env_map = {
        "app_id": ("FEISHU_APP_ID", "MIRAI_FEISHU_APP_ID"),
        "app_secret": ("FEISHU_APP_SECRET", "MIRAI_FEISHU_APP_SECRET"),
        "webhook_url": ("FEISHU_WEBHOOK_URL", "MIRAI_FEISHU_WEBHOOK_URL"),
        "enabled": (None, "MIRAI_FEISHU_ENABLED"),
    }

    for f in fields(FeishuConfig):
        legacy_key, new_key = env_map.get(f.name, (None, f"MIRAI_FEISHU_{f.name.upper()}"))

        # Priority: new env > legacy env > TOML > default
        env_val = os.getenv(new_key) if new_key else None
        if env_val is None and legacy_key:
            env_val = os.getenv(legacy_key)

        if env_val is not None:
            kwargs[f.name] = _coerce(env_val, f.type)
        elif f.name in toml_section:
            kwargs[f.name] = _coerce(toml_section[f.name], f.type)

    return FeishuConfig(**kwargs)

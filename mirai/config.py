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
from typing import Any, Tuple, Type

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

# Default config file location
CONFIG_DIR = Path.home() / ".mirai"
CONFIG_PATH = CONFIG_DIR / "config.toml"


class LLMConfig(BaseModel):
    """LLM provider configuration.

    provider: Which backend to use. Supported values:
        - "antigravity" (Google Cloud Code Assist, default)
        - "anthropic"   (Direct Anthropic API)
        - "openai"      (OpenAI or any OpenAI-compatible endpoint)
    api_key: API key for the provider (reads from env if not set)
    base_url: Custom API endpoint (for OpenAI-compatible providers like DeepSeek)
    """

    provider: str = "antigravity"
    default_model: str = "gemini-3-flash"
    api_key: str | None = None
    base_url: str | None = None
    max_tokens: int = 4096
    max_retries: int = 3


class FeishuConfig(BaseModel):
    """Feishu/Lark integration configuration."""

    app_id: str | None = None
    app_secret: str | None = None
    webhook_url: str | None = None
    curator_chat_id: str | None = None
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


class RegistryConfig(BaseModel):
    """Model registry configuration."""

    refresh_interval: int = 300  # seconds between registry refreshes
    fallback_models: list[str] = Field(
        default_factory=list,
        description="Ordered list of fallback model IDs to try when the primary model fails.",
    )


class TomlSource(PydanticBaseSettingsSource):
    """Custom settings source to load from TOML file using standard lib (py3.11+)."""

    def get_field_value(
        self, field: Any, field_name: str
    ) -> Tuple[Any, str, bool]:
        # Not used because we return full dict in __call__
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        # Check for override in init_kwargs or default path
        # Note: We can't easily access init_kwargs for config_path here without hacks.
        # So we stick to a global/default PATH for implicit loading.
        # For explicit loading, users should presumably use init args, 
        # but our load() method handles explicit paths by passing them as init args 
        # (which overrides this source anyway).
        # So this source is mainly for the "implicit default file" case.
        
        path = CONFIG_PATH
        if path.exists():
            try:
                import tomllib
                with open(path, "rb") as f:
                    return tomllib.load(f)
            except Exception:
                pass
        return {}


class MiraiConfig(BaseSettings):
    """Root configuration for the Mirai system.

    Load order (highest priority wins):
    1. Init arguments
    2. Environment variables (MIRAI_ prefix)
    3. TOML config file (~/.mirai/config.toml)
    4. Built-in defaults

    Legacy Feishu env vars (FEISHU_APP_ID, etc.) are also honored.
    """

    model_config = SettingsConfigDict(
        env_prefix="MIRAI_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    llm: LLMConfig = Field(default_factory=LLMConfig)
    feishu: FeishuConfig = Field(default_factory=FeishuConfig)
    heartbeat: HeartbeatConfig = Field(default_factory=HeartbeatConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    registry: RegistryConfig = Field(default_factory=RegistryConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            TomlSource(settings_cls),
            file_secret_settings,
        )

    def model_post_init(self, __context: Any) -> None:
        """Apply legacy Feishu env var overrides after normal loading."""
        legacy_map = {
            "app_id": "FEISHU_APP_ID",
            "app_secret": "FEISHU_APP_SECRET",
            "webhook_url": "FEISHU_WEBHOOK_URL",
            "curator_chat_id": "FEISHU_CURATOR_CHAT_ID",
        }
        for field_name, env_key in legacy_map.items():
            val = os.getenv(env_key)
            if val and not getattr(self.feishu, field_name):
                object.__setattr__(self.feishu, field_name, val)

    @classmethod
    def load(cls, config_path: Path | None = None) -> MiraiConfig:
        """Load configuration.

        Args:
            config_path: Override for TOML file location. 
                         If provided, it explicitly loads it and passes as init args
                         (winning over env vars due to init args priority).
                         To respect env vars while loading a custom file, 
                         we'd need a dynamic source, but for now this matches legacy behavior
                         for explicit file loading.
                         
                         For the default path (~/.mirai/config.toml), simple cls() 
                         uses TomlSource with correct precedence (Env > TOML).
        """
        if config_path and config_path.exists():
            import tomllib
            with open(config_path, "rb") as f:
                toml_data = tomllib.load(f)
            # Flatten/map manually if needed or just pass dict
            # We must map sections to model classes if passing as kwargs? 
            # No, pydantic handles dict->model conversion for nested fields.
            return cls(**toml_data)
            
        return cls()

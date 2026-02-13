"""Unit tests for mirai.config — pydantic-settings configuration system."""

import os
from mirai.config import (
    AgentConfig,
    DatabaseConfig,
    FeishuConfig,
    HeartbeatConfig,
    LLMConfig,
    MiraiConfig,
    ServerConfig,
)

# ---------------------------------------------------------------------------
# Sub-config defaults
# ---------------------------------------------------------------------------


class TestConfigDefaults:
    def test_llm_defaults(self):
        cfg = LLMConfig()
        assert cfg.default_model == "gemini-3-flash"
        assert cfg.max_tokens == 4096
        assert cfg.max_retries == 3

    def test_feishu_defaults(self):
        cfg = FeishuConfig()
        assert cfg.app_id is None
        assert cfg.app_secret is None
        assert cfg.webhook_url is None
        assert cfg.enabled is True

    def test_heartbeat_defaults(self):
        cfg = HeartbeatConfig()
        assert cfg.interval == 3600.0
        assert cfg.enabled is True

    def test_server_defaults(self):
        cfg = ServerConfig()
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 8000
        assert cfg.log_level == "INFO"

    def test_database_defaults(self):
        cfg = DatabaseConfig()
        assert "sqlite" in cfg.sqlite_url
        assert cfg.duckdb_path == "mirai_hdd.duckdb"

    def test_agent_defaults(self):
        cfg = AgentConfig()
        assert len(cfg.collaborator_id) > 0


# ---------------------------------------------------------------------------
# MiraiConfig.load() — defaults (no TOML, no env)
# ---------------------------------------------------------------------------


class TestMiraiConfigLoad:
    def test_load_returns_defaults(self, monkeypatch):
        """With no TOML file and no env vars, all defaults should apply."""
        # Ensure no MIRAI_ env vars leak
        for key in list(os.environ):
            if key.startswith("MIRAI_"):
                monkeypatch.delenv(key, raising=False)

        cfg = MiraiConfig.load()
        assert cfg.llm.default_model == "gemini-3-flash"
        assert cfg.server.port == 8000
        assert cfg.heartbeat.enabled is True

    def test_load_from_toml(self, tmp_toml):
        """Values from TOML should override defaults."""
        toml_path = tmp_toml(
            """
[llm]
default_model = "test-model-from-toml"
max_tokens = 2048

[server]
port = 9999
"""
        )
        cfg = MiraiConfig.load(config_path=toml_path)
        assert cfg.llm.default_model == "test-model-from-toml"
        assert cfg.llm.max_tokens == 2048
        assert cfg.server.port == 9999
        # Unspecified fields keep defaults
        assert cfg.heartbeat.enabled is True

    def test_env_overrides_defaults(self, monkeypatch):
        """MIRAI_ env vars should override defaults."""
        monkeypatch.setenv("MIRAI_LLM__DEFAULT_MODEL", "env-model")
        monkeypatch.setenv("MIRAI_SERVER__PORT", "7777")

        cfg = MiraiConfig.load()
        assert cfg.llm.default_model == "env-model"
        assert cfg.server.port == 7777


# ---------------------------------------------------------------------------
# Legacy Feishu env var compat
# ---------------------------------------------------------------------------


class TestFeishuLegacyEnvVars:
    def test_legacy_feishu_app_id(self, monkeypatch):
        monkeypatch.setenv("FEISHU_APP_ID", "legacy_app_id")
        cfg = MiraiConfig.load()
        assert cfg.feishu.app_id == "legacy_app_id"

    def test_legacy_feishu_app_secret(self, monkeypatch):
        monkeypatch.setenv("FEISHU_APP_SECRET", "legacy_secret")
        cfg = MiraiConfig.load()
        assert cfg.feishu.app_secret == "legacy_secret"

    def test_legacy_feishu_webhook_url(self, monkeypatch):
        monkeypatch.setenv("FEISHU_WEBHOOK_URL", "https://example.com/hook")
        cfg = MiraiConfig.load()
        assert cfg.feishu.webhook_url == "https://example.com/hook"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestConfigEdgeCases:
    def test_nonexistent_toml_path_uses_defaults(self, tmp_path):
        """If config_path doesn't exist, defaults should be used."""
        fake_path = tmp_path / "nonexistent.toml"
        cfg = MiraiConfig.load(config_path=fake_path)
        # Defaults to gemini-3-flash now
        assert cfg.llm.default_model == "gemini-3-flash"

    def test_empty_toml_uses_defaults(self, tmp_toml):
        """An empty TOML file should still work with all defaults."""
        toml_path = tmp_toml("")
        cfg = MiraiConfig.load(config_path=toml_path)
        assert cfg.server.port == 8000
        assert cfg.llm.max_tokens == 4096

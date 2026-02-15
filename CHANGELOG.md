# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-02-15

### Added
- Core agent loop with multi-provider LLM support (Anthropic, OpenAI, MiniMax).
- Cron scheduling system with JSON5 config, hot-reload, and startup recovery.
- 10 built-in tools: echo, shell, editor, workspace, git, system, model, config, cron, IM.
- Feishu (Lark) IM integration with WebSocket event handling and image processing.
- DuckDB storage with versioned schema migration system.
- Pydantic-settings configuration with TOML + environment variable overlay.
- Model registry with live refresh and external enrichment (models.dev, free providers).
- AgentDreamer background reflection service.
- Heartbeat monitoring with IM notifications.
- OpenTelemetry tracing integration.
- OAuth2 authentication for Antigravity API.
- Protocol-based provider abstraction (`ProviderProtocol`).

### Infrastructure
- GitHub Actions CI pipeline (ruff lint, ruff format, mypy, pytest + coverage).
- Pre-commit hooks (ruff, mypy, pytest-collect).
- 789 tests with 78% coverage.
- 5 RFC design documents (`docs/rfcs/0001`–`0005`).

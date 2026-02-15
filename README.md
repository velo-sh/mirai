# Mirai — AI Collaborator & Co-creation Platform

> **Mirai**: Empowering AI Collaborators to work as true partners in co-creation.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    FastAPI Server                     │
│                   (granian ASGI)                      │
├──────────┬───────────┬──────────────┬────────────────┤
│  Agent   │   Cron    │   Feishu IM  │  REST API      │
│  Loop    │ Scheduler │  Receiver    │  Endpoints     │
├──────────┴───────────┴──────────────┴────────────────┤
│                  Tool Registry                        │
│  echo · shell · editor · git · system · model · ...   │
├──────────────────────────────────────────────────────┤
│              Provider Abstraction Layer               │
│  Anthropic · OpenAI · MiniMax · Antigravity · Free    │
├──────────┬───────────────────────────┬───────────────┤
│  DuckDB  │   LanceDB (Vector)       │  SQLModel     │
│  Storage │   Embeddings             │  Sessions     │
└──────────┴───────────────────────────┴───────────────┘
```

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://astral.sh/uv/) package manager

### Quick Start

```bash
# Clone and setup
git clone <repo-url> && cd mirai
uv sync --dev
uv run pre-commit install

# Configure
cp .env.example .env   # Edit with your API keys

# Run
uv run main.py
```

### Docker

```bash
docker compose up --build
```

## Development

> **All commands must be run via `uv run`** to ensure the correct virtual
> environment and dependencies are used.

```bash
# Run the application
uv run main.py

# Run tests
uv run pytest tests/ -q

# Run tests with coverage
uv run pytest tests/ --cov=mirai --cov-report=term-missing

# Run pre-commit hooks
uv run pre-commit run --all-files

# Security scan
uv run bandit -r mirai/ -c pyproject.toml -q
```

## Project Structure

```
mirai/
├── agent/              # Core agent loop, tools, providers
│   ├── agent_loop.py   # Main reasoning loop
│   ├── tools/          # 10 built-in tools (shell, editor, git, ...)
│   ├── providers/      # LLM provider implementations
│   └── im/             # Feishu (Lark) IM integration
├── auth/               # OAuth2 authentication
├── db/                 # DuckDB storage + schema migrations
├── cron.py             # Cron scheduling system
├── config.py           # Pydantic-settings configuration
└── bootstrap.py        # Application initialization
tests/                  # 789 tests (78% coverage)
docs/rfcs/              # Architecture decision records
```

## Key Design Decisions

| Decision | Document |
|---|---|
| Biological Memory Architecture | [RFC 0001](docs/rfcs/0001-biological-memory-architecture.md) |
| Reasoning Loop | [RFC 0002](docs/rfcs/0002-reasoning-loop.md) |
| Proactive Autonomy | [RFC 0003](docs/rfcs/0003-proactive-autonomy.md) |
| Cron Service | [RFC 0004](docs/rfcs/0004-cron-service.md) |
| Cron JSON5 + Git | [RFC 0005](docs/rfcs/0005-cron-json5-git.md) |

## CI Pipeline

| Step | Tool |
|---|---|
| Lint + Format | Ruff |
| Type Check | mypy |
| Security Scan | Bandit |
| Tests + Coverage | pytest + pytest-cov |
| Dependency Updates | Dependabot |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

This project is licensed under the terms of the LICENSE file.

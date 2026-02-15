# Contributing to Mirai

Thank you for your interest in contributing to Mirai!

## Development Setup

```bash
# 1. Clone the repository
git clone <repo-url> && cd mirai

# 2. Install uv (if not already)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Install dependencies
uv sync --dev

# 4. Install pre-commit hooks
uv run pre-commit install

# 5. Run the app
uv run main.py
```

## Code Style

- **Formatter / Linter**: [Ruff](https://docs.astral.sh/ruff/) — enforced via pre-commit hooks
- **Type Checker**: [mypy](https://mypy-lang.org/) in strict mode
- **Line Length**: 120 characters
- **Python Version**: 3.11+

All formatting and lint issues are auto-fixed on commit. If pre-commit hooks fail,
stage the auto-fixed files and commit again.

## Testing

```bash
# Run the full test suite
uv run pytest tests/ -q

# Run with coverage report
uv run pytest tests/ --cov=mirai --cov-report=term-missing

# Run a specific test file
uv run pytest tests/test_cron.py -v
```

### Test Conventions

- Place tests in `tests/` with the naming pattern `test_<module>.py`.
- Use `pytest.mark.asyncio` for all async tests (asyncio_mode is `strict`).
- Mock external services; never make real network calls in CI.

## Pull Request Workflow

1. Create a feature branch from `main`.
2. Make your changes. Keep commits focused and descriptive.
3. Ensure all checks pass locally:
   ```bash
   uv run pre-commit run --all-files
   uv run pytest tests/ -q
   ```
4. Open a Pull Request against `main`.
5. CI will automatically run lint, type-check, and tests.

## Commit Message Format

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new cron scheduling feature
fix: resolve token refresh race condition
refactor: extract CronStore from CronScheduler
docs: update README with architecture overview
ci: add Bandit security scanning
```

## Architecture Overview

See `docs/rfcs/` for design decisions and architectural context:

- `0001` — Biological Memory Architecture
- `0002` — Reasoning Loop
- `0003` — Proactive Autonomy
- `0004` — Cron Service
- `0005` — Cron JSON5 + Git Integration

## Reporting Issues

Open an issue with a clear description, steps to reproduce, and expected vs actual behavior.
For security vulnerabilities, see [SECURITY.md](SECURITY.md).

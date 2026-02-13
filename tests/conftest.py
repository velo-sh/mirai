"""Shared test fixtures for Mirai test suite."""

import pytest


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch):
    """Remove real credentials from env to prevent tests from hitting live APIs."""
    for key in ("ANTHROPIC_API_KEY", "FEISHU_APP_ID", "FEISHU_APP_SECRET"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def tmp_toml(tmp_path):
    """Create a temporary TOML config file and return its Path."""

    def _write(content: str):
        p = tmp_path / "config.toml"
        p.write_text(content)
        return p

    return _write


@pytest.fixture
def tmp_db_paths(tmp_path):
    """Return isolated database paths for tests."""
    return {
        "sqlite_url": f"sqlite+aiosqlite:///{tmp_path / 'test.db'}",
        "duckdb_path": str(tmp_path / "test.duckdb"),
    }


@pytest.fixture
def duckdb_storage():
    """Create an in-memory DuckDBStorage — no file lock conflicts."""
    from mirai.db.duck import DuckDBStorage

    storage = DuckDBStorage(db_path=":memory:")
    yield storage
    storage.close()

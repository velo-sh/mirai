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


@pytest.fixture
def duck_path_storage(tmp_path):
    """Create a DuckDBStorage at a temporary on-disk path.

    Returns (storage, db_path) so tests can verify persistence.
    """
    from mirai.db.duck import DuckDBStorage

    db_path = str(tmp_path / "test_l3.duckdb")
    storage = DuckDBStorage(db_path=db_path)
    yield storage, db_path
    storage.close()


@pytest.fixture
def mock_agent(duckdb_storage):
    """Create an AgentLoop with MockProvider and in-memory DuckDB.

    Injects storage via the ``l3_storage`` parameter so no monkey-patching
    of ``DuckDBStorage.__init__`` is needed.
    """
    from mirai.agent.agent_loop import AgentLoop
    from mirai.agent.providers import MockProvider
    from mirai.agent.tools.echo import EchoTool

    agent = AgentLoop(
        provider=MockProvider(),
        tools=[EchoTool()],
        collaborator_id="test-agent",
        l3_storage=duckdb_storage,
    )
    agent.name = "Mira"
    agent.role = "collaborator"
    agent.base_system_prompt = "You are Mira, a helpful collaborator."
    agent.soul_content = ""
    return agent


@pytest.fixture
def mock_registry(tmp_path):
    """Create a ModelRegistry for testing without disk I/O.

    Uses ``ModelRegistry.for_testing()`` with defaults. Override via
    ``ModelRegistry.for_testing(data=...)`` if you need custom data.
    """
    from mirai.agent.registry import ModelRegistry

    return ModelRegistry.for_testing(path=tmp_path / "registry.json")

import asyncio
import os
import threading

import pytest

from mirai.agent.agent_loop import AgentLoop
from mirai.agent.providers import MockProvider
from mirai.agent.tools.editor import EditorTool
from mirai.agent.tools.shell import ShellTool
from mirai.db.session import init_db


@pytest.mark.asyncio
async def test_executive_multi_step_workflow(monkeypatch):
    """
    QA Integration Test: Verify the 'Search -> Thinking -> Write' multi-step workflow.
    """
    # Use temporary DBs to avoid locks
    test_sqlite = "tests/data/test_l1.db"
    test_duck = "tests/data/test_l3.duckdb"

    if os.path.exists(test_sqlite):
        os.remove(test_sqlite)
    if os.path.exists(test_duck):
        os.remove(test_duck)

    # Mock the DB paths
    monkeypatch.setenv("SQLITE_DB_URL", f"sqlite+aiosqlite:///{test_sqlite}")

    from mirai.db.duck import DuckDBStorage

    # We need to ensure DuckDBStorage uses our test path
    def mock_duck_init(self, db_path=test_duck):
        self.db_path = db_path
        import duckdb

        self.conn = duckdb.connect(db_path)
        self._lock = threading.Lock()
        self._init_schema()

    monkeypatch.setattr(DuckDBStorage, "__init__", mock_duck_init)

    await init_db()

    # Setup Agent with Executive Tools
    provider = MockProvider()
    collaborator_id = "01AN4Z048W7N7DF3SQ5G16CYAJ"
    tools = [ShellTool(), EditorTool()]
    agent = await AgentLoop.create(provider, tools, collaborator_id)

    # 2. Add specific mock behavior for this workflow
    # We want to see: shell_tool (ls) -> editor_tool (write)
    # The current MockProvider keyword matching needs to be robust.

    # We will use a targeted message that triggers the mock logic
    query = "Find the SOUL file and write a summary to 'soul_summary.txt'."

    # 3. Process the run
    # Note: MockProvider needs to be smart enough to return 'stop_reason=tool_use' twice.
    # I'll update MockProvider first to ensure it supports sequential tool use in tests.

    await agent.run(query)

    # 4. Verifications
    assert os.path.exists("soul_summary.txt"), "Executive chain failed: soul_summary.txt was not created."

    with open("soul_summary.txt") as f:
        content = f.read()
        assert "summary" in content.lower(), "Summary content is missing or incorrect."

    # Clean up
    if os.path.exists("soul_summary.txt"):
        os.remove("soul_summary.txt")

    print("\n[QA] Executive Multi-Step Integration Test Passed: Search-and-Write flow verified.")


if __name__ == "__main__":
    asyncio.run(test_executive_multi_step_workflow())

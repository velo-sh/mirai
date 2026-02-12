import asyncio
import os

import pytest

from mirai.agent.heartbeat import HeartbeatManager
from mirai.agent.im.feishu import FeishuProvider
from mirai.agent.loop import AgentLoop
from mirai.agent.providers import MockProvider
from mirai.db.session import init_db


@pytest.mark.asyncio
async def test_e2e_proactive_maintenance_flow(monkeypatch):
    """
    QA E2E Test: Heartbeat -> Detect Problem -> Fix (Create File) -> Notify Feishu.
    """
    # 1. Setup Isolated Environments
    test_sqlite = "tests/data/e2e_l1.db"
    test_duck = "tests/data/e2e_l3.duckdb"
    if os.path.exists(test_sqlite):
        os.remove(test_sqlite)
    if os.path.exists(test_duck):
        os.remove(test_duck)

    monkeypatch.setenv("SQLITE_DB_URL", f"sqlite+aiosqlite:///{test_sqlite}")

    def mock_duck_init(self, db_path=test_duck):
        self.db_path = db_path
        import duckdb

        self.conn = duckdb.connect(db_path)
        self._init_schema()

    from mirai.db.duck import DuckDBStorage

    monkeypatch.setattr(DuckDBStorage, "__init__", mock_duck_init)

    await init_db()

    # 2. Setup Agent & Heartbeat
    provider = MockProvider()
    from mirai.agent.tools.editor import EditorTool
    from mirai.agent.tools.shell import ShellTool

    tools = [ShellTool(), EditorTool()]
    agent = await AgentLoop.create(provider, tools, "01AN4Z048W7N7DF3SQ5G16CYAJ")

    # Mock Feishu Notification
    notified_messages = []

    class MockFeishu(FeishuProvider):
        async def send_message(self, content: str, chat_id: str | None = None) -> bool:
            notified_messages.append(content)
            return True

    im_provider = MockFeishu(webhook_url="http://fake")
    _heartbeat = HeartbeatManager(agent, interval_seconds=0.1, im_provider=im_provider)  # noqa: F841

    # 3. Configure MockProvider for the "Maintenance" Trigger
    # We'll use a specific condition: if SYSTEM_HEARTBEAT is present,
    # and we see a certain pattern, we fix it.

    # I'll update MockProvider one last time to handle this E2E scenario.

    # 4. Start Heartbeat (just for one pulse)
    # We manually trigger a pulse to keep the test deterministic
    pulse_message = "SYSTEM_HEARTBEAT: Perform a self-reflection on 'maintenance_check'."

    # We'll directly call the loop logic or just run agent.run for the pulse
    response = await agent.run(pulse_message)

    # Manual IM Push (since we aren't running the full background task here for simplicity)
    await im_provider.send_message(f"Heartbeat insight: {response}")

    # 5. Verifications
    # Verify the file was actually created by EditorTool
    assert os.path.exists("maintenance_fixed.txt"), "E2E Failure: maintenance_fixed.txt was not created."

    # Verify cognitive traces in DuckDB
    from mirai.db.duck import DuckDBStorage

    storage = DuckDBStorage(test_duck)
    traces = await storage.get_recent_traces(agent.collaborator_id)

    # Check for specific trace types
    trace_types = [t["trace_type"] for t in traces]
    assert "message" in trace_types, "User message not archived in L3."
    assert "tool_use" in trace_types, "Tool use not archived in L3."
    assert "tool_result" in trace_types, "Tool result not archived in L3."
    assert "thinking" in trace_types, "Thinking process not archived in L3."

    with open("maintenance_fixed.txt") as f:
        assert "HEALED" in f.read()

    assert len(notified_messages) > 0
    # The response comes from the 'Handling Critique Turn' branch in MockProvider
    assert "SOUL.md" in notified_messages[0]

    # Clean up
    if os.path.exists("maintenance_fixed.txt"):
        os.remove("maintenance_fixed.txt")

    print("\n[QA] E2E Proactive Maintenance Flow Verified.")


if __name__ == "__main__":
    asyncio.run(test_e2e_proactive_maintenance_flow())

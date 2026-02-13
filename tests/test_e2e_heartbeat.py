import asyncio

import pytest

from mirai.agent.heartbeat import HeartbeatManager
from mirai.agent.loop import AgentLoop
from mirai.agent.providers import MockProvider
from mirai.agent.tools.workspace import WorkspaceTool
from mirai.db.session import init_db


@pytest.mark.asyncio
async def test_heartbeat_proactive_insight_flow(duckdb_storage):
    """
    QA E2E Test: Verify the full flow from Heartbeat Pulse to L3 Trace.
    """
    await init_db()

    # 1. Setup Agent and Heartbeat
    provider = MockProvider()
    collaborator_id = "01AN4Z048W7N7DF3SQ5G16CYAJ"
    tools = [WorkspaceTool()]
    agent = await AgentLoop.create(provider, tools, collaborator_id, l3_storage=duckdb_storage)
    _hb = HeartbeatManager(agent, interval_seconds=1)  # noqa: F841

    # 2. Trigger the Heartbeat
    # We'll run the pulse manually for deterministic testing
    print("\n[QA] Manually triggering Heartbeat Pulse...")
    pulse_message = "SYSTEM_HEARTBEAT: Perform a self-reflection."
    response = await agent.run(pulse_message)

    # 3. Verify the chain of events
    # The MockProvider for SYSTEM_HEARTBEAT should have match: Workspace Scan
    assert "completed the proactive scan" in response

    # 4. Check L3 (HDD) traces for the 'thinking' and 'message' (insight)
    l3 = duckdb_storage
    traces = await l3.get_recent_traces(collaborator_id, limit=20)

    # Find the 'thinking' trace from the heartbeat
    thinking_traces = [t for t in traces if t["trace_type"] == "thinking"]

    assistant_responses = []
    for t in traces:
        if t["trace_type"] == "message":
            # DuckDB might return the JSON as a string or dict
            meta = t["metadata_json"]
            if isinstance(meta, str):
                import json

                meta = json.loads(meta)

            if meta.get("role") == "assistant":
                assistant_responses.append(t)

    assert len(thinking_traces) > 0, "No thinking trace found for heartbeat!"
    assert any("completed the proactive scan" in t["content"] for t in assistant_responses), (
        "No insight message found in L3!"
    )

    print("\n[QA] Heartbeat E2E Flow Test Passed: Pulse -> Thinking -> Insight -> L3 Archive.")


if __name__ == "__main__":
    asyncio.run(test_heartbeat_proactive_insight_flow())

import asyncio

from mirai.agent.heartbeat import HeartbeatManager
from mirai.agent.loop import AgentLoop
from mirai.agent.providers import MockProvider
from mirai.agent.tools.workspace import WorkspaceTool
from mirai.db.session import init_db


import pytest
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_heartbeat(tmp_path):
    print("--- Starting Heartbeat Logic Test ---")
    
    # Use temp db
    db_path = tmp_path / "test_heartbeat.db"
    db_url = f"sqlite+aiosqlite:///{db_path}"
    
    # Initialize DB (this sets the global engine in session.py)
    await init_db(db_url)
    
    # IMPORTANT: We must also patch where get_session gets its engine if it doesn't use the global one? 
    # session.py uses global _engine. init_db sets it. So this should work if we ensure isolation.
    # To be safe, we must ensure init_db REUSABLE or resets.
    # session.py _get_engine checks if _engine is None.
    # If a previous test set it, we might be using that one!
    # We should probably reset it. But session.py doesn't have reset.
    # For now, let's assume this test runs isolated or we patch _get_engine.
    # Actually, let's just use the default ./mirai.db but delete it first?
    # No, risky. 
    # Let's try to trust init_db updates the engine if we call it?
    # session.py: if _engine is None: create...
    # It does NOT update if already set.
    # We need to force reset the engine in session.py fixtures, but here we are in a standalone test script?
    # Wait, the error is "no such table". This implies the engine is connected to a DB that wasn't initialized.
    # If previous tests initialized ./mirai.db, it should exist.
    # If this test initializes it, it should exist.
    # The failures "OperationalError" often happen with concurrency on SQLite.
    
    # Let's try to force reset the engine variable in session module
    from mirai.db import session
    if session._engine:
        await session._engine.dispose()
        session._engine = None
        session._async_session = None
        
    await init_db(db_url)

    # Initialize components
    provider = MockProvider()
    collaborator_id = "01AN4Z048W7N7DF3SQ5G16CYAJ"
    tools = [WorkspaceTool()]
    
    # Use mocks
    agent = await AgentLoop.create(
        provider=provider, 
        tools=tools, 
        collaborator_id=collaborator_id,
        l3_storage=AsyncMock(),
        l2_storage=AsyncMock(),
        embedder=AsyncMock()
    )
    agent.embedder.get_embeddings = AsyncMock(return_value=[0.0] * 1536)

    # Initialize Heartbeat with 1 second interval for test
    hb = HeartbeatManager(agent, interval_seconds=1)

    print("\n[test] Starting Heartbeat Manager...")
    await hb.start()

    # Wait for a pulse
    print("[test] Waiting for first pulse...")
    await asyncio.sleep(3)

    print("\n[test] Stopping Heartbeat Manager...")
    await hb.stop()

    print("\n--- Test Complete ---")


if __name__ == "__main__":
    asyncio.run(test_heartbeat())

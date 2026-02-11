import asyncio
from mirai.agent.providers import MockProvider
from mirai.agent.loop import AgentLoop
from mirai.agent.tools.workspace import WorkspaceTool
from mirai.agent.heartbeat import HeartbeatManager
from mirai.db.session import init_db

async def test_heartbeat():
    print("--- Starting Heartbeat Logic Test ---")
    await init_db()
    
    # Initialize components
    provider = MockProvider()
    collaborator_id = "01AN4Z048W7N7DF3SQ5G16CYAJ"
    tools = [WorkspaceTool()]
    agent = await AgentLoop.create(
        provider=provider,
        tools=tools,
        collaborator_id=collaborator_id
    )
    
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

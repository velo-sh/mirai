import asyncio
from mirai.agent.providers import MockProvider
from mirai.agent.loop import AgentLoop
from mirai.agent.tools.echo import EchoTool
from mirai.agent.tools.memory import MemorizeTool
from mirai.db.session import init_db

async def test_agent_loop():
    print("--- Starting Agent Loop Mock Test ---")
    await init_db()
    
    # Initialize components
    provider = MockProvider()
    collaborator_id = "01AN4Z048W7N7DF3SQ5G16CYAJ"
    tools = [EchoTool(), MemorizeTool(collaborator_id=collaborator_id)]
    agent = AgentLoop(
        provider=provider,
        tools=tools,
        system_prompt="You are a helpful AI assistant.",
        collaborator_id=collaborator_id
    )
    
    # Run loop
    print("[test] Sending message: 'Please test the echo tool.'")
    response = await agent.run("Please test the echo tool.")
    
    print(f"\n[test] Final Agent Response: {response}")
    print("--- Test Complete ---")

if __name__ == "__main__":
    asyncio.run(test_agent_loop())

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
    # Turn 1: Store an insight
    print("\n--- Turn 1: Storing Insight ---")
    response1 = await agent.run("The project code is located in /Users/antigravity/rust_source/mirai. Remember this.")
    print(f"[test] Final Agent Response 1: {response1}")
    
    # Turn 2: Ask about it (Retrieval)
    print("\n--- Turn 2: Retrieval Check ---")
    response2 = await agent.run("Where is the project code located?")
    print(f"[test] Final Agent Response 2: {response2}")
    
    print("\n--- Test Complete ---")

if __name__ == "__main__":
    asyncio.run(test_agent_loop())

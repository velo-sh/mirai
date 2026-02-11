import asyncio
from mirai.agent.providers import MockProvider
from mirai.agent.loop import AgentLoop
from mirai.agent.tools.workspace import WorkspaceTool
from mirai.db.session import init_db

async def test_workspace():
    print("--- Starting Workspace Tool Test ---")
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
    
    # Test 'list' action
    print("\n[test] Requesting file list...")
    result_list = await agent.run("List the files in the current directory.")
    print(f"[test] Result: {result_list}")
    
    # Test 'read' action
    print("\n[test] Requesting to read main.py...")
    result_read = await agent.run("Read the content of main.py.")
    print(f"[test] Result: {result_read}")

    print("\n--- Test Complete ---")

if __name__ == "__main__":
    asyncio.run(test_workspace())

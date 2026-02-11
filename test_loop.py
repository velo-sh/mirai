import asyncio
from mirai.agent.providers import MockProvider
from mirai.agent.loop import AgentLoop
from mirai.agent.tools.echo import EchoTool

async def test_agent_loop():
    print("--- Starting Agent Loop Mock Test ---")
    
    # Initialize components
    provider = MockProvider()
    tools = [EchoTool()]
    agent = AgentLoop(
        provider=provider,
        tools=tools,
        system_prompt="You are a helpful AI assistant."
    )
    
    # Run loop
    print("[test] Sending message: 'Please test the echo tool.'")
    response = await agent.run("Please test the echo tool.")
    
    print(f"\n[test] Final Agent Response: {response}")
    print("--- Test Complete ---")

if __name__ == "__main__":
    asyncio.run(test_agent_loop())

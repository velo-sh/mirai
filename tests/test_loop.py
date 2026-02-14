import asyncio
from unittest.mock import AsyncMock

import pytest

from mirai.agent.agent_loop import AgentLoop
from mirai.agent.providers import MockProvider
from mirai.agent.tools.echo import EchoTool
from mirai.agent.tools.memory import MemorizeTool
from mirai.collaborator.models import CollaboratorCreate
from mirai.db.session import get_session, init_db


@pytest.mark.asyncio
async def test_agent_loop(tmp_path):
    print("--- Starting Agent Loop Mock Test ---")
    await init_db(f"sqlite+aiosqlite:///{tmp_path / 'mirai.db'}")

    # Seed collaborator
    from mirai.collaborator.manager import CollaboratorManager

    collaborator_id = "01AN4Z048W7N7DF3SQ5G16CYAJ"
    async for session in get_session():
        manager = CollaboratorManager(session)
        await manager.create_collaborator(
            CollaboratorCreate(
                id=collaborator_id,
                name="Test Bot",
                role="AI assistant",
                system_prompt="You are a helpful AI collaborator.",
            )
        )

    # Initialize components
    provider = MockProvider()
    tools = [EchoTool(), MemorizeTool()]

    # Use mocks for storage
    agent = await AgentLoop.create(
        provider=provider,
        tools=tools,
        collaborator_id=collaborator_id,
        l3_storage=AsyncMock(),
        l2_storage=AsyncMock(),
        embedder=AsyncMock(),
    )
    # Mock embedder return
    agent.embedder.get_embeddings = AsyncMock(return_value=[0.0] * 1536)
    print(f"[test] Agent Name: {agent.name}")
    print(f"[test] Agent Role: {agent.role}")

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

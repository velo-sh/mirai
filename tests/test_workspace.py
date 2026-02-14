import asyncio
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from mirai.agent.agent_loop import AgentLoop
from mirai.agent.providers import MockProvider
from mirai.agent.tools.workspace import WorkspaceTool
from mirai.db.session import init_db


@pytest_asyncio.fixture(autouse=True)
async def setup_db():
    await init_db()


@pytest.mark.asyncio
async def test_workspace():
    print("--- Starting Workspace Tool Test ---")

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
        embedder=AsyncMock(),
    )
    agent.embedder.get_embeddings = AsyncMock(return_value=[0.0] * 1536)

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

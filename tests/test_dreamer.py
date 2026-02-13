import asyncio
import json
import os
import shutil

from mirai.agent.loop import AgentLoop
from mirai.agent.providers import MockProvider
from mirai.agent.tools.echo import EchoTool
from mirai.db.session import init_db
from mirai.memory.dreamer import Dreamer
from mirai.memory.vector_db import VectorStore


import pytest
from mirai.db.duck import DuckDBStorage
from mirai.memory.vector_db import VectorStore

@pytest.mark.asyncio
async def test_dreaming_flow(tmp_path):
    print("--- Starting Dreaming Engine Test ---")

    # Use temp paths
    db_path = str(tmp_path / "test_dreamer.duckdb")
    l3 = DuckDBStorage(db_path=db_path)
    
    # Vector store might need similar handling but let's assume it's okay or mock it if possible.
    # Actually VectorStore init uses lancedb.connect("mirai_vectors"). check if we can override.
    # VectorStore def __init__(self, uri="mirai_vectors")
    l2_path = str(tmp_path / "mirai_vectors")
    l2 = VectorStore(db_path=l2_path)
    
    await init_db()

    collaborator_id = "01AN4Z048W7N7DF3SQ5G16CYAJ"
    provider = MockProvider()
    tools = [EchoTool()]
    
    from unittest.mock import AsyncMock
    mock_embedder = AsyncMock()
    mock_embedder.get_embeddings = AsyncMock(return_value=[0.0] * 1536)
    
    # Inject storage
    agent = await AgentLoop.create(
        provider=provider, 
        tools=tools, 
        collaborator_id=collaborator_id,
        l3_storage=l3,
        l2_storage=l2,
        embedder=mock_embedder
    )

    # 1. First interaction (NOT memorized explicitly)
    print("\n--- Interaction 1 (Normal Chat) ---")
    # We need to bypass the mock logic that triggers 'memorize' tool
    # Let's just run it; if it doesn't use the tool, it's just a message in L3
    provider.call_count = 10  # Force it past the hardcoded tool-call logic
    await agent.run("Mirai's secret password is 'Antigravity2026'.")

    # 2. Check L2 (Should be empty)
    vdb = l2
    results = await vdb.search(vector=[0.0] * 1536, limit=1)  # Search for anything
    print(f"[test] Memories in L2 before dreaming: {len(results)}")

    # 3. Running the Dreamer
    print("\n--- Dreaming Phase ---")
    print("\n--- Dreaming Phase ---")
    dreamer = Dreamer(agent, l3)
    await dreamer.dream_once()

    # 4. Check L2 again (Should have the secret password)
    results = await vdb.search(vector=[0.0] * 1536, limit=5)
    print(f"[test] Memories in L2 after dreaming: {len(results)}")
    has_secret = any(
        "secret password" in json.loads(r["metadata"]).get("content", "") or "secret password" in r.get("content", "")
        for r in results
    )
    # Wait, LanceDB search results structure depends on to_list()
    # In my VectorStore, search returns query.to_list()
    print(f"[test] Secret found in L2 index: {has_secret}")

    # 5. Retrieval Check
    print("\n--- Retrieval Verification ---")
    # Now ask about the secret. Total Recall should fetch it from L3.
    response = await agent.run("What is Mirai's secret password?")
    # Check if memories were injected (we'll see it in the mock log if it worked)
    print(f"[test] Final Answer: {response}")


if __name__ == "__main__":
    asyncio.run(test_dreaming_flow())

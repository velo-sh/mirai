import asyncio
from unittest.mock import AsyncMock

import pytest

from mirai.agent.agent_loop import AgentLoop
from mirai.agent.providers import MockProvider
from mirai.agent.tools.memory import MemorizeTool
from mirai.collaborator.manager import CollaboratorManager
from mirai.collaborator.models import CollaboratorCreate
from mirai.db.session import get_session, init_db


@pytest.mark.asyncio
async def test_memory_isolation_between_collaborators(tmp_path):
    """
    QA Integration Test: Verify that memories are strictly isolated by collaborator_id.
    """
    await init_db()

    # Use isolated vector db
    from mirai.memory.vector_db import VectorStore

    isolated_l2 = VectorStore(db_path=str(tmp_path / "vectors"))

    # 1. Setup two distinct collaborators
    async for session in get_session():
        manager = CollaboratorManager(session)

        collab_a_id = "01AN4Z048A_A_A_A_A_A_A_A_A"
        collab_b_id = "01AN4Z048B_B_B_B_B_B_B_B_B"

        if not await manager.get_collaborator(collab_a_id):
            await manager.create_collaborator(
                CollaboratorCreate(id=collab_a_id, name="Alice", role="QA", system_prompt="Alice prompt")
            )
        if not await manager.get_collaborator(collab_b_id):
            await manager.create_collaborator(
                CollaboratorCreate(id=collab_b_id, name="Bob", role="Dev", system_prompt="Bob prompt")
            )

    # 2. Alice stores a secret memory
    provider = MockProvider()

    # Use mock L3 storage to avoid DuckDB file lock contention
    mock_l3 = AsyncMock()
    mock_l3.append_trace = AsyncMock()
    mock_l3.get_traces_by_ids = AsyncMock(return_value=[])

    mock_embedder = AsyncMock()
    mock_embedder.get_embeddings = AsyncMock(return_value=[0.1] * 1536)

    alice_agent = await AgentLoop.create(
        provider=provider,
        tools=[MemorizeTool(collaborator_id=collab_a_id, vector_store=isolated_l2, l3_storage=mock_l3)],
        collaborator_id=collab_a_id,
        l3_storage=mock_l3,
        l2_storage=isolated_l2,
        embedder=mock_embedder,
    )

    # Alice memorizes something
    await alice_agent.run("Alice's secret is: WonderLand.")

    import asyncio

    await asyncio.sleep(1)  # Allow DB flush

    # 3. Bob tries to retrieve it
    bob_agent = await AgentLoop.create(
        provider=provider,
        tools=[MemorizeTool(collaborator_id=collab_b_id, vector_store=isolated_l2, l3_storage=mock_l3)],
        collaborator_id=collab_b_id,
        l3_storage=mock_l3,
        l2_storage=isolated_l2,
        embedder=mock_embedder,
    )

    # Bob asks for the secret
    await bob_agent.run("What is Alice's secret?")

    # Verification: Check Bob's retrieval results
    query_vector = await bob_agent.embedder.get_embeddings("Alice's secret")
    bob_memories = await bob_agent.l2_storage.search(
        vector=query_vector, limit=5, filter=f"collaborator_id = '{collab_b_id}'"
    )

    # Check Bob's results
    for mem in bob_memories:
        assert "WonderLand" not in mem["content"], f"Data Leak! Bob found Alice's secret: {mem['content']}"

    # Now check Alice's results (she should see it)

    # Direct check on isolated_l2
    await isolated_l2.search(
        vector=[0.1] * 1536,  # MockEmbedder default
        limit=10,
    )

    query_vector_alice = await alice_agent.embedder.get_embeddings("secret")
    alice_memories = await alice_agent.l2_storage.search(
        vector=query_vector_alice, limit=5, filter=f"collaborator_id = '{collab_a_id}'"
    )

    assert any("WonderLand" in m["content"] for m in alice_memories), "Alice couldn't find her own secret!"

    print("\n[QA] Memory Isolation Test Passed: Alice and Bob have separate cognitive spaces.")


if __name__ == "__main__":
    asyncio.run(test_memory_isolation_between_collaborators())

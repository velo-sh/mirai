import asyncio

import pytest

from mirai.agent.loop import AgentLoop
from mirai.agent.providers import MockProvider
from mirai.agent.tools.memory import MemorizeTool
from mirai.collaborator.manager import CollaboratorManager
from mirai.collaborator.models import CollaboratorCreate
from mirai.db.session import get_session, init_db


@pytest.mark.asyncio
async def test_memory_isolation_between_collaborators():
    """
    QA Integration Test: Verify that memories are strictly isolated by collaborator_id.
    """
    await init_db()

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
    alice_agent = await AgentLoop.create(
        provider=provider, tools=[MemorizeTool(collaborator_id=collab_a_id)], collaborator_id=collab_a_id
    )

    # Alice memorizes something
    await alice_agent.run("Alice's secret is: WonderLand.")

    # 3. Bob tries to retrieve it
    bob_agent = await AgentLoop.create(
        provider=provider, tools=[MemorizeTool(collaborator_id=collab_b_id)], collaborator_id=collab_b_id
    )

    # Bob asks for the secret
    # AgentLoop.run handles retrieval internally. We check the system prompt constructed for Bob.
    # We can't easily peek into private memories from the outside,
    # but we can check if Bob's agent finds it in L2/L3.

    await bob_agent.run("What is Alice's secret?")

    # Verification: Check Bob's retrieval results
    # Bob should not see Alice's memories
    query_vector = await bob_agent.embedder.get_embeddings("Alice's secret")
    bob_memories = await bob_agent.l2_storage.search(
        vector=query_vector, limit=5, filter=f"collaborator_id = '{collab_b_id}'"
    )

    # Check Bob's results
    for mem in bob_memories:
        assert "WonderLand" not in mem["content"], f"Data Leak! Bob found Alice's secret: {mem['content']}"

    # Now check Alice's results (she should see it)
    query_vector_alice = await alice_agent.embedder.get_embeddings("secret")
    alice_memories = await alice_agent.l2_storage.search(
        vector=query_vector_alice, limit=5, filter=f"collaborator_id = '{collab_a_id}'"
    )
    assert any("WonderLand" in m["content"] for m in alice_memories), "Alice couldn't find her own secret!"

    print("\n[QA] Memory Isolation Test Passed: Alice and Bob have separate cognitive spaces.")


if __name__ == "__main__":
    asyncio.run(test_memory_isolation_between_collaborators())

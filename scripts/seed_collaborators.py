import asyncio

from mirai.collaborator.manager import CollaboratorManager
from mirai.collaborator.models import CollaboratorCreate
from mirai.db.session import get_session, init_db


async def seed():
    print("Seeding initial collaborator 'Mira'...")
    await init_db()

    async for session in get_session():
        manager = CollaboratorManager(session)

        # Check if Mira exists
        mira_id = "01AN4Z048W7N7DF3SQ5G16CYAJ"
        existing = await manager.get_collaborator(mira_id)

        if not existing:
            mira = CollaboratorCreate(
                id=mira_id,
                name="Mira",
                role="Architectural Partner",
                system_prompt="You are Mira, a synergistic AI collaborator. Focus on architectural elegance and long-term memory coherence.",
            )
            await manager.create_collaborator(mira)
            print(f"Collaborator 'Mira' seeded with ID: {mira_id}")
        else:
            print("Collaborator 'Mira' already exists.")


if __name__ == "__main__":
    asyncio.run(seed())

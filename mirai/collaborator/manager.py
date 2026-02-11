from typing import List, Optional
from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession
from mirai.collaborator.models import Collaborator, CollaboratorCreate
from ulid import ULID

class CollaboratorManager:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_collaborator(self, data: CollaboratorCreate) -> Collaborator:
        collaborator_id = data.id or str(ULID())
        collaborator = Collaborator(
            id=collaborator_id,
            name=data.name,
            role=data.role,
            system_prompt=data.system_prompt,
            avatar_url=data.avatar_url
        )
        self.session.add(collaborator)
        await self.session.commit()
        await self.session.refresh(collaborator)
        return collaborator

    async def get_collaborator(self, collaborator_id: str) -> Optional[Collaborator]:
        statement = select(Collaborator).where(Collaborator.id == collaborator_id)
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def list_collaborators(self) -> List[Collaborator]:
        statement = select(Collaborator)
        result = await self.session.execute(statement)
        return result.scalars().all()

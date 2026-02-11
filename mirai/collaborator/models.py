from typing import Optional
from sqlmodel import Field, SQLModel, Session, create_engine

class CollaboratorBase(SQLModel):
    name: str = Field(index=True)
    role: str
    system_prompt: str
    avatar_url: Optional[str] = None

class Collaborator(CollaboratorBase, table=True):
    id: Optional[str] = Field(default=None, primary_key=True)

class CollaboratorCreate(CollaboratorBase):
    pass

class CollaboratorRead(CollaboratorBase):
    id: int

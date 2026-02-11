from typing import Optional
from sqlmodel import Field, SQLModel, Session, create_engine

class CollaboratorBase(SQLModel):
    name: str = Field(index=True)
    role: str
    system_prompt: str
    avatar_url: Optional[str] = None

class Collaborator(CollaboratorBase, table=True):
    id: str = Field(primary_key=True)

class CollaboratorCreate(CollaboratorBase):
    id: Optional[str] = None # Allow providing a ULID or generate one

class CollaboratorRead(CollaboratorBase):
    id: str

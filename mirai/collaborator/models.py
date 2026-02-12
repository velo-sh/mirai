from sqlmodel import Field, SQLModel


class CollaboratorBase(SQLModel):
    name: str = Field(index=True)
    role: str
    system_prompt: str
    avatar_url: str | None = None


class Collaborator(CollaboratorBase, table=True):
    id: str = Field(primary_key=True)


class CollaboratorCreate(CollaboratorBase):
    id: str | None = None  # Allow providing a ULID or generate one


class CollaboratorRead(CollaboratorBase):
    id: str

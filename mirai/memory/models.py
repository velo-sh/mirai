from datetime import datetime
from uuid import UUID, uuid4
from typing import Optional, Dict, Any, List
from sqlmodel import Field, SQLModel

class CognitiveTrace(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    timestamp: datetime = Field(default_factory=datetime.utcnow, index=True)
    collaborator_id: int = Field(index=True)
    trace_type: str = Field(index=True) # e.g., "message", "thought", "tool"
    content: str
    metadata_json: str = Field(default="{}")
    importance: float = Field(default=0.0)
    vector_id: Optional[str] = Field(default=None, index=True)

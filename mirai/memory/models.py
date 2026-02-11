from datetime import datetime
from ulid import ULID
from typing import Optional, Dict, Any, List
from sqlmodel import Field, SQLModel

def get_ulid():
    return str(ULID())

class CognitiveTrace(SQLModel, table=True):
    id: str = Field(default_factory=get_ulid, primary_key=True)
    timestamp: datetime = Field(default_factory=datetime.utcnow, index=True)
    collaborator_id: str = Field(index=True)
    trace_type: str = Field(index=True) # e.g., "message", "thought", "tool"
    content: str
    metadata_json: str = Field(default="{}")
    importance: float = Field(default=0.0)
    vector_id: Optional[str] = Field(default=None, index=True)

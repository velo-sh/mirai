from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DBRecall(BaseModel):
    """A recalled memory from vector search."""

    model_config = ConfigDict(populate_by_name=True)

    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    distance: float | None = None


class DBTrace(BaseModel):
    """A trace entry in cognitive_traces table."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    collaborator_id: str
    trace_type: str  # e.g., 'thought', 'action', 'error'
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict, alias="metadata_json")
    importance: float = 0.0
    vector_id: str | None = None

    @property
    def metadata_json(self) -> dict[str, Any]:
        return self.metadata

    @field_validator("metadata", mode="before")
    @classmethod
    def parse_json_string(cls, v: Any) -> Any:
        if isinstance(v, str):
            import orjson

            return orjson.loads(v)
        return v


class FeishuMessage(BaseModel):
    """A message turn in feishu_history table."""

    model_config = ConfigDict(populate_by_name=True)

    chat_id: str
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

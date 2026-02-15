"""Domain entities for town_elder."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Represents a document in the vector store."""
    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)

    model_config = {"frozen": True}


class Chunk(BaseModel):
    """Represents a chunk of a document."""
    id: str
    document_id: str
    text: str
    index: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """Represents a search result."""
    document: Document
    chunk: Chunk | None = None
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)

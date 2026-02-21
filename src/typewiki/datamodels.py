from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

Role = Literal['user', 'assistant']


class ChatMessage(BaseModel):
    role: Role
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    """
    Request model for POST /v1/chat

    - message: the new user message for this turn
    - conversation_id: stable identifier for the conversation/thread
    - history: prior messages (optional). If you pass this, keep it small.
    """

    message: str = Field(..., min_length=1, description="User's new message for this turn")
    session_id: str = Field(
        ...,
        description='Client-provided conversation/thread id. If omitted, server may create one.',
    )
    history: list[ChatMessage] = Field(
        default_factory=list,
        description='Optional conversation history, most recent first or chronological.',
    )


class SourceRef(BaseModel):
    """
    One retrieved chunk (or doc) used to support the answer.
    """

    chunk_id: str = Field(..., description='Deterministic id for the chunk/vector')
    score: float = Field(..., ge=0.0, description='Similarity score (higher is more relevant)')
    article_title: str = Field(..., description='Human-readable article title')
    source_url: str = Field(..., description='Canonical source URL for attribution')


class ChatResponse(BaseModel):
    """
    Response model for POST /v1/chat
    """

    conversation_id: str = Field(..., description='Conversation/thread id')
    answer: str = Field(..., description='Assistant answer text')
    sources: list[SourceRef] = Field(default_factory=list)

    model: str | None = Field(default=None, description='LLM model name used')
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

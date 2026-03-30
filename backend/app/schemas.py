from pydantic import BaseModel, Field
from typing import List, Optional


class WebIngestRequest(BaseModel):
    url: str


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    history: Optional[List[ChatMessage]] = []


class SourceItem(BaseModel):
    doc_id: str
    doc_name: str
    source: str
    chunk_index: int
    score: float
    content: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
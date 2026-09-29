"""Pydantic models for the application."""

from typing import List, Optional

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    filename: str
    collection_name: str
    num_chunks: int
    message: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    collection_name: Optional[str] = Field(None, description="Defaults to CHROMA_COLLECTION")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Defaults to RETRIEVAL_TOP_K")


class SourceDocument(BaseModel):
    content: str
    source: str
    page: Optional[int] = None
    relevance_score: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    agent_reasoning: List[str]
    retrievals: int
    query_time_seconds: float


class CollectionInfo(BaseModel):
    name: str
    num_chunks: int


class CollectionListResponse(BaseModel):
    collections: List[CollectionInfo]


class DeleteResponse(BaseModel):
    message: str
    collection_name: str


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str

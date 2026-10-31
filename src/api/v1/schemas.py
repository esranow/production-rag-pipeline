from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, description="The user's question or search query")
    top_k: Optional[int] = Field(default=5, ge=1, le=20, description="Number of source chunks to return after reranking")
    use_hybrid: bool = Field(default=True, description="Enable hybrid dense + sparse retrieval")
    use_reranker: bool = Field(default=True, description="Enable cross-encoder reranker")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filter conditions")


class SourceChunk(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    relevance_score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[SourceChunk]
    model_used: str
    latency_ms: float


class IngestDocumentRequest(BaseModel):
    document_id: str
    title: str
    content: str
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    status: str
    document_id: str
    chunks_created: int
    message: str

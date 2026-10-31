import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from src.core.config import settings

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """Manages Qdrant vector storage, payload indexing, and similarity search."""

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None, collection_name: Optional[str] = None):
        self.host = host or settings.QDRANT_HOST
        self.port = port or settings.QDRANT_PORT
        self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
        self.client = QdrantClient(host=self.host, port=self.port, api_key=settings.QDRANT_API_KEY)

    def initialize_collection(self, vector_size: int = 1536, distance: models.Distance = models.Distance.COSINE):
        """Ensures the collection exists with correct vector dimensions and payload indices."""
        collections = [col.name for col in self.client.get_collections().collections]
        if self.collection_name not in collections:
            logger.info(f"Creating Qdrant collection: {self.collection_name} (dim={vector_size})")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=distance),
                optimizers_config=models.OptimizersConfigDiff(indexing_threshold=10000),
                hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100)
            )
            # Create payload indexes for fast filtering
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="document_id",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="parent_id",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
        else:
            logger.info(f"Qdrant collection '{self.collection_name}' already exists.")

    def upsert_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """Batch upsert chunks with their vector embeddings and metadata payload."""
        points = [
            models.PointStruct(
                id=chunk["chunk_id"],
                vector=embedding,
                payload={
                    "document_id": chunk["document_id"],
                    "parent_id": chunk.get("parent_id"),
                    "content": chunk["content"],
                    "metadata": chunk.get("metadata", {}),
                }
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info(f"Upserted {len(points)} chunks into {self.collection_name}.")

    def search_dense(self, query_vector: List[float], limit: int = 25, filter_conditions: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Perform dense vector search with optional payload filters."""
        q_filter = None
        if filter_conditions:
            must_clauses = [
                models.FieldCondition(key=k, match=models.MatchValue(value=v))
                for k, v in filter_conditions.items()
            ]
            q_filter = models.Filter(must=must_clauses)

        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=q_filter,
            with_payload=True
        )

        return [
            {
                "chunk_id": str(hit.id),
                "score": hit.score,
                "content": hit.payload.get("content", ""),
                "document_id": hit.payload.get("document_id", ""),
                "parent_id": hit.payload.get("parent_id"),
                "metadata": hit.payload.get("metadata", {})
            }
            for hit in search_results
        ]

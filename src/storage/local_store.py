import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from src.core.config import settings

logger = logging.getLogger(__name__)


class LocalVectorStore:
    """
    Zero-dependency, persistent local vector store using NumPy for vector math.
    Allows testing and running the RAG pipeline out-of-the-box without Docker.
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or settings.LOCAL_STORAGE_PATH
        self.points: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self.points = json.load(f)
                logger.info(f"Loaded {len(self.points)} chunks from {self.storage_path}")
            except Exception as e:
                logger.warning(f"Could not load local storage: {e}")
                self.points = {}

    def _save(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.storage_path)), exist_ok=True)
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self.points, f, indent=2)

    def upsert_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        for chunk, emb in zip(chunks, embeddings):
            c_id = chunk["chunk_id"]
            self.points[c_id] = {
                "chunk_id": c_id,
                "document_id": chunk["document_id"],
                "parent_id": chunk.get("parent_id"),
                "content": chunk["content"],
                "vector": emb,
                "metadata": chunk.get("metadata", {}),
            }
        self._save()
        logger.info(f"LocalVectorStore saved {len(chunks)} chunks. Total points: {len(self.points)}")

    def search_dense(
        self,
        query_vector: List[float],
        limit: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        if not self.points:
            return []

        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []

        scored = []
        for point in self.points.values():
            # Check filters if any
            if filter_conditions:
                match = True
                for k, v in filter_conditions.items():
                    if point.get("metadata", {}).get(k) != v and point.get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            p_vec = np.array(point["vector"], dtype=np.float32)
            p_norm = np.linalg.norm(p_vec)
            if p_norm == 0:
                continue

            similarity = float(np.dot(q_vec, p_vec) / (q_norm * p_norm))
            scored.append({
                "chunk_id": point["chunk_id"],
                "score": round(similarity, 4),
                "content": point["content"],
                "document_id": point["document_id"],
                "parent_id": point.get("parent_id"),
                "metadata": point.get("metadata", {})
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def clear(self):
        self.points = {}
        self._save()

    @property
    def total_count(self) -> int:
        return len(self.points)

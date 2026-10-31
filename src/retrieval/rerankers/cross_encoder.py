import logging
from typing import List, Dict, Any
from src.core.config import settings

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranks candidate chunks using a cross-encoder model (e.g., BAAI/bge-reranker-v2-m3).
    A cross-encoder scores query-chunk pairs simultaneously via full cross-attention.
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.RERANKER_MODEL
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading CrossEncoder model: {self.model_name}")
                self._model = CrossEncoder(self.model_name)
            except Exception as e:
                logger.warning(f"Could not load local CrossEncoder ({e}). Falling back to heuristic/passthrough.")
                self._model = False

    def rerank(self, query: str, candidate_chunks: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
        if not candidate_chunks:
            return []

        self._load_model()

        if self._model:
            pairs = [[query, chunk["content"]] for chunk in candidate_chunks]
            scores = self._model.predict(pairs)
            for idx, score in enumerate(scores):
                candidate_chunks[idx]["rerank_score"] = float(score)

            # Sort descending by rerank score
            sorted_chunks = sorted(candidate_chunks, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
            return sorted_chunks[:top_n]
        else:
            # Fallback: keep original fusion order
            return candidate_chunks[:top_n]

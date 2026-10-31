import logging
import requests
from typing import List, Optional
from src.embeddings.base import BaseEmbeddingProvider
from src.core.config import settings

logger = logging.getLogger(__name__)


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """
    Resilient Google Gemini Embedding Provider.
    Attempts google.genai SDK first, seamlessly falls back to direct REST API
    (avoiding any 'cannot import name genai from google' namespace collision errors).
    """

    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        self.model_name = model_name or settings.DEFAULT_EMBEDDING_MODEL
        self.api_key = api_key or settings.GEMINI_API_KEY
        self._dim = settings.EMBEDDING_DIMENSION
        self.sdk_client = None

        # Attempt to load google.genai if available without namespace errors
        try:
            from google import genai
            self.sdk_client = genai.Client(api_key=self.api_key)
            logger.info("Using google.genai SDK client.")
        except Exception as e:
            logger.info(f"SDK unavailable or namespace conflict ({e}); using direct Gemini REST API.")
            self.sdk_client = None

    def _embed_via_rest(self, text: str) -> List[float]:
        clean_model = self.model_name.replace("models/", "")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:embedContent?key={self.api_key}"
        payload = {"content": {"parts": [{"text": text}]}}
        resp = requests.post(url, json=payload, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"Gemini Embedding API Error ({resp.status_code}): {resp.text}")
        data = resp.json()
        return data["embedding"]["values"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            if self.sdk_client:
                try:
                    res = self.sdk_client.models.embed_content(
                        model=self.model_name,
                        contents=text
                    )
                    embeddings.append(res.embeddings[0].values)
                    continue
                except Exception:
                    # Fallback to REST
                    pass
            embeddings.append(self._embed_via_rest(text))
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        if self.sdk_client:
            try:
                res = self.sdk_client.models.embed_content(
                    model=self.model_name,
                    contents=text
                )
                return res.embeddings[0].values
            except Exception:
                pass
        return self._embed_via_rest(text)

    @property
    def dimension(self) -> int:
        return self._dim


class SentenceTransformerProvider(BaseEmbeddingProvider):
    """Local offline embedding using sentence-transformers (zero API keys)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self._dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Initialized SentenceTransformer: {model_name} (dim={self._dim})")
        except Exception as e:
            raise RuntimeError(f"Could not load SentenceTransformer: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode([text], normalize_embeddings=True)[0]
        return embedding.tolist()

    @property
    def dimension(self) -> int:
        return self._dim

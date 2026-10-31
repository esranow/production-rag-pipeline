import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # App
    APP_NAME: str = "Production-RAG-Pipeline"
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Google Gemini & LLM Provider
    # Prioritizes os.environ, then Streamlit Cloud secrets, then None
    GEMINI_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    DEFAULT_LLM_MODEL: str = "gemini-2.5-flash-lite"
    DEFAULT_EMBEDDING_MODEL: str = "gemini-embedding-001"
    EMBEDDING_DIMENSION: int = 3072

    # Storage Backend ('local' for zero-docker persistent JSON/numpy, or 'qdrant')
    VECTOR_STORE_BACKEND: str = "local"
    LOCAL_STORAGE_PATH: str = "data/vector_store.json"
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "production_knowledge_base"
    QDRANT_API_KEY: Optional[str] = None

    # Search & Reranking
    TOP_K_RETRIEVAL: int = 10
    TOP_K_RERANK: int = 4
    HYBRID_ALPHA: float = 0.5
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    MIN_SIMILARITY_SCORE: float = 0.35

    # Redis Cache
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    CACHE_TTL_SECONDS: int = 3600


settings = Settings()

# Check for Streamlit Cloud secrets fallback
if not settings.GEMINI_API_KEY:
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            settings.GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
            settings.GOOGLE_API_KEY = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

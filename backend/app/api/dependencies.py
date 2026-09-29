"""Shared, lazily created singletons injected into the API routes.

Each object is built once per process (the embedding model, vector store, LLM
client and compiled agent graph are all expensive to create) and reused for
every request. Tests replace them with app.dependency_overrides.
"""

import threading
from functools import lru_cache

from fastapi import HTTPException

from app.agent.graph import LLMNotConfiguredError, build_agent_graph, build_llm
from app.config import Settings, get_settings as _core_get_settings
from app.ingestion.embedder import EmbeddingService
from app.vectorstore.chroma_store import ChromaStore

_lock = threading.Lock()


def get_settings() -> Settings:
    return _core_get_settings()


@lru_cache
def _embedding_service() -> EmbeddingService:
    return EmbeddingService(model_name=get_settings().EMBEDDING_MODEL)


@lru_cache
def _chroma_store() -> ChromaStore:
    settings = get_settings()
    return ChromaStore(
        persist_dir=settings.CHROMA_PERSIST_DIR,
        default_collection=settings.CHROMA_COLLECTION,
        embedding_service=_embedding_service(),
    )


@lru_cache
def _agent_graph():
    # build_llm raises LLMNotConfiguredError when the API key is missing; lru_cache
    # does not cache exceptions, so the graph is built once the key is set.
    return build_agent_graph(_chroma_store(), build_llm(get_settings()))


def get_chroma_store() -> ChromaStore:
    with _lock:  # sync routes run in a thread pool; avoid loading the model twice
        return _chroma_store()


def get_agent_graph():
    with _lock:
        try:
            return _agent_graph()
        except LLMNotConfiguredError as e:
            raise HTTPException(
                status_code=503,
                detail=f"LLM is not configured: {e}. Set it in .env and restart the backend.",
            ) from e

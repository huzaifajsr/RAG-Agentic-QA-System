import functools
from app.config import Settings, get_settings as core_get_settings
from app.ingestion import EmbeddingService
from app.vectorstore.chroma_store import ChromaStore

@functools.lru_cache()
def get_settings() -> Settings:
    return core_get_settings()

_embedding_service = None
_chroma_store = None

def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        settings = get_settings()
        _embedding_service = EmbeddingService(model_name=settings.EMBEDDING_MODEL)
    return _embedding_service

def get_chroma_store() -> ChromaStore:
    global _chroma_store
    if _chroma_store is None:
        settings = get_settings()
        emb_service = get_embedding_service()
        _chroma_store = ChromaStore(
            persist_dir=settings.CHROMA_PERSIST_DIR,
            collection_name=settings.CHROMA_COLLECTION,
            embedding_service=emb_service
        )
    return _chroma_store

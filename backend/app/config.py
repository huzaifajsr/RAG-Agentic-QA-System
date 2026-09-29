"""Application configuration module."""

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    LLM_PROVIDER: str = "google"
    GOOGLE_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    LLM_MODEL: str = "gemini-2.0-flash"
    LLM_TEMPERATURE: float = 0.3
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    CHROMA_COLLECTION: str = "documents"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    UPLOAD_DIR: str = "./data/uploads"
    MAX_FILE_SIZE_MB: int = 50
    MAX_AGENT_ITERATIONS: int = 3
    RETRIEVAL_TOP_K: int = 5
    # Comma-separated list of origins allowed to call the API from a browser.
    CORS_ORIGINS: str = "http://localhost:8501"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return application settings (read once per process)."""
    return Settings()

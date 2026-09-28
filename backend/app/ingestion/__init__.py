"""Document ingestion pipeline — loading, chunking, and embedding."""

from app.ingestion.loader import DocumentLoader
from app.ingestion.chunker import TextChunker
from app.ingestion.embedder import EmbeddingService

__all__ = ["DocumentLoader", "TextChunker", "EmbeddingService"]

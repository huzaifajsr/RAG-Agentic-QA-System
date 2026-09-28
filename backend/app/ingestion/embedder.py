"""Embedding service functionality."""

import logging
from typing import List
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service to generate embeddings for texts."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedding service.
        
        Args:
            model_name: The SentenceTransformer model to load.
        """
        self.model_name = model_name
        logger.info(f"Loading embedding model: {model_name}")
        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded with dimension: {self._dimension}")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Encode a batch of texts.
        
        Args:
            texts: List of strings to embed.
            
        Returns:
            List of embedding vectors.
        """
        embeddings = self._model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Encode a single query string.
        
        Args:
            query: The query string to embed.
            
        Returns:
            The embedding vector.
        """
        embedding = self._model.encode([query])[0]
        return embedding.tolist()

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self._dimension

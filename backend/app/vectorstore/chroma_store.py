"""ChromaDB storage module."""

import hashlib
import logging
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.errors import NotFoundError

logger = logging.getLogger(__name__)


class CollectionNotFoundError(Exception):
    """Raised when a named collection does not exist."""


class ChromaStore:
    """Vector store using ChromaDB.

    Collections are looked up by name on every call instead of caching a single
    collection handle, so deleting and re-creating a collection is safe and every
    endpoint can work on any named collection.
    """

    def __init__(self, persist_dir: Optional[str], default_collection: str, embedding_service):
        """Initialize the ChromaDB store.

        Args:
            persist_dir: Directory to persist ChromaDB data (None = in-memory, for tests).
            default_collection: Collection used when a call does not name one.
            embedding_service: Object with embed_texts() and embed_query().
        """
        self.client = (
            chromadb.PersistentClient(path=persist_dir)
            if persist_dir
            else chromadb.EphemeralClient()
        )
        self.default_collection = default_collection
        self.embedding_service = embedding_service

    # ------------------------------------------------------------------ helpers

    def _name(self, collection_name: Optional[str]) -> str:
        return collection_name or self.default_collection

    def _get(self, collection_name: Optional[str]):
        name = self._name(collection_name)
        try:
            return self.client.get_collection(name=name)
        except (NotFoundError, ValueError) as e:
            raise CollectionNotFoundError(name) from e

    @staticmethod
    def _chunk_id(chunk: Dict[str, Any]) -> str:
        meta = chunk["metadata"]
        key = f"{meta.get('source', '')}|{meta.get('page', '')}|{meta.get('chunk_index', 0)}|{chunk['content']}"
        return hashlib.md5(key.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------ public API

    def add_documents(self, chunks: List[Dict[str, Any]], collection_name: Optional[str] = None) -> int:
        """Add (or overwrite) document chunks in a collection, creating it if needed.

        Returns:
            Number of chunks written.
        """
        if not chunks:
            return 0
        collection = self.client.get_or_create_collection(name=self._name(collection_name))
        texts = [c["content"] for c in chunks]
        # upsert: re-uploading the same file replaces its chunks instead of
        # silently skipping them as duplicate IDs.
        collection.upsert(
            ids=[self._chunk_id(c) for c in chunks],
            embeddings=self.embedding_service.embed_texts(texts),
            documents=texts,
            metadatas=[c["metadata"] for c in chunks],
        )
        return len(chunks)

    def search(self, query: str, k: int = 5, collection_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return up to k chunks most similar to the query.

        Raises:
            CollectionNotFoundError: if the collection does not exist.
        """
        collection = self._get(collection_name)
        count = collection.count()
        if count == 0:
            return []

        results = collection.query(
            query_embeddings=[self.embedding_service.embed_query(query)],
            n_results=min(k, count),
            include=["documents", "metadatas", "distances"],
        )

        structured = []
        docs = results["documents"][0] if results["documents"] else []
        for i, content in enumerate(docs):
            distance = results["distances"][0][i] if results["distances"] else 0.0
            structured.append({
                "content": content,
                "metadata": (results["metadatas"][0][i] if results["metadatas"] else {}) or {},
                # Maps distance (0 = identical) to a 0-1 score where higher is better.
                "score": 1 / (1 + distance),
            })
        return structured

    def get_collection_info(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        collection = self._get(collection_name)
        return {"name": collection.name, "count": collection.count()}

    def list_collections(self) -> List[Dict[str, Any]]:
        info = []
        for col in self.client.list_collections():
            # chromadb >= 0.6 returns Collection objects; older versions returned names.
            name = col if isinstance(col, str) else col.name
            info.append(self.get_collection_info(name))
        return info

    def delete_collection(self, collection_name: Optional[str] = None) -> None:
        name = self._name(collection_name)
        self._get(name)  # raises CollectionNotFoundError if missing
        self.client.delete_collection(name=name)

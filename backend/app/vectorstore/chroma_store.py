"""ChromaDB storage module."""

import hashlib
import logging
from typing import List, Dict, Any
import chromadb
from app.ingestion.embedder import EmbeddingService

logger = logging.getLogger(__name__)

class ChromaStore:
    """Vector store using ChromaDB."""

    def __init__(self, persist_dir: str, collection_name: str, embedding_service: EmbeddingService):
        """Initialize the ChromaDB store.
        
        Args:
            persist_dir: Directory to persist ChromaDB data.
            collection_name: Name of the collection.
            embedding_service: Service to generate embeddings.
        """
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection_name = collection_name
        self.embedding_service = embedding_service
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, chunks: List[Dict[str, Any]]) -> int:
        """Add document chunks to the collection.
        
        Args:
            chunks: List of chunk dictionaries containing 'content' and 'metadata'.
            
        Returns:
            Count of added documents.
        """
        if not chunks:
            return 0
            
        texts = [chunk["content"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        try:
            embeddings = self.embedding_service.embed_texts(texts)
            
            ids = []
            for chunk in chunks:
                content = chunk["content"]
                source = chunk["metadata"].get("source", "")
                chunk_index = chunk["metadata"].get("chunk_index", 0)
                
                unique_str = f"{content}{source}{chunk_index}"
                doc_id = hashlib.md5(unique_str.encode("utf-8")).hexdigest()
                ids.append(doc_id)
            
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            return len(chunks)
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
            raise

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents.
        
        Args:
            query: The search query.
            k: Number of results to return.
            
        Returns:
            List of dictionaries with 'content', 'metadata', and 'score'.
        """
        try:
            query_embedding = self.embedding_service.embed_query(query)
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            
            structured_results = []
            if results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    distance = results["distances"][0][i] if results["distances"] else 0
                    similarity_score = 1 / (1 + distance)
                    
                    structured_results.append({
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "score": similarity_score
                    })
            
            return structured_results
        except Exception as e:
            logger.error(f"Error searching in ChromaDB: {e}")
            raise

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            count = self.collection.count()
            return {
                "name": self.collection_name,
                "count": count
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            raise

    def delete_collection(self) -> None:
        """Delete the collection."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception as e:
            logger.error(f"Error deleting collection {self.collection_name}: {e}")
            raise

    def list_collections(self) -> List[str]:
        """List all collections."""
        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            raise

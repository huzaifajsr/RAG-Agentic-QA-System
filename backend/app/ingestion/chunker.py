"""Text chunking functionality."""

from typing import List, Dict, Any

class TextChunker:
    """Splits text into overlapping chunks."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split document contents into overlapping chunks.
        
        Args:
            documents: List of dictionaries with 'content' and 'metadata'.
            
        Returns:
            List of chunk dictionaries with updated metadata.
        """
        chunks = []
        for doc in documents:
            text = doc["content"]
            metadata = doc["metadata"]
            
            text_chunks = self._split_text(text)
            for i, chunk_text in enumerate(text_chunks):
                chunk_meta = metadata.copy()
                chunk_meta["chunk_index"] = i
                chunks.append({
                    "content": chunk_text,
                    "metadata": chunk_meta
                })
        return chunks

    def _split_text(self, text: str) -> List[str]:
        """Internal method to recursively split text.
        
        Args:
            text: The text to split.
            
        Returns:
            List of string chunks.
        """
        if len(text) <= self.chunk_size:
            return [text]

        separators = ["\n\n", "\n", ". ", " "]
        for sep in separators:
            parts = text.split(sep)
            if len(parts) > 1:
                break
        else:
            parts = [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]
            return parts

        chunks = []
        current_chunk = []
        current_length = 0

        for part in parts:
            part_len = len(part) + len(sep)
            if current_length + part_len > self.chunk_size and current_chunk:
                chunks.append(sep.join(current_chunk))
                
                # Overlap logic
                overlap_chunk = []
                overlap_length = 0
                for p in reversed(current_chunk):
                    if overlap_length + len(p) + len(sep) <= self.chunk_overlap:
                        overlap_chunk.insert(0, p)
                        overlap_length += len(p) + len(sep)
                    else:
                        break
                
                current_chunk = overlap_chunk + [part]
                current_length = overlap_length + part_len
            else:
                current_chunk.append(part)
                current_length += part_len

        if current_chunk:
            chunks.append(sep.join(current_chunk))

        return chunks

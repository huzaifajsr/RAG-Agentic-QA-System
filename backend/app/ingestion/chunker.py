"""Text chunking functionality."""

from typing import Any, Dict, List, Optional

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " "]


class TextChunker:
    """Splits text into overlapping chunks no longer than ``chunk_size`` characters.

    Splitting is recursive: the text is split on the coarsest separator present
    (paragraphs, then lines, then sentences, then words). Any piece that is still
    too long is split again with the next, finer separator. A piece with no
    separator left (e.g. one very long token) is cut at fixed character positions.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200,
                 separators: Optional[List[str]] = None):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or DEFAULT_SEPARATORS

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split document contents into overlapping chunks.

        Args:
            documents: List of dictionaries with 'content' and 'metadata'.

        Returns:
            List of chunk dictionaries; each chunk's metadata is a copy of its
            document's metadata plus a 'chunk_index' (position within that document).
        """
        chunks = []
        for doc in documents:
            metadata = doc.get("metadata", {})
            for i, chunk_text in enumerate(self.split_text(doc["content"])):
                chunk_meta = dict(metadata)
                chunk_meta["chunk_index"] = i
                chunks.append({"content": chunk_text, "metadata": chunk_meta})
        return chunks

    def split_text(self, text: str) -> List[str]:
        """Split a single string into chunks of at most ``chunk_size`` characters."""
        text = text.strip()
        if not text:
            return []
        return [c for c in self._split(text, self.separators) if c]

    # ------------------------------------------------------------------ internals

    def _split(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text.strip()]

        # Pick the coarsest separator that actually occurs in this text.
        sep, remaining = None, []
        for i, candidate in enumerate(separators):
            if candidate in text:
                sep, remaining = candidate, separators[i + 1:]
                break
        if sep is None:
            return self._hard_split(text)

        chunks: List[str] = []
        pending: List[str] = []  # pieces small enough to be merged together
        for piece in text.split(sep):
            if not piece.strip():
                continue
            if len(piece) <= self.chunk_size:
                pending.append(piece)
            else:
                chunks.extend(self._merge(pending, sep))
                pending = []
                chunks.extend(self._split(piece, remaining))
        chunks.extend(self._merge(pending, sep))
        return chunks

    def _merge(self, pieces: List[str], sep: str) -> List[str]:
        """Greedily join pieces into chunks, carrying ``chunk_overlap`` characters forward."""
        chunks: List[str] = []
        current: List[str] = []
        length = 0  # length of sep.join(current)

        for piece in pieces:
            added = len(piece) + (len(sep) if current else 0)
            if current and length + added > self.chunk_size:
                chunks.append(sep.join(current).strip())
                # Drop pieces from the front until what is left fits the overlap
                # budget and leaves room for the new piece.
                while current and (
                    length > self.chunk_overlap
                    or length + len(sep) + len(piece) > self.chunk_size
                ):
                    removed = current.pop(0)
                    length -= len(removed) + (len(sep) if current else 0)
                length = max(length, 0)
                added = len(piece) + (len(sep) if current else 0)
            current.append(piece)
            length += added

        if current:
            chunks.append(sep.join(current).strip())
        return chunks

    def _hard_split(self, text: str) -> List[str]:
        step = self.chunk_size - self.chunk_overlap
        chunks = []
        for start in range(0, len(text), step):
            chunks.append(text[start:start + self.chunk_size])
            if start + self.chunk_size >= len(text):
                break
        return chunks

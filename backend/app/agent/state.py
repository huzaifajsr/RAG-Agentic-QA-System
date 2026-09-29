from operator import add
from typing import Annotated, List, TypedDict


class AgentState(TypedDict):
    """State schema for the RAG agent graph."""
    question: str              # Original user question
    refined_question: str      # Current search query used for retrieval
    collection_name: str       # Chroma collection to search
    top_k: int                 # Chunks to fetch per retrieval
    context: List[str]         # Retrieved chunks, accumulated across retrievals (deduplicated)
    sources: List[dict]        # Source metadata, aligned index-for-index with `context`
    reasoning: Annotated[List[str], add]  # Agent's reasoning trace (append-only)
    answer: str                # Final generated answer
    needs_more_context: bool   # Whether agent needs another retrieval pass
    iteration: int             # Number of retrievals performed so far
    max_iterations: int        # Maximum number of retrievals allowed

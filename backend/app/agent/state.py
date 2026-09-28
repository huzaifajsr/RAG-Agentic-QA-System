from typing import TypedDict, List, Annotated
from operator import add

class AgentState(TypedDict):
    """State schema for the RAG agent graph."""
    question: str              # Original user question
    refined_question: str      # Refined/expanded question for better retrieval
    context: List[str]         # Retrieved document chunks
    sources: List[dict]        # Source metadata for citations
    reasoning: Annotated[List[str], add]  # Agent's reasoning trace (append-only)
    answer: str                # Final generated answer
    needs_more_context: bool   # Whether agent needs another retrieval pass
    iteration: int             # Current iteration count
    max_iterations: int        # Maximum allowed iterations

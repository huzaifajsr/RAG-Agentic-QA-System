from langgraph.graph import END, START, StateGraph

from app.agent.nodes import create_nodes
from app.agent.state import AgentState
from app.config import Settings


class LLMNotConfiguredError(Exception):
    """Raised when the selected LLM provider has no API key."""


def build_llm(settings: Settings):
    provider = settings.LLM_PROVIDER.lower()
    if provider == "google":
        if not settings.GOOGLE_API_KEY:
            raise LLMNotConfiguredError("GOOGLE_API_KEY is not set")
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=settings.LLM_TEMPERATURE,
        )
    if provider == "openai":
        if not settings.OPENAI_API_KEY:
            raise LLMNotConfiguredError("OPENAI_API_KEY is not set")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=settings.LLM_TEMPERATURE,
        )
    raise LLMNotConfiguredError(f"Unsupported LLM_PROVIDER: {settings.LLM_PROVIDER}")


def should_continue(state: AgentState) -> str:
    """Loop back for another retrieval only while context is insufficient and budget remains.

    `iteration` counts retrievals, so MAX_AGENT_ITERATIONS=3 allows exactly 3 retrievals.
    """
    if state["needs_more_context"] and state["iteration"] < state["max_iterations"]:
        return "refine_query"
    return "generate_answer"


def build_agent_graph(chroma_store, llm):
    """Compile the agent graph. Built once at startup and reused for every query."""
    workflow = StateGraph(AgentState)
    for name, fn in create_nodes(chroma_store, llm).items():
        workflow.add_node(name, fn)

    workflow.add_edge(START, "analyze_query")
    workflow.add_edge("analyze_query", "retrieve")
    workflow.add_edge("retrieve", "evaluate_context")
    workflow.add_conditional_edges(
        "evaluate_context",
        should_continue,
        {"refine_query": "refine_query", "generate_answer": "generate_answer"},
    )
    workflow.add_edge("refine_query", "retrieve")
    workflow.add_edge("generate_answer", END)
    return workflow.compile()


def run_agent_query(graph, question: str, collection_name: str, top_k: int, max_iterations: int) -> dict:
    initial_state: AgentState = {
        "question": question,
        "refined_question": question,
        "collection_name": collection_name,
        "top_k": top_k,
        "context": [],
        "sources": [],
        "reasoning": [],
        "answer": "",
        "needs_more_context": False,
        "iteration": 0,
        "max_iterations": max_iterations,
    }
    return graph.invoke(initial_state)

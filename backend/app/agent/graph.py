from langgraph.graph import StateGraph, START, END
from app.config import Settings
from app.vectorstore.chroma_store import ChromaStore
from app.agent.state import AgentState
from app.agent.nodes import create_nodes

def build_agent_graph(chroma_store: ChromaStore, llm, settings: Settings):
    workflow = StateGraph(AgentState)
    nodes = create_nodes(chroma_store, llm, settings)
    
    workflow.add_node("analyze_query", nodes["analyze_query"])
    workflow.add_node("retrieve", nodes["retrieve"])
    workflow.add_node("evaluate_context", nodes["evaluate_context"])
    workflow.add_node("refine_query", nodes["refine_query"])
    workflow.add_node("generate_answer", nodes["generate_answer"])
    
    workflow.add_edge(START, "analyze_query")
    workflow.add_edge("analyze_query", "retrieve")
    workflow.add_edge("retrieve", "evaluate_context")
    
    def should_continue(state: AgentState) -> str:
        if state["needs_more_context"] and state["iteration"] < state["max_iterations"]:
            return "refine_query"
        return "generate_answer"
        
    workflow.add_conditional_edges(
        "evaluate_context",
        should_continue,
        {
            "refine_query": "refine_query",
            "generate_answer": "generate_answer"
        }
    )
    workflow.add_edge("refine_query", "retrieve")
    workflow.add_edge("generate_answer", END)
    
    return workflow.compile()

def run_agent_query(question: str, chroma_store: ChromaStore, settings: Settings) -> dict:
    if settings.LLM_PROVIDER.lower() == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL, 
            google_api_key=settings.GOOGLE_API_KEY, 
            temperature=settings.LLM_TEMPERATURE
        )
    elif settings.LLM_PROVIDER.lower() == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=settings.LLM_MODEL, 
            api_key=settings.OPENAI_API_KEY, 
            temperature=settings.LLM_TEMPERATURE
        )
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {settings.LLM_PROVIDER}")

    graph = build_agent_graph(chroma_store, llm, settings)
    
    initial_state = {
        "question": question,
        "refined_question": "",
        "context": [],
        "sources": [],
        "reasoning": [],
        "answer": "",
        "needs_more_context": False,
        "iteration": 0,
        "max_iterations": settings.MAX_AGENT_ITERATIONS
    }
    
    final_state = graph.invoke(initial_state)
    return final_state

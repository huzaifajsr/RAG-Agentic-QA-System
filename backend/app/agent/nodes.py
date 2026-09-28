from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from app.config import Settings
from app.vectorstore.chroma_store import ChromaStore
from app.agent.state import AgentState
from app.agent.tools import retrieve_documents, format_context

def create_nodes(chroma_store: ChromaStore, llm: BaseChatModel, settings: Settings) -> dict:
    top_k = settings.RETRIEVAL_TOP_K
    
    def analyze_query(state: AgentState) -> dict:
        prompt = f"Given this question, generate an optimized search query that would help find relevant information in a document collection. Return ONLY the search query, nothing else. Question: {state['question']}"
        response = llm.invoke([HumanMessage(content=prompt)])
        refined_query = response.content.strip()
        return {
            "refined_question": refined_query,
            "reasoning": [f"Analyzed query: Generated optimized search query '{refined_query}'"],
            "iteration": 1
        }
    
    def retrieve(state: AgentState) -> dict:
        results = retrieve_documents(state['refined_question'], chroma_store, top_k=top_k)
        docs = results["documents"]
        return {
            "context": docs,
            "sources": results["sources"],
            "reasoning": [f"Retrieved {len(docs)} documents"]
        }
    
    def evaluate_context(state: AgentState) -> dict:
        formatted = format_context(state['context'])
        prompt = f"Given this question and context, can you provide a comprehensive answer? Respond with ONLY 'SUFFICIENT' or 'INSUFFICIENT'.\n\nQuestion: {state['question']}\nContext: {formatted}"
        response = llm.invoke([HumanMessage(content=prompt)])
        needs_more = "INSUFFICIENT" in response.content.upper()
        return {
            "needs_more_context": needs_more,
            "reasoning": [f"Context evaluation: {'INSUFFICIENT' if needs_more else 'SUFFICIENT'}"],
            "iteration": state["iteration"] + 1
        }
    
    def refine_query(state: AgentState) -> dict:
        prompt = f"The previous search didn't find enough information. Generate a DIFFERENT search query to find more relevant information.\n\nOriginal question: {state['question']}\nPrevious query: {state['refined_question']}\nReturn ONLY the new search query."
        response = llm.invoke([HumanMessage(content=prompt)])
        new_query = response.content.strip()
        return {
            "refined_question": new_query,
            "reasoning": [f"Refined query: Generated new search query '{new_query}'"]
        }
    
    def generate_answer(state: AgentState) -> dict:
        formatted = format_context(state['context'])
        prompt = f"Answer the following question based ONLY on the provided context. If the context doesn't contain enough information, say so clearly. Include specific references to source documents.\n\nQuestion: {state['question']}\n\nContext:\n{formatted}\n\nProvide a comprehensive, well-structured answer:"
        response = llm.invoke([HumanMessage(content=prompt)])
        return {
            "answer": response.content,
            "reasoning": ["Generated final answer"]
        }
        
    return {
        "analyze_query": analyze_query,
        "retrieve": retrieve,
        "evaluate_context": evaluate_context,
        "refine_query": refine_query,
        "generate_answer": generate_answer,
    }

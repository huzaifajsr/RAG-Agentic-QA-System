from langchain_core.messages import HumanMessage

from app.agent.state import AgentState
from app.agent.tools import format_context, merge_context, retrieve_documents


def _text(response) -> str:
    """Return a chat model response as plain text."""
    content = response.content
    if isinstance(content, list):  # some providers return content blocks
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content).strip()


def create_nodes(chroma_store, llm) -> dict:

    def analyze_query(state: AgentState) -> dict:
        prompt = (
            "Given this question, generate an optimized search query that would help find "
            "relevant information in a document collection. Return ONLY the search query, "
            f"nothing else. Question: {state['question']}"
        )
        refined_query = _text(llm.invoke([HumanMessage(content=prompt)])) or state["question"]
        return {
            "refined_question": refined_query,
            "reasoning": [f"Analyzed query: generated search query '{refined_query}'"],
        }

    def retrieve(state: AgentState) -> dict:
        results = retrieve_documents(
            state["refined_question"], chroma_store,
            top_k=state["top_k"], collection_name=state["collection_name"],
        )
        docs, sources, added = merge_context(
            state["context"], state["sources"], results["documents"], results["sources"]
        )
        iteration = state["iteration"] + 1
        return {
            "context": docs,
            "sources": sources,
            "iteration": iteration,
            "reasoning": [
                f"Retrieval {iteration}/{state['max_iterations']}: fetched "
                f"{len(results['documents'])} chunks, {added} new (total {len(docs)})"
            ],
        }

    def evaluate_context(state: AgentState) -> dict:
        if not state["context"]:
            needs_more = True
        else:
            formatted = format_context(state["context"], state["sources"])
            prompt = (
                "Given this question and context, can you provide a comprehensive answer? "
                "Respond with ONLY 'SUFFICIENT' or 'INSUFFICIENT'.\n\n"
                f"Question: {state['question']}\nContext: {formatted}"
            )
            verdict = _text(llm.invoke([HumanMessage(content=prompt)])).upper()
            needs_more = "INSUFFICIENT" in verdict
        return {
            "needs_more_context": needs_more,
            "reasoning": [f"Context evaluation: {'INSUFFICIENT' if needs_more else 'SUFFICIENT'}"],
        }

    def refine_query(state: AgentState) -> dict:
        prompt = (
            "The previous search didn't find enough information. Generate a DIFFERENT search "
            "query to find more relevant information.\n\n"
            f"Original question: {state['question']}\n"
            f"Previous query: {state['refined_question']}\n"
            "Return ONLY the new search query."
        )
        new_query = _text(llm.invoke([HumanMessage(content=prompt)])) or state["question"]
        return {
            "refined_question": new_query,
            "reasoning": [f"Refined query: generated new search query '{new_query}'"],
        }

    def generate_answer(state: AgentState) -> dict:
        if not state["context"]:
            return {
                "answer": "I couldn't find any relevant information in the uploaded documents "
                          "to answer this question.",
                "reasoning": ["No relevant context found; skipped answer generation"],
            }
        formatted = format_context(state["context"], state["sources"])
        prompt = (
            "Answer the following question based ONLY on the provided context. If the context "
            "doesn't contain enough information, say so clearly. Cite sources using the "
            "document name and page shown in each [Document] header.\n\n"
            f"Question: {state['question']}\n\nContext:\n{formatted}\n\n"
            "Provide a comprehensive, well-structured answer:"
        )
        return {
            "answer": _text(llm.invoke([HumanMessage(content=prompt)])),
            "reasoning": ["Generated final answer"],
        }

    return {
        "analyze_query": analyze_query,
        "retrieve": retrieve,
        "evaluate_context": evaluate_context,
        "refine_query": refine_query,
        "generate_answer": generate_answer,
    }

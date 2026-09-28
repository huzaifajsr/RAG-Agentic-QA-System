from app.vectorstore.chroma_store import ChromaStore

def retrieve_documents(query: str, chroma_store: ChromaStore, top_k: int = 5) -> dict:
    results = chroma_store.search(query, k=top_k)
    documents = [res["content"] for res in results]
    sources = [{
        "content": res["content"],
        "source": res.get("metadata", {}).get("source", "Unknown"),
        "page": res.get("metadata", {}).get("page"),
        "score": res.get("score", 0.0)
    } for res in results]
    return {"documents": documents, "sources": sources}

def format_context(documents: list[str]) -> str:
    formatted = []
    for i, doc in enumerate(documents, 1):
        formatted.append(f"[Document {i}]\n{doc}")
    return "\n\n".join(formatted)

from typing import List, Optional


def retrieve_documents(query: str, chroma_store, top_k: int = 5,
                       collection_name: Optional[str] = None) -> dict:
    results = chroma_store.search(query, k=top_k, collection_name=collection_name)
    documents = [res["content"] for res in results]
    sources = [{
        "content": res["content"],
        "source": res.get("metadata", {}).get("source", "Unknown"),
        "page": res.get("metadata", {}).get("page"),
        "score": res.get("score", 0.0),
    } for res in results]
    return {"documents": documents, "sources": sources}


def merge_context(old_docs: List[str], old_sources: List[dict],
                  new_docs: List[str], new_sources: List[dict]) -> tuple:
    """Append newly retrieved chunks to the existing context, skipping duplicates.

    Keeps earlier results, so a refined query that retrieves worse chunks can
    never throw away the good chunks found by an earlier query.
    """
    docs, sources = list(old_docs), list(old_sources)
    seen = set(docs)
    added = 0
    for doc, src in zip(new_docs, new_sources):
        if doc not in seen:
            seen.add(doc)
            docs.append(doc)
            sources.append(src)
            added += 1
    return docs, sources, added


def format_context(documents: List[str], sources: Optional[List[dict]] = None) -> str:
    formatted = []
    for i, doc in enumerate(documents, 1):
        header = f"[Document {i}]"
        if sources and i - 1 < len(sources):
            src = sources[i - 1]
            header += f" (source: {src.get('source', 'Unknown')}"
            if src.get("page") is not None:
                header += f", page {src['page']}"
            header += ")"
        formatted.append(f"{header}\n{doc}")
    return "\n\n".join(formatted)

import pytest

from app.vectorstore.chroma_store import CollectionNotFoundError


def chunks(*texts, source="doc.txt"):
    return [{"content": t, "metadata": {"source": source, "chunk_index": i}} for i, t in enumerate(texts)]


def test_add_and_search_returns_most_similar_first(store):
    store.add_documents(chunks("cats purr and sleep", "rockets launch into orbit", "stock markets fell"))
    results = store.search("rockets orbit", k=2)
    assert len(results) == 2
    assert results[0]["content"] == "rockets launch into orbit"
    assert results[0]["score"] >= results[1]["score"]


def test_search_k_larger_than_collection(store):
    store.add_documents(chunks("only one chunk"))
    assert len(store.search("chunk", k=10)) == 1


def test_reupload_does_not_duplicate(store):
    store.add_documents(chunks("a b c", "d e f"))
    store.add_documents(chunks("a b c", "d e f"))
    assert store.get_collection_info()["count"] == 2


def test_upload_after_delete_works(store):
    """Regression: deleting a collection left a stale handle and broke later uploads."""
    store.add_documents(chunks("first"))
    store.delete_collection("documents")
    store.add_documents(chunks("second"))
    assert store.get_collection_info("documents")["count"] == 1


def test_delete_only_named_collection(store):
    store.add_documents(chunks("keep me"), collection_name="keep")
    store.add_documents(chunks("delete me"), collection_name="drop")
    store.delete_collection("drop")
    names = {c["name"] for c in store.list_collections()}
    assert "keep" in names and "drop" not in names


def test_list_collections_reports_real_counts(store):
    store.add_documents(chunks("one", "two", "three"), collection_name="three-chunks")
    info = {c["name"]: c["count"] for c in store.list_collections()}
    assert info["three-chunks"] == 3


def test_missing_collection_raises(store):
    with pytest.raises(CollectionNotFoundError):
        store.search("anything", collection_name="does-not-exist")
    with pytest.raises(CollectionNotFoundError):
        store.delete_collection("does-not-exist")

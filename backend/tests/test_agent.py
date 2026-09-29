import pytest

from app.agent.graph import LLMNotConfiguredError, build_agent_graph, build_llm, run_agent_query
from app.config import Settings
from tests.conftest import FakeLLM

DOCS = [
    ("Cats sleep for up to sixteen hours a day.", 1),
    ("Rockets need escape velocity to leave orbit.", 2),
    ("Refined query results mention rockets and fuel.", 3),
    ("Bananas are rich in potassium.", 4),
]


@pytest.fixture
def filled_store(store):
    store.add_documents([
        {"content": text, "metadata": {"source": "facts.pdf", "page": page, "chunk_index": 0}}
        for text, page in DOCS
    ])
    return store


def run(store, llm, max_iterations=3, top_k=1, collection="documents"):
    graph = build_agent_graph(store, llm)
    return run_agent_query(graph, "How do rockets leave orbit?", collection, top_k, max_iterations)


def test_sufficient_context_retrieves_once(filled_store):
    out = run(filled_store, FakeLLM(verdicts=["SUFFICIENT"]))
    assert out["iteration"] == 1
    assert out["answer"] == "Fake answer [Document 1]"


@pytest.mark.parametrize("max_iterations", [1, 2, 3, 4])
def test_max_iterations_is_number_of_retrievals(filled_store, max_iterations):
    """Regression: MAX_AGENT_ITERATIONS=3 used to allow only 2 retrievals."""
    out = run(filled_store, FakeLLM(verdicts=["INSUFFICIENT"]), max_iterations=max_iterations)
    assert out["iteration"] == max_iterations
    retrieval_steps = [r for r in out["reasoning"] if r.startswith("Retrieval ")]
    assert len(retrieval_steps) == max_iterations


def test_context_accumulates_across_retrievals(filled_store):
    """Regression: each retrieval used to replace the context, dropping earlier chunks."""
    first_pass = run(filled_store, FakeLLM(verdicts=["SUFFICIENT"]), top_k=2)["context"]
    out = run(filled_store, FakeLLM(verdicts=["INSUFFICIENT", "SUFFICIENT"]), top_k=2)
    assert out["iteration"] == 2
    assert set(first_pass) <= set(out["context"])            # earlier chunks kept
    assert len(out["context"]) == len(set(out["context"]))  # no duplicates
    assert len(out["sources"]) == len(out["context"])        # stays aligned
    assert [s["content"] for s in out["sources"]] == out["context"]


def test_answer_prompt_includes_source_and_page(filled_store):
    llm = FakeLLM()
    run(filled_store, llm)
    answer_prompt = llm.prompts[-1]
    assert "source: facts.pdf, page" in answer_prompt


def test_empty_collection_skips_llm_answer(store):
    store.client.get_or_create_collection("empty-col")
    llm = FakeLLM()
    out = run(store, llm, collection="empty-col", max_iterations=2)
    assert out["iteration"] == 2
    assert "couldn't find any relevant information" in out["answer"]
    assert not any("Answer the following question" in p for p in llm.prompts)


def test_missing_api_key_is_reported():
    with pytest.raises(LLMNotConfiguredError):
        build_llm(Settings(_env_file=None, LLM_PROVIDER="google", GOOGLE_API_KEY=""))
    with pytest.raises(LLMNotConfiguredError):
        build_llm(Settings(_env_file=None, LLM_PROVIDER="anthropic-typo"))

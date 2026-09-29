import io
import os

import pytest
from fastapi.testclient import TestClient

from app.agent.graph import LLMNotConfiguredError, build_agent_graph
from app.api import dependencies
from app.config import Settings
from app.main import app
from tests.conftest import FakeLLM


@pytest.fixture
def client(store, tmp_path):
    settings = Settings(_env_file=None, UPLOAD_DIR=str(tmp_path / "uploads"), MAX_FILE_SIZE_MB=1,
                        CHUNK_SIZE=200, CHUNK_OVERLAP=20)
    llm = FakeLLM()
    app.dependency_overrides[dependencies.get_settings] = lambda: settings
    app.dependency_overrides[dependencies.get_chroma_store] = lambda: store
    app.dependency_overrides[dependencies.get_agent_graph] = lambda: build_agent_graph(store, llm)
    c = TestClient(app)
    c.upload_dir = settings.UPLOAD_DIR
    c.llm = llm
    yield c
    app.dependency_overrides.clear()


def upload(client, name="notes.txt", body=b"Rockets need escape velocity to leave orbit.", collection=None):
    data = {"collection_name": collection} if collection else {}
    return client.post("/upload", files={"file": (name, io.BytesIO(body), "text/plain")}, data=data)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "healthy"


def test_upload_no_file(client):
    assert client.post("/upload").status_code == 422


def test_query_empty_question(client):
    assert client.post("/query", json={"question": ""}).status_code == 422


def test_query_top_k_out_of_range(client):
    assert client.post("/query", json={"question": "hi", "top_k": 500}).status_code == 422


def test_upload_then_query(client):
    r = upload(client)
    assert r.status_code == 200, r.text
    assert r.json()["num_chunks"] == 1 and r.json()["collection_name"] == "documents"

    r = client.post("/query", json={"question": "How do rockets leave orbit?"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"] == "Fake answer [Document 1]"
    assert body["retrievals"] == 1
    assert body["sources"][0]["source"] == "notes.txt"


def test_uppercase_extension_accepted(client):
    assert upload(client, name="NOTES.TXT").status_code == 200


def test_unsupported_extension_rejected(client):
    assert upload(client, name="evil.exe").status_code == 400


def test_path_traversal_filename_stays_in_upload_dir(client, tmp_path):
    r = upload(client, name="../../outside.txt")
    assert r.status_code == 200, r.text
    assert r.json()["filename"] == "outside.txt"
    assert os.path.exists(os.path.join(client.upload_dir, "outside.txt"))
    assert not (tmp_path / "outside.txt").exists()


def test_oversized_file_rejected_and_not_kept(client):
    r = upload(client, name="big.txt", body=b"a " * (700 * 1024))  # ~1.4 MB > 1 MB limit
    assert r.status_code == 413
    assert not os.path.exists(os.path.join(client.upload_dir, "big.txt"))


def test_empty_file_rejected(client):
    assert upload(client, name="blank.txt", body=b"   ").status_code == 422


def test_collection_and_top_k_are_used(client):
    upload(client, name="a.txt", body=b"alpha facts about rockets", collection="team-a")
    upload(client, name="b.txt", body=b"beta facts about bananas", collection="team-b")

    r = client.post("/query", json={"question": "rockets", "collection_name": "team-a", "top_k": 1})
    assert r.status_code == 200, r.text
    assert [s["source"] for s in r.json()["sources"]] == ["a.txt"]


def test_query_unknown_collection_404(client):
    r = client.post("/query", json={"question": "hi", "collection_name": "nope-nope"})
    assert r.status_code == 404


def test_collections_list_counts_and_delete(client):
    upload(client, name="a.txt", body=b"one\n\n" + b"two " * 100, collection="counted")
    listing = {c["name"]: c["num_chunks"] for c in client.get("/collections").json()["collections"]}
    assert listing["counted"] >= 2

    assert client.get("/collections/counted").json()["num_chunks"] == listing["counted"]
    assert client.delete("/collections/counted").status_code == 200
    assert client.get("/collections/counted").status_code == 404
    assert client.delete("/collections/counted").status_code == 404
    # Regression: uploads used to fail after a delete.
    assert upload(client, collection="counted").status_code == 200


def test_invalid_collection_name(client):
    assert upload(client, collection="../x").status_code == 400


def test_missing_llm_key_returns_503(client, monkeypatch):
    del app.dependency_overrides[dependencies.get_agent_graph]

    def boom():
        raise LLMNotConfiguredError("GOOGLE_API_KEY is not set")

    monkeypatch.setattr(dependencies, "_agent_graph", boom)
    r = client.post("/query", json={"question": "hi"})
    assert r.status_code == 503
    assert "GOOGLE_API_KEY" in r.json()["detail"]

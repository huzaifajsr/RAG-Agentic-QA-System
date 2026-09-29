"""Shared test fixtures.

Tests never download the embedding model or call a real LLM: FakeEmbedding
turns text into a deterministic bag-of-words vector, and FakeLLM returns
scripted replies. ChromaDB runs in memory.
"""

import hashlib
import uuid

import numpy as np
import pytest

from app.vectorstore.chroma_store import ChromaStore

DIM = 64


class FakeEmbedding:
    """Hashes words into a fixed-size vector, so texts sharing words are close."""

    def _vec(self, text: str):
        v = np.zeros(DIM)
        for word in text.lower().split():
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = np.linalg.norm(v)
        return (v / norm if norm else v).tolist()

    def embed_texts(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, query):
        return self._vec(query)


class Reply:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    """Answers each prompt type from a script and records every prompt it saw."""

    def __init__(self, verdicts=("SUFFICIENT",), answer="Fake answer [Document 1]"):
        self.verdicts = list(verdicts)
        self.answer = answer
        self.prompts = []

    def invoke(self, messages):
        prompt = messages[0].content
        self.prompts.append(prompt)
        if "Respond with ONLY 'SUFFICIENT' or 'INSUFFICIENT'" in prompt:
            verdict = self.verdicts.pop(0) if len(self.verdicts) > 1 else self.verdicts[0]
            return Reply(verdict)
        if "Generate a DIFFERENT search query" in prompt:
            return Reply(f"refined query {len(self.prompts)}")
        if "generate an optimized search query" in prompt:
            return Reply("optimized query")
        return Reply(self.answer)


@pytest.fixture
def store():
    s = ChromaStore(persist_dir=None, default_collection="documents", embedding_service=FakeEmbedding())
    yield s
    # EphemeralClient instances share one in-memory backend: clean up between tests.
    for info in s.list_collections():
        s.delete_collection(info["name"])


@pytest.fixture
def unique_name():
    return f"test-{uuid.uuid4().hex[:8]}"

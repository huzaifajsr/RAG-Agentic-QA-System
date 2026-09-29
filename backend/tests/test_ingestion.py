import pytest

from app.ingestion.chunker import TextChunker
from app.ingestion.loader import DocumentLoader


def make_doc(text, **meta):
    return [{"content": text, "metadata": meta or {"source": "doc.txt"}}]


def test_chunker_splits_and_adds_metadata():
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    text = "This is a simple text document meant to be chunked into multiple pieces for testing."
    chunks = chunker.chunk_documents(make_doc(text, source="a.txt", page=3))

    assert len(chunks) > 1
    assert [c["metadata"]["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert all(c["metadata"]["source"] == "a.txt" and c["metadata"]["page"] == 3 for c in chunks)


def test_small_doc_is_one_chunk():
    chunks = TextChunker(chunk_size=1000, chunk_overlap=100).chunk_documents(make_doc("Small document."))
    assert len(chunks) == 1
    assert chunks[0]["content"] == "Small document."


@pytest.mark.parametrize("text", [
    "word " * 600 + "\n\nshort para",                 # regression: 3000-char paragraph was one chunk
    ("Sentence number one is here. " * 80),            # sentences only
    "line of text\n" * 300,                            # lines only
    "A" * 2500,                                         # no separators at all
    ("para " * 50 + "\n\n") * 10 + "x" * 1500,          # mix of short paras and one long token
])
def test_no_chunk_exceeds_chunk_size(text):
    chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
    chunks = chunker.split_text(text)
    assert chunks, "expected at least one chunk"
    assert max(len(c) for c in chunks) <= 1000


def test_no_text_is_lost():
    words = [f"w{i}" for i in range(2000)]
    chunks = TextChunker(chunk_size=300, chunk_overlap=50).split_text(" ".join(words))
    seen = set(" ".join(chunks).split())
    assert seen == set(words)


def test_consecutive_chunks_overlap():
    words = [f"w{i}" for i in range(400)]
    chunks = TextChunker(chunk_size=200, chunk_overlap=50).split_text(" ".join(words))
    assert len(chunks) > 2
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev.split()[-1] in nxt.split(), "last word of a chunk should reappear in the next"


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, chunk_overlap=100)


def test_loader_txt_returns_list_with_basename(tmp_path):
    f = tmp_path / "test_doc.txt"
    f.write_text("Hello world! This is a test file.")
    docs = DocumentLoader().load(str(f))

    assert isinstance(docs, list) and len(docs) == 1
    assert docs[0]["content"] == "Hello world! This is a test file."
    assert docs[0]["metadata"]["source"] == "test_doc.txt"


def test_loader_empty_txt_returns_nothing(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("   \n  ")
    assert DocumentLoader().load(str(f)) == []


def test_loader_unsupported():
    with pytest.raises(ValueError):
        DocumentLoader().load("unsupported_file.xyz")

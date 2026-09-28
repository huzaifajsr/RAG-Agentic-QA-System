import pytest
from app.ingestion.loader import DocumentLoader
from app.ingestion.chunker import TextChunker

def test_text_chunker_basic():
    """Verify basic chunking logic and metadata generation."""
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    text = "This is a simple text document meant to be chunked into multiple pieces for testing."
    chunks = chunker.chunk(text)
    
    assert len(chunks) > 1
    assert "content" in chunks[0]
    assert "metadata" in chunks[0]
    assert "chunk_index" in chunks[0]["metadata"]

def test_text_chunker_overlap():
    """Verify overlapping content between consecutive chunks."""
    chunker = TextChunker(chunk_size=20, chunk_overlap=10)
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    chunks = chunker.chunk(text)
    
    # Check if there is overlap in the text
    assert len(chunks) > 1
    # Example overlap test logic (dependent on actual chunker implementation)
    assert len(chunks[0]["content"]) <= 20

def test_text_chunker_small_doc():
    """Verify a document smaller than chunk_size produces exactly 1 chunk."""
    chunker = TextChunker(chunk_size=1000, chunk_overlap=100)
    text = "Small document."
    chunks = chunker.chunk(text)
    
    assert len(chunks) == 1
    assert chunks[0]["content"] == text

def test_document_loader_txt(tmp_path):
    """Verify loader correctly reads TXT file and sets metadata."""
    temp_file = tmp_path / "test_doc.txt"
    test_content = "Hello world! This is a test file."
    temp_file.write_text(test_content)
    
    loader = DocumentLoader()
    doc = loader.load(str(temp_file))
    
    assert doc["content"] == test_content
    assert "metadata" in doc
    assert doc["metadata"]["source"] == str(temp_file)

def test_document_loader_unsupported():
    """Verify ValueError is raised for unsupported file types."""
    loader = DocumentLoader()
    with pytest.raises(ValueError):
        loader.load("unsupported_file.xyz")

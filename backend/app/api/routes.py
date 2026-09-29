"""API routes.

Routes are plain `def` (not `async def`) because they call blocking code
(embedding model, ChromaDB, LLM HTTP calls). FastAPI runs `def` routes in a
thread pool, so one slow query no longer freezes the whole server.
"""

import os
import re
import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.agent.graph import run_agent_query
from app.api.dependencies import get_agent_graph, get_chroma_store, get_settings
from app.config import Settings
from app.ingestion import DocumentLoader, TextChunker
from app.models import (
    CollectionInfo, CollectionListResponse, DeleteResponse, DocumentUploadResponse,
    HealthResponse, QueryRequest, QueryResponse, SourceDocument,
)
from app.vectorstore.chroma_store import ChromaStore, CollectionNotFoundError

router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".txt"}
COLLECTION_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,61}[A-Za-z0-9]$")
_READ_BLOCK = 1024 * 1024  # 1 MB


def safe_filename(filename: Optional[str]) -> str:
    """Strip directories and unsafe characters so uploads can't escape UPLOAD_DIR."""
    name = os.path.basename((filename or "").replace("\\", "/"))
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name).lstrip(".")
    return name[:200]


def validate_collection_name(name: str) -> str:
    if not COLLECTION_NAME_RE.match(name) or ".." in name:
        raise HTTPException(
            status_code=400,
            detail="Collection name must be 3-63 characters: letters, digits, '.', '_' or '-', "
                   "starting and ending with a letter or digit.",
        )
    return name


@router.post("/upload", response_model=DocumentUploadResponse, tags=["documents"])
def upload_document(
    file: UploadFile = File(...),
    collection_name: Optional[str] = Form(None),
    settings: Settings = Depends(get_settings),
    chroma_store: ChromaStore = Depends(get_chroma_store),
):
    collection = validate_collection_name(collection_name or settings.CHROMA_COLLECTION)
    filename = safe_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()
    if not filename or ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")

    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(settings.UPLOAD_DIR, filename)
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024

    try:
        written = 0
        with open(file_path, "wb") as buffer:
            while block := file.file.read(_READ_BLOCK):
                written += len(block)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File is larger than the {settings.MAX_FILE_SIZE_MB} MB limit",
                    )
                buffer.write(block)

        try:
            documents = DocumentLoader().load(file_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not read file: {e}") from e
        if not documents:
            raise HTTPException(
                status_code=422,
                detail="No extractable text found (scanned/image-only PDFs need OCR first)",
            )

        chunker = TextChunker(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
        chunks = chunker.chunk_documents(documents)
        chroma_store.add_documents(chunks, collection_name=collection)
    except Exception:
        # Don't leave half-processed or rejected files on disk.
        if os.path.exists(file_path):
            os.remove(file_path)
        raise

    return DocumentUploadResponse(
        filename=filename,
        collection_name=collection,
        num_chunks=len(chunks),
        message="Document uploaded and indexed successfully",
    )


@router.post("/query", response_model=QueryResponse, tags=["query"])
def query_documents(
    request: QueryRequest,
    settings: Settings = Depends(get_settings),
    graph=Depends(get_agent_graph),
):
    collection = validate_collection_name(request.collection_name or settings.CHROMA_COLLECTION)
    start_time = time.perf_counter()
    try:
        final_state = run_agent_query(
            graph,
            question=request.question,
            collection_name=collection,
            top_k=request.top_k or settings.RETRIEVAL_TOP_K,
            max_iterations=settings.MAX_AGENT_ITERATIONS,
        )
    except CollectionNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{collection}' not found. Upload a document to it first.",
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Agent failed: {e}") from e

    sources = [
        SourceDocument(
            content=src["content"],
            source=src["source"],
            page=src.get("page"),
            relevance_score=round(src.get("score", 0.0), 4),
        )
        for src in final_state.get("sources", [])
    ]
    return QueryResponse(
        answer=final_state["answer"],
        sources=sources,
        agent_reasoning=final_state["reasoning"],
        retrievals=final_state["iteration"],
        query_time_seconds=round(time.perf_counter() - start_time, 3),
    )


@router.get("/collections", response_model=CollectionListResponse, tags=["collections"])
def list_collections(chroma_store: ChromaStore = Depends(get_chroma_store)):
    return CollectionListResponse(
        collections=[
            CollectionInfo(name=c["name"], num_chunks=c["count"])
            for c in chroma_store.list_collections()
        ]
    )


@router.get("/collections/{name}", response_model=CollectionInfo, tags=["collections"])
def get_collection_info(name: str, chroma_store: ChromaStore = Depends(get_chroma_store)):
    try:
        info = chroma_store.get_collection_info(name)
    except CollectionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
    return CollectionInfo(name=info["name"], num_chunks=info["count"])


@router.delete("/collections/{name}", response_model=DeleteResponse, tags=["collections"])
def delete_collection(name: str, chroma_store: ChromaStore = Depends(get_chroma_store)):
    try:
        chroma_store.delete_collection(name)
    except CollectionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
    return DeleteResponse(message="Collection deleted successfully", collection_name=name)


@router.get("/health", response_model=HealthResponse, tags=["health"])
def health_check():
    return HealthResponse(
        status="healthy",
        version="1.1.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

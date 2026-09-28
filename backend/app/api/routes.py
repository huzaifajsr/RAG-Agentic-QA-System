import os
import time
import shutil
from datetime import datetime
from typing import List
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from app.config import Settings
from app.models import (
    QueryRequest, QueryResponse, SourceDocument,
    DocumentUploadResponse, CollectionListResponse, CollectionInfo,
    DeleteResponse, HealthResponse
)
from app.api.dependencies import get_settings, get_chroma_store
from app.vectorstore.chroma_store import ChromaStore
from app.agent.graph import run_agent_query
from app.ingestion import DocumentLoader, TextChunker

router = APIRouter()

@router.post("/upload", response_model=DocumentUploadResponse, tags=["documents"])
async def upload_document(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    chroma_store: ChromaStore = Depends(get_chroma_store)
):
    if not file.filename.endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")
        
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(settings.UPLOAD_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    loader = DocumentLoader()
    documents = loader.load(file_path)
    
    chunker = TextChunker(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
    chunks = chunker.chunk_documents(documents)
    
    chroma_store.add_documents(chunks)
    
    return DocumentUploadResponse(
        filename=file.filename,
        num_chunks=len(chunks),
        message="Document uploaded and indexed successfully"
    )

@router.post("/query", response_model=QueryResponse, tags=["query"])
async def query_documents(
    request: QueryRequest,
    settings: Settings = Depends(get_settings),
    chroma_store: ChromaStore = Depends(get_chroma_store)
):
    start_time = time.time()
    
    try:
        final_state = run_agent_query(request.question, chroma_store, settings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    query_time = time.time() - start_time
    
    sources = [
        SourceDocument(
            content=src["content"],
            source=src["source"],
            page=src.get("page"),
            relevance_score=src.get("score", 0.0)
        )
        for src in final_state.get("sources", [])
    ]
    
    return QueryResponse(
        answer=final_state["answer"],
        sources=sources,
        agent_reasoning=final_state["reasoning"],
        query_time_seconds=query_time
    )

@router.get("/collections", response_model=CollectionListResponse, tags=["collections"])
async def list_collections(chroma_store: ChromaStore = Depends(get_chroma_store)):
    collections = chroma_store.list_collections()
    # Mock returning collection info based on chroma store structure
    return CollectionListResponse(
        collections=[CollectionInfo(name=name, num_documents=0) for name in collections]
    )

@router.get("/collections/{name}", response_model=CollectionInfo, tags=["collections"])
async def get_collection_info(
    name: str, 
    chroma_store: ChromaStore = Depends(get_chroma_store)
):
    # This might require chroma_store to switch collections, but using simple representation for now
    info = chroma_store.get_collection_info()
    if info.get("name") != name:
        info["name"] = name  # Placeholder if actual store doesn't match
    return CollectionInfo(name=info["name"], num_documents=info.get("count", 0))

@router.delete("/collections/{name}", response_model=DeleteResponse, tags=["collections"])
async def delete_collection(
    name: str,
    chroma_store: ChromaStore = Depends(get_chroma_store)
):
    try:
        chroma_store.delete_collection()
        return DeleteResponse(message="Collection deleted successfully", collection_name=name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat()
    )

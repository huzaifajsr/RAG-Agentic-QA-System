import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
    yield


app = FastAPI(
    title="RAG Agentic Q&A System",
    description="An intelligent document Q&A system powered by RAG and LangGraph agents",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origin_list,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root():
    return {"message": "Welcome to the RAG Agentic Q&A System", "docs": "/docs"}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

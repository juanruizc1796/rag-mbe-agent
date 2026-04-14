"""
FastAPI application for the MBE RAG Agent.
Exposes REST endpoints for chat, ingestion, and health checks.
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from graph import get_compiled_graph
from utils.config import settings
from utils.db import init_db, get_recent_history
from utils.rag import ingest_pdfs

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting MBE Agent API...")
    await init_db()
    # Pre-compile the graph
    get_compiled_graph()
    logger.info("Agent ready.")
    yield
    logger.info("Shutting down MBE Agent API.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MBE RAG Agent API",
    description="Conversational Evidence-Based Medicine agent powered by LangGraph + Llama 3.1 + BioBERT",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User question")
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for multi-turn conversations. Auto-generated if not provided.",
    )


class ChatResponse(BaseModel):
    session_id: str
    response: str
    intent: Optional[str]
    language: str
    rewritten_query: Optional[str]
    retrieval_sufficient: Optional[bool]
    max_score: Optional[float]
    node_path: List[str]
    latency_ms: float


class IngestRequest(BaseModel):
    pdf_dir: str = Field(
        default="/app/data/pdfs",
        description="Absolute path to directory containing PDF files to ingest.",
    )


class HistoryResponse(BaseModel):
    session_id: str
    messages: List[dict]


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "model": settings.OLLAMA_MODEL}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    start = time.time()

    initial_state = {
        "session_id": session_id,
        "user_query": request.query,
        "language": settings.DEFAULT_LANGUAGE,
        "history": [],
        "history_summary": "",
        "intent": None,
        "intent_confidence": None,
        "rewritten_query": None,
        "use_rag": True,
        "retrieved_docs": [],
        "similarity_scores": [],
        "max_score": 0.0,
        "retrieval_sufficient": False,
        "final_response": "",
        "node_path": [],
        "start_time": start,
        "error": None,
    }

    try:
        graph = get_compiled_graph()
        # Run the graph (synchronous invoke in executor to not block event loop)
        loop = asyncio.get_event_loop()
        final_state = await loop.run_in_executor(
            None, graph.invoke, initial_state
        )
    except Exception as exc:
        logger.error("Graph execution error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent error: {str(exc)}")

    latency = (time.time() - start) * 1000

    return ChatResponse(
        session_id=session_id,
        response=final_state.get("final_response", ""),
        intent=final_state.get("intent"),
        language=final_state.get("language", settings.DEFAULT_LANGUAGE),
        rewritten_query=final_state.get("rewritten_query"),
        retrieval_sufficient=final_state.get("retrieval_sufficient"),
        max_score=final_state.get("max_score"),
        node_path=final_state.get("node_path", []),
        latency_ms=round(latency, 2),
    )


@app.post("/ingest")
async def ingest(request: IngestRequest):
    """Trigger PDF ingestion and FAISS index rebuild."""
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, ingest_pdfs, request.pdf_dir)
        return {"status": "success", "message": f"Ingested PDFs from {request.pdf_dir}"}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("Ingestion error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str):
    """Retrieve conversation history for a session."""
    messages = await get_recent_history(session_id, n=50)
    return HistoryResponse(session_id=session_id, messages=messages)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.API_PORT,
        reload=False,
        log_level=settings.LOG_LEVEL.lower(),
    )

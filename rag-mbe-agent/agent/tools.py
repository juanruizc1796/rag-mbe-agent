"""
LangChain tool definitions for the MBE RAG Agent.
Each tool is decorated with @tool and receives/returns typed Pydantic models.
"""

import logging
from typing import List

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from utils.config import settings
from utils.rag import retrieve

logger = logging.getLogger(__name__)


# ── Input / Output schemas ────────────────────────────────────────────────────
class RAGInput(BaseModel):
    query: str = Field(description="English query to retrieve relevant MBE documents.")
    top_k: int = Field(
        default=settings.FAISS_TOP_K,
        description="Number of documents to retrieve.",
    )


class RetrievedDocument(BaseModel):
    content: str
    source: str
    page: int | None
    score: float


class RAGOutput(BaseModel):
    documents: List[RetrievedDocument]
    scores: List[float]
    max_score: float
    sufficient: bool  # True if max_score >= threshold


# ── Tool: RAG ─────────────────────────────────────────────────────────────────
@tool(args_schema=RAGInput)
def rag_tool(query: str, top_k: int = settings.FAISS_TOP_K) -> dict:
    """
    Retrieve relevant Evidence-Based Medicine (EBM) documents from the
    FAISS vector index using BioBERT embeddings.

    Use this tool whenever a medical/clinical question requires evidence from
    the corpus.

    Returns a dict with:
    - documents: list of relevant chunks with source and score
    - scores: similarity scores per document
    - max_score: highest similarity score
    - sufficient: whether retrieval quality exceeds the threshold
    """
    logger.info("RAG tool called. Query: '%s', top_k=%d", query[:80], top_k)

    docs, scores = retrieve(query, top_k=top_k)

    if not docs:
        return RAGOutput(
            documents=[],
            scores=[],
            max_score=0.0,
            sufficient=False,
        ).model_dump()

    retrieved = [
        RetrievedDocument(
            content=doc.page_content,
            source=doc.metadata.get("source", "unknown"),
            page=doc.metadata.get("page"),
            score=score,
        )
        for doc, score in zip(docs, scores)
    ]

    max_score = max(scores) if scores else 0.0
    sufficient = max_score >= settings.FAISS_SIMILARITY_THRESHOLD

    output = RAGOutput(
        documents=retrieved,
        scores=scores,
        max_score=max_score,
        sufficient=sufficient,
    )

    logger.info(
        "RAG tool complete. max_score=%.3f, sufficient=%s, docs=%d",
        max_score,
        sufficient,
        len(docs),
    )

    return output.model_dump()


# ── All tools export ──────────────────────────────────────────────────────────
ALL_TOOLS = [rag_tool]

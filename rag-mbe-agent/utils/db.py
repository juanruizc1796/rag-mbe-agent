"""
Async PostgreSQL database layer using SQLAlchemy + asyncpg.
Handles:
- Connection pooling with retry logic
- Chat history (short-term + long-term)
- Technical logs for traceability
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import List, Optional
from uuid import uuid4

import asyncpg
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from utils.config import settings

logger = logging.getLogger(__name__)

# ── Engine ────────────────────────────────────────────────────────────────────
_engine = None
_session_factory = None


def _build_engine():
    global _engine, _session_factory
    _engine = create_async_engine(
        settings.postgres_dsn,
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW,
        pool_pre_ping=True,
        echo=False,
    )
    _session_factory = async_sessionmaker(
        _engine, expire_on_commit=False, class_=AsyncSession
    )


def get_session_factory() -> async_sessionmaker:
    global _session_factory
    if _session_factory is None:
        _build_engine()
    return _session_factory


# ── Schema Initialisation ─────────────────────────────────────────────────────
CREATE_TABLES_SQL = [
    """
    CREATE TABLE IF NOT EXISTS chat_history (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        session_id TEXT NOT NULL,
        role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
        content TEXT NOT NULL,
        language TEXT DEFAULT 'es',
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_chat_history_session
    ON chat_history(session_id, created_at)
    """,
    """
    CREATE TABLE IF NOT EXISTS agent_logs (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        session_id TEXT NOT NULL,
        query_original TEXT,
        query_rewritten TEXT,
        retrieved_docs JSONB,
        similarity_scores JSONB,
        latency_ms FLOAT,
        tokens_used INT,
        node_path JSONB,
        error TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_logs_session
    ON agent_logs(session_id, created_at)
    """,
]


async def init_db(retries: int = None, delay: float = None) -> None:
    """Create tables if they don't exist. Retries on connection failure."""
    retries = retries or settings.DB_MAX_RETRIES
    delay = delay or settings.DB_RETRY_DELAY
    factory = get_session_factory()

    for attempt in range(1, retries + 1):
        try:
            async with factory() as session:
                for query in CREATE_TABLES_SQL:
                    await session.execute(text(query))
                await session.commit()
            logger.info("Database initialised successfully.")
            return
        except Exception as exc:
            logger.warning("DB init attempt %d/%d failed: %s", attempt, retries, exc)
            if attempt < retries:
                await asyncio.sleep(delay)
            else:
                raise RuntimeError(
                    "Could not initialise database after retries."
                ) from exc


# ── Chat History ──────────────────────────────────────────────────────────────
async def save_message(
    session_id: str,
    role: str,
    content: str,
    language: str = "es",
) -> None:
    """Persist a single chat message."""
    factory = get_session_factory()
    async with factory() as session:
        await session.execute(
            text(
                """
                INSERT INTO chat_history (session_id, role, content, language)
                VALUES (:session_id, :role, :content, :language)
                """
            ),
            {
                "session_id": session_id,
                "role": role,
                "content": content,
                "language": language,
            },
        )
        await session.commit()


async def get_recent_history(
    session_id: str,
    n: int = None,
) -> List[dict]:
    """Return the last n messages for a session (short-term memory)."""
    n = n or settings.SHORT_TERM_HISTORY_N
    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(
            text(
                """
                SELECT role, content, created_at
                FROM chat_history
                WHERE session_id = :session_id
                ORDER BY created_at DESC
                LIMIT :n
                """
            ),
            {"session_id": session_id, "n": n},
        )
        rows = result.fetchall()
    # Return in chronological order
    return [
        {"role": r.role, "content": r.content, "created_at": str(r.created_at)}
        for r in reversed(rows)
    ]


async def get_full_history(session_id: str) -> List[dict]:
    """Return the complete history for a session (long-term memory)."""
    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(
            text(
                """
                SELECT role, content, language, created_at
                FROM chat_history
                WHERE session_id = :session_id
                ORDER BY created_at ASC
                """
            ),
            {"session_id": session_id},
        )
        rows = result.fetchall()
    return [
        {
            "role": r.role,
            "content": r.content,
            "language": r.language,
            "created_at": str(r.created_at),
        }
        for r in rows
    ]


# ── Logging ───────────────────────────────────────────────────────────────────
async def save_log(
    session_id: str,
    query_original: str,
    query_rewritten: Optional[str] = None,
    retrieved_docs: Optional[list] = None,
    similarity_scores: Optional[list] = None,
    latency_ms: Optional[float] = None,
    tokens_used: Optional[int] = None,
    node_path: Optional[list] = None,
    error: Optional[str] = None,
) -> None:
    """Persist a technical log entry for traceability."""
    factory = get_session_factory()
    async with factory() as session:
        await session.execute(
            text(
                """
                INSERT INTO agent_logs
                    (session_id, query_original, query_rewritten, retrieved_docs,
                     similarity_scores, latency_ms, tokens_used, node_path, error)
                VALUES
                    (:session_id, :query_original, :query_rewritten, :retrieved_docs,
                     :similarity_scores, :latency_ms, :tokens_used, :node_path, :error)
                """
            ),
            {
                "session_id": session_id,
                "query_original": query_original,
                "query_rewritten": query_rewritten,
                "retrieved_docs": json.dumps(retrieved_docs or []),
                "similarity_scores": json.dumps(similarity_scores or []),
                "latency_ms": latency_ms,
                "tokens_used": tokens_used,
                "node_path": json.dumps(node_path or []),
                "error": error,
            },
        )
        await session.commit()

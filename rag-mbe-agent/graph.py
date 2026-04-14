"""
LangGraph agent graph for the MBE RAG conversational agent.

Flow:
START → detect_language → classify_intent
classify_intent:
    NON_MBE → rejection_node → log_node → END
    MBE     → query_rewriting
query_rewriting:
    use_rag=True  → rag_node → validate_retrieval
    use_rag=False → memory_response → log_node → END
validate_retrieval:
    sufficient → generate_response → log_node → END
    insufficient → insufficient_node → log_node → END
"""

import json
import logging
import time
from typing import Annotated, Any, Dict, List, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from agent.helpers import (
    build_insufficient_retrieval_message,
    build_intent_prompt,
    build_memory_only_prompt,
    build_rejection_message,
    build_response_prompt,
    build_rewriting_prompt,
    detect_language,
    extract_json,
    format_history,
    get_llm,
    invoke_with_retry,
)
from agent.tools import rag_tool
from utils.config import settings
from utils.db import get_recent_history, save_log, save_message

logger = logging.getLogger(__name__)


# ── State ─────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    # Inputs
    session_id: str
    user_query: str

    # Derived
    language: str
    history: List[Dict]
    history_summary: str

    # Intent
    intent: Optional[str]
    intent_confidence: Optional[float]

    # Query rewriting
    rewritten_query: Optional[str]
    use_rag: bool

    # Retrieval
    retrieved_docs: List[Dict]
    similarity_scores: List[float]
    max_score: float
    retrieval_sufficient: bool

    # Response
    final_response: str

    # Tracing
    node_path: List[str]
    start_time: float
    error: Optional[str]


# ── Node implementations ──────────────────────────────────────────────────────
def _get_history(state: AgentState) -> dict:
    """Load conversation history synchronously (wrapped async call)."""
    import asyncio

    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        history = loop.run_until_complete(get_recent_history(state["session_id"]))
    except Exception as exc:
        logger.warning("Could not load history: %s", exc)
        history = []
    return history


def node_detect_language(state: AgentState) -> AgentState:
    """Detect input language and load short-term memory."""
    node = "detect_language"
    logger.info("[%s] Query: '%s'", node, state["user_query"][:80])

    lang = detect_language(state["user_query"])
    history = _get_history(state)
    history_summary = format_history(history)

    return {
        **state,
        "language": lang,
        "history": history,
        "history_summary": history_summary,
        "node_path": state.get("node_path", []) + [node],
        "start_time": state.get("start_time", time.time()),
    }


def node_classify_intent(state: AgentState) -> AgentState:
    """Classify whether the query is MBE-related or not."""
    node = "classify_intent"
    logger.info("[%s] Classifying intent.", node)

    prompt = build_intent_prompt(state["user_query"], state["history_summary"])
    llm = get_llm()

    try:
        response = invoke_with_retry(llm.invoke, [{"role": "user", "content": prompt}])
        parsed = extract_json(response.content)
        intent = parsed.get("intent", "NON_MBE").upper()
        confidence = float(parsed.get("confidence", 0.0))
    except Exception as exc:
        logger.error(
            "[%s] Failed to classify intent: %s. Defaulting to MBE.", node, exc
        )
        intent = "MBE"
        confidence = 0.5

    logger.info("[%s] Intent=%s, confidence=%.2f", node, intent, confidence)

    return {
        **state,
        "intent": intent,
        "intent_confidence": confidence,
        "node_path": state["node_path"] + [node],
    }


def node_rejection(state: AgentState) -> AgentState:
    """Return polite rejection for non-MBE queries."""
    node = "rejection"
    logger.info("[%s] Rejecting non-MBE query.", node)
    message = build_rejection_message(state["language"], state["user_query"])
    return {
        **state,
        "final_response": message,
        "node_path": state["node_path"] + [node],
    }


def node_query_rewriting(state: AgentState) -> AgentState:
    """Rewrite query for optimal retrieval, applying PICO framework."""
    node = "query_rewriting"
    logger.info("[%s] Rewriting query.", node)

    prompt = build_rewriting_prompt(
        state["user_query"],
        state["history_summary"],
        state["language"],
    )
    llm = get_llm()

    try:
        response = invoke_with_retry(llm.invoke, [{"role": "user", "content": prompt}])
        parsed = extract_json(response.content)
        rewritten = parsed.get("rewritten_query", state["user_query"])
        use_rag = True
    except Exception as exc:
        logger.error("[%s] Rewriting failed: %s. Using original query.", node, exc)
        rewritten = state["user_query"]
        use_rag = True

    logger.info("[%s] Rewritten: '%s', use_rag=%s", node, rewritten[:80], use_rag)

    return {
        **state,
        "rewritten_query": rewritten,
        "use_rag": use_rag,
        "node_path": state["node_path"] + [node],
    }


def node_rag_tool(state: AgentState) -> AgentState:
    """Execute RAG retrieval using the FAISS tool."""
    node = "rag_tool"
    logger.info("[%s] Executing retrieval.", node)

    query = state.get("rewritten_query") or state["user_query"]

    try:
        result = rag_tool.invoke({"query": query, "top_k": settings.FAISS_TOP_K})
        docs = result.get("documents", [])
        scores = result.get("scores", [])
        max_score = result.get("max_score", 0.0)
        sufficient = result.get("sufficient", False)
    except Exception as exc:
        logger.error("[%s] RAG tool failed: %s", node, exc)
        docs, scores, max_score, sufficient = [], [], 0.0, False

    logger.info("[%s] max_score=%.3f, sufficient=%s", node, max_score, sufficient)

    return {
        **state,
        "retrieved_docs": docs,
        "similarity_scores": scores,
        "max_score": max_score,
        "retrieval_sufficient": sufficient,
        "node_path": state["node_path"] + [node],
    }


def node_validate_retrieval(state: AgentState) -> AgentState:
    """Validate retrieval quality. Sets retrieval_sufficient flag."""
    node = "validate_retrieval"
    sufficient = state.get("retrieval_sufficient", False)
    logger.info(
        "[%s] max_score=%.3f, threshold=%.2f, sufficient=%s",
        node,
        state.get("max_score", 0.0),
        settings.FAISS_SIMILARITY_THRESHOLD,
        sufficient,
    )
    return {
        **state,
        "node_path": state["node_path"] + [node],
    }


def node_insufficient_retrieval(state: AgentState) -> AgentState:
    """Inform user that retrieval was insufficient."""
    node = "insufficient_retrieval"
    message = build_insufficient_retrieval_message(state["language"])
    return {
        **state,
        "final_response": message,
        "node_path": state["node_path"] + [node],
    }


def node_generate_response(state: AgentState) -> AgentState:
    """Generate the final answer using retrieved context + history."""
    node = "generate_response"
    logger.info("[%s] Generating response.", node)

    docs = state.get("retrieved_docs", [])
    context_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.get("source", "unknown")
        page = doc.get("page", "?")
        content = doc.get("content", "")
        score = doc.get("score", 0.0)
        context_parts.append(
            f"[{i}] Source: {source}, Page: {page}, Score: {score:.3f}\n{content}"
        )
    context = "\n\n---\n\n".join(context_parts) if context_parts else "(No context)"

    prompt = build_response_prompt(
        query=state["user_query"],
        context=context,
        history_summary=state["history_summary"],
        language=state["language"],
    )
    llm = get_llm()

    try:
        response = invoke_with_retry(llm.invoke, [{"role": "user", "content": prompt}])
        answer = response.content.strip()
    except Exception as exc:
        logger.error("[%s] Response generation failed: %s", node, exc)
        answer = (
            "Lo siento, ocurrió un error al generar la respuesta. Por favor, inténtalo de nuevo."
            if state["language"] == "es"
            else "Sorry, an error occurred while generating the response. Please try again."
        )

    return {
        **state,
        "final_response": answer,
        "node_path": state["node_path"] + [node],
    }


def node_memory_response(state: AgentState) -> AgentState:
    """Generate response from memory only (no RAG needed)."""
    node = "memory_response"
    logger.info("[%s] Memory-only response.", node)

    prompt = build_memory_only_prompt(
        state["user_query"],
        state["history_summary"],
        state["language"],
    )
    llm = get_llm()

    try:
        response = invoke_with_retry(llm.invoke, [{"role": "user", "content": prompt}])
        answer = response.content.strip()
    except Exception as exc:
        logger.error("[%s] Memory response failed: %s", node, exc)
        answer = "Error generating response from memory."

    return {
        **state,
        "retrieved_docs": [],
        "similarity_scores": [],
        "final_response": answer,
        "node_path": state["node_path"] + [node],
    }


def node_logging(state: AgentState) -> AgentState:
    """Persist messages and technical log to PostgreSQL."""
    node = "logging"
    import asyncio

    latency_ms = (time.time() - state.get("start_time", time.time())) * 1000

    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        async def _persist():
            await save_message(
                state["session_id"], "user", state["user_query"], state["language"]
            )
            await save_message(
                state["session_id"],
                "assistant",
                state["final_response"],
                state["language"],
            )
            await save_log(
                session_id=state["session_id"],
                query_original=state["user_query"],
                query_rewritten=state.get("rewritten_query"),
                retrieved_docs=[
                    {"content": d.get("content", "")[:200], "source": d.get("source")}
                    for d in state.get("retrieved_docs", [])
                ],
                similarity_scores=state.get("similarity_scores", []),
                latency_ms=latency_ms,
                node_path=state.get("node_path", []),
                error=state.get("error"),
            )

        loop.run_until_complete(_persist())
        logger.info("[%s] Persisted. Latency: %.0f ms.", node, latency_ms)
    except Exception as exc:
        logger.error("[%s] Logging failed: %s", node, exc)

    return {
        **state,
        "node_path": state["node_path"] + [node],
    }


# ── Conditional edges ─────────────────────────────────────────────────────────
def route_intent(state: AgentState) -> str:
    if state.get("intent") == "MBE":
        return "query_rewriting"
    return "rejection"


def route_rag(state: AgentState) -> str:
    if state.get("use_rag", True):
        return "rag_tool"
    return "memory_response"


def route_retrieval(state: AgentState) -> str:
    if state.get("retrieval_sufficient", False):
        return "generate_response"
    return "insufficient_retrieval"


# ── Graph builder ─────────────────────────────────────────────────────────────
def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("detect_language", node_detect_language)
    graph.add_node("classify_intent", node_classify_intent)
    graph.add_node("rejection", node_rejection)
    graph.add_node("query_rewriting", node_query_rewriting)
    graph.add_node("rag_tool", node_rag_tool)
    graph.add_node("validate_retrieval", node_validate_retrieval)
    graph.add_node("insufficient_retrieval", node_insufficient_retrieval)
    graph.add_node("generate_response", node_generate_response)
    graph.add_node("memory_response", node_memory_response)
    graph.add_node("logging", node_logging)

    # Entry point
    graph.add_edge(START, "detect_language")
    graph.add_edge("detect_language", "classify_intent")

    # Intent routing
    graph.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "query_rewriting": "query_rewriting",
            "rejection": "rejection",
        },
    )

    # RAG vs memory routing
    graph.add_conditional_edges(
        "query_rewriting",
        route_rag,
        {
            "rag_tool": "rag_tool",
            "memory_response": "memory_response",
        },
    )

    # Retrieval validation routing
    graph.add_edge("rag_tool", "validate_retrieval")
    graph.add_conditional_edges(
        "validate_retrieval",
        route_retrieval,
        {
            "generate_response": "generate_response",
            "insufficient_retrieval": "insufficient_retrieval",
        },
    )

    # All terminal nodes → logging → END
    graph.add_edge("rejection", "logging")
    graph.add_edge("generate_response", "logging")
    graph.add_edge("memory_response", "logging")
    graph.add_edge("insufficient_retrieval", "logging")
    graph.add_edge("logging", END)

    return graph.compile()


# Singleton compiled graph
_compiled_graph = None


def get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph

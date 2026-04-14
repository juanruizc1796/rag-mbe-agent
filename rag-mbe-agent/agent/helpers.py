"""
Shared helpers used across agent nodes.
Includes: language detection, translation, LLM factory, prompt builders.
"""

import json
import logging
import re
import time
from typing import Optional

from langchain_ollama import ChatOllama

from utils.config import settings

logger = logging.getLogger(__name__)


# ── LLM factory ───────────────────────────────────────────────────────────────
def get_llm(tools: list | None = None) -> ChatOllama:
    """
    Build a ChatOllama instance.
    Retries are handled at the graph node level.
    """
    llm = ChatOllama(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0.1,
        timeout=settings.OLLAMA_TIMEOUT,
    )
    if tools:
        llm = llm.bind_tools(tools)
    return llm


# ── Language detection ────────────────────────────────────────────────────────
_SPANISH_MARKERS = {
    "qué", "que", "cómo", "como", "cuál", "cual", "cuáles", "cuales",
    "cuándo", "cuando", "dónde", "donde", "por qué", "por que", "cuánto",
    "cuanto", "es", "son", "los", "las", "del", "una", "uno", "con",
    "para", "sobre", "más", "más", "también", "también", "según",
}

_SPANISH_PATTERN = re.compile(
    r"\b(el|la|los|las|un|una|unos|unas|de|del|es|son|está|están|"
    r"tiene|tienen|con|para|que|qué|cómo|cuál|por|entre|se|si|no)\b",
    re.IGNORECASE,
)


def detect_language(text: str) -> str:
    """
    Heuristic language detection: returns 'es' or 'en'.
    Falls back to settings.DEFAULT_LANGUAGE.
    """
    text_lower = text.lower()
    words = set(re.findall(r"\b\w+\b", text_lower))
    overlap = words & _SPANISH_MARKERS
    matches = len(_SPANISH_PATTERN.findall(text))

    if len(overlap) >= 2 or matches >= 2:
        return "es"

    return "en"


# ── Prompt templates ──────────────────────────────────────────────────────────
def build_intent_prompt(query: str, history_summary: str) -> str:
    return f"""You are an intent classifier for a Medical Evidence-Based Medicine (EBM) assistant.

Conversation context:
{history_summary}

User query: "{query}"

Classify the query intent. Respond with a JSON object ONLY:
{{
  "intent": "MBE" | "NON_MBE",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<brief explanation>"
}}

MBE topics include: clinical trials, systematic reviews, meta-analyses, diagnostic tests,
treatment efficacy, epidemiology, clinical guidelines, biostatistics, pharmacology,
evidence grading, PICO framework, NNT/NNH, odds ratios, hazard ratios, p-values,
confidence intervals, sensitivity/specificity, clinical outcomes, RCTs.

NON_MBE topics include: general chat, politics, entertainment, cooking, sports, coding,
anything unrelated to medical evidence or clinical practice.

Respond ONLY with the JSON object. No preamble."""


def build_rewriting_prompt(query: str, history_summary: str, language: str) -> str:
    return f"""You are a query optimisation specialist for an Evidence-Based Medicine retrieval system.

Conversation context:
{history_summary}

Original query (language: {language}): "{query}"

Rewrite the query to maximise semantic retrieval from a biomedical English corpus.
Apply PICO framework if applicable (Population, Intervention, Comparison, Outcome).
The rewritten query MUST be in English regardless of input language.

Respond with a JSON object ONLY:
{{
  "rewritten_query": "<optimised English query>",
  "pico": {{
    "population": "<or null>",
    "intervention": "<or null>",
    "comparison": "<or null>",
    "outcome": "<or null>"
  }},
  "use_rag": <true|false>
}}

Set use_rag=false ONLY if the query can be answered from conversation history alone.
Respond ONLY with the JSON object. No preamble."""


def build_response_prompt(
    query: str,
    context: str,
    history_summary: str,
    language: str,
) -> str:
    lang_instruction = (
        "Responde en español." if language == "es" else "Respond in English."
    )
    return f"""You are an expert Evidence-Based Medicine (EBM) assistant.
{lang_instruction}
Be precise, cite evidence from the context, and never hallucinate.
If the context is insufficient, explicitly state it.

Conversation history:
{history_summary}

Retrieved evidence context:
{context}

User question: {query}

Structure your response as:
1. Direct answer
2. Evidence summary (cite sources if available)
3. Clinical implications (if relevant)
4. Limitations or caveats

If context is empty or irrelevant, say explicitly that you cannot answer based on available evidence."""


def build_memory_only_prompt(
    query: str,
    history_summary: str,
    language: str,
) -> str:
    lang_instruction = (
        "Responde en español." if language == "es" else "Respond in English."
    )
    return f"""You are an expert Evidence-Based Medicine (EBM) assistant.
{lang_instruction}

Conversation history:
{history_summary}

User question: {query}

Answer based on the conversation context. Be concise and accurate."""


def build_rejection_message(language: str, query: str) -> str:
    if language == "es":
        return (
            f"Lo siento, tu consulta «{query}» no está relacionada con Medicina Basada "
            "en la Evidencia (MBE). Este asistente está especializado exclusivamente en "
            "temas de MBE como ensayos clínicos, revisiones sistemáticas, guías clínicas, "
            "bioestadística y práctica clínica basada en evidencia. "
            "Por favor, reformula tu pregunta dentro de este dominio."
        )
    return (
        f"I'm sorry, your query «{query}» is not related to Evidence-Based Medicine (EBM). "
        "This assistant specialises exclusively in EBM topics such as clinical trials, "
        "systematic reviews, clinical guidelines, biostatistics, and evidence-based practice. "
        "Please reformulate your question within this domain."
    )


def build_insufficient_retrieval_message(language: str) -> str:
    if language == "es":
        return (
            "No encontré evidencia suficientemente relevante en el corpus para responder "
            "tu pregunta con precisión. Por favor, intenta reformularla con más detalle o "
            "utilizando términos clínicos específicos (por ejemplo, siguiendo el marco PICO)."
        )
    return (
        "I could not find sufficiently relevant evidence in the corpus to answer your "
        "question accurately. Please try rephrasing it with more detail or using specific "
        "clinical terms (e.g., following the PICO framework)."
    )


# ── JSON extraction ────────────────────────────────────────────────────────────
def extract_json(text: str) -> dict:
    """
    Robustly extract a JSON object from an LLM response.
    Handles markdown fences and trailing garbage.
    """
    # Strip markdown fences
    text = re.sub(r"```(?:json)?", "", text).strip()
    text = text.replace("```", "").strip()

    # Find first JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from LLM output: {text[:200]}")


# ── History formatting ─────────────────────────────────────────────────────────
def format_history(messages: list[dict]) -> str:
    """Format DB history rows into a readable string for prompts."""
    if not messages:
        return "(No previous conversation.)"
    lines = []
    for msg in messages:
        role = msg["role"].upper()
        content = msg["content"][:400]  # Truncate long messages
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


# ── Retry wrapper ─────────────────────────────────────────────────────────────
def invoke_with_retry(fn, *args, retries: int = None, delay: float = None, **kwargs):
    """Synchronous retry wrapper for LLM calls."""
    retries = retries or settings.OLLAMA_MAX_RETRIES
    delay = delay or settings.OLLAMA_RETRY_DELAY
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Attempt %d/%d failed: %s. Retrying in %.1fs...",
                attempt, retries, exc, delay,
            )
            time.sleep(delay)
    raise RuntimeError(f"All {retries} attempts failed.") from last_exc

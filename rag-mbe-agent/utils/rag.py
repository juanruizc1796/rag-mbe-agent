"""
RAG utilities: FAISS index management, document ingestion, and retrieval.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.config import settings
from utils.embeddings import get_embeddings

logger = logging.getLogger(__name__)

# ── Types ─────────────────────────────────────────────────────────────────────
RetrievalResult = Tuple[List[Document], List[float]]

# ── Paths ─────────────────────────────────────────────────────────────────────
INDEX_FILE = os.path.join(settings.FAISS_INDEX_PATH, "index.faiss")
DOCS_FILE = os.path.join(settings.FAISS_INDEX_PATH, "documents.pkl")

# ── Globals ───────────────────────────────────────────────────────────────────
_faiss_index: faiss.IndexFlatIP | None = None
_documents: List[Document] = []


# ── Index Management ──────────────────────────────────────────────────────────
def _save_index(index: faiss.IndexFlatIP, docs: List[Document]) -> None:
    Path(settings.FAISS_INDEX_PATH).mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, INDEX_FILE)
    with open(DOCS_FILE, "wb") as f:
        pickle.dump(docs, f)
    logger.info("FAISS index saved: %d vectors.", index.ntotal)


def _load_index() -> Tuple[faiss.IndexFlatIP | None, List[Document]]:
    if not os.path.exists(INDEX_FILE) or not os.path.exists(DOCS_FILE):
        logger.warning("FAISS index files not found. Index needs to be built.")
        return None, []
    index = faiss.read_index(INDEX_FILE)
    with open(DOCS_FILE, "rb") as f:
        docs = pickle.load(f)
    logger.info("FAISS index loaded: %d vectors.", index.ntotal)
    return index, docs


def get_faiss_index() -> Tuple[faiss.IndexFlatIP | None, List[Document]]:
    """Return cached index or load from disk."""
    global _faiss_index, _documents
    if _faiss_index is None:
        _faiss_index, _documents = _load_index()
    return _faiss_index, _documents


# ── Ingestion ─────────────────────────────────────────────────────────────────
def ingest_pdfs(pdf_dir: str) -> None:
    """
    Load PDFs from a directory, chunk them, embed with BioBERT, and persist
    a FAISS index to disk.
    """
    pdf_dir_path = Path(pdf_dir)
    pdf_files = list(pdf_dir_path.glob("**/*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in '{pdf_dir}'.")

    logger.info("Found %d PDF files. Ingesting...", len(pdf_files))

    # Load & chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n\n", "\n", ". ", " "],
    )
    all_docs: List[Document] = []
    for pdf_path in pdf_files:
        logger.info("Loading: %s", pdf_path.name)
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        chunks = splitter.split_documents(pages)
        for chunk in chunks:
            chunk.metadata["source"] = pdf_path.name
        all_docs.extend(chunks)

    logger.info("Total chunks: %d. Generating embeddings...", len(all_docs))

    # Embed
    embeddings_model = get_embeddings()
    texts = [doc.page_content for doc in all_docs]
    vectors = embeddings_model.embed_documents(texts)
    matrix = np.array(vectors, dtype="float32")

    # Build inner-product (cosine) index (vectors are L2-normalised)
    dim = matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    _save_index(index, all_docs)

    # Update globals
    global _faiss_index, _documents
    _faiss_index = index
    _documents = all_docs

    logger.info("Ingestion complete. Index contains %d vectors.", index.ntotal)


# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve(query: str, top_k: int = None) -> RetrievalResult:
    """
    Retrieve the top_k most similar documents for a query.

    Returns:
        (documents, scores) where scores are cosine similarities in [0, 1].
    """
    top_k = top_k or settings.FAISS_TOP_K
    index, docs = get_faiss_index()

    if index is None or len(docs) == 0:
        logger.error("FAISS index is empty or not loaded.")
        return [], []

    embeddings_model = get_embeddings()
    query_vec = embeddings_model.embed_query(query)
    query_matrix = np.array([query_vec], dtype="float32")

    # IndexFlatIP returns inner products (= cosine sim for normalised vectors)
    scores, indices = index.search(query_matrix, min(top_k, index.ntotal))

    retrieved_docs: List[Document] = []
    retrieved_scores: List[float] = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        retrieved_docs.append(docs[idx])
        # Clamp to [0, 1]
        retrieved_scores.append(float(max(0.0, min(1.0, score))))

    logger.debug(
        "Retrieved %d docs. Scores: %s",
        len(retrieved_docs),
        [f"{s:.3f}" for s in retrieved_scores],
    )

    return retrieved_docs, retrieved_scores

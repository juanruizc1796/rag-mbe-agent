"""
BioBERT embeddings wrapper compatible with LangChain's Embeddings interface.
"""

import logging
from typing import List

import torch
from langchain_core.embeddings import Embeddings
from transformers import AutoModel, AutoTokenizer

from utils.config import settings

logger = logging.getLogger(__name__)


class BioBERTEmbeddings(Embeddings):
    """
    LangChain-compatible embeddings using BioBERT from HuggingFace.
    Uses mean pooling over the last hidden state.
    """

    def __init__(self):
        self._tokenizer = None
        self._model = None
        self._device = settings.EMBEDDING_DEVICE
        self._model_name = settings.EMBEDDING_MODEL
        self._batch_size = settings.EMBEDDING_BATCH_SIZE

    def _load(self):
        if self._model is None:
            logger.info("Loading BioBERT model: %s on %s", self._model_name, self._device)
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModel.from_pretrained(self._model_name).to(self._device)
            self._model.eval()
            logger.info("BioBERT model loaded successfully.")

    def _mean_pooling(self, model_output, attention_mask) -> torch.Tensor:
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def _embed(self, texts: List[str]) -> List[List[float]]:
        self._load()
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                output = self._model(**encoded)

            embeddings = self._mean_pooling(output, encoded["attention_mask"])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.extend(embeddings.cpu().tolist())

        return all_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        logger.debug("Embedding %d documents.", len(texts))
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        logger.debug("Embedding query: %s", text[:80])
        return self._embed([text])[0]


# Singleton
_embeddings_instance: BioBERTEmbeddings | None = None


def get_embeddings() -> BioBERTEmbeddings:
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = BioBERTEmbeddings()
    return _embeddings_instance

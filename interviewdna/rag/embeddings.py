"""
Local embeddings via Sentence Transformers.

Used by both the ingestion pipeline (rag/ingestion.py) and, through
LlamaIndex's embedding interface, by the retriever (rag/retriever.py).
"""
from __future__ import annotations

import logging
import os
import time
from functools import lru_cache
from typing import List

logger = logging.getLogger("interviewdna.rag.embeddings")

DEFAULT_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer

    logger.info(
        "Loading embedding model '%s' (first call only -- may download "
        "weights the very first time, ~90MB for MiniLM)", DEFAULT_MODEL_NAME
    )
    start = time.monotonic()
    model = SentenceTransformer(DEFAULT_MODEL_NAME)
    logger.info("Embedding model loaded in %.1fs", time.monotonic() - start)
    return model


def get_embedding_dim() -> int:
    return _get_model().get_sentence_embedding_dimension()


def embed_texts(texts: List[str]) -> List[List[float]]:
    model = _get_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vectors]


def embed_text(text: str) -> List[float]:
    return embed_texts([text])[0]


class SentenceTransformerEmbedding:
    """Thin adapter exposing the shape LlamaIndex's BaseEmbedding-like usage
    expects (embed / embed_batch), so rag/retriever.py and rag/ingestion.py
    can use either LlamaIndex's own wrapper or this one interchangeably.
    """

    def get_text_embedding(self, text: str) -> List[float]:
        return embed_text(text)

    def get_text_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        return embed_texts(texts)

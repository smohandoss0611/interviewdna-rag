"""
Cross-encoder reranking.

Vector search and BM25 are both FAST but somewhat APPROXIMATE -- they score
a query against thousands of chunks independently, without ever letting the
query and each candidate "look at each other" closely (that's what makes
them fast enough to search a large index).

A cross-encoder is slower but more accurate: it takes the (query, candidate)
PAIR together as a single input and outputs one relevance score, letting the
model directly attend to both texts at once. Too slow to run over an entire
index, but perfectly fine to run over the ~10-20 candidates that vector
search + BM25 already narrowed things down to -- which is exactly the
"retrieve broad, then rerank precisely" pattern real search systems use.

Uses a small, CPU-friendly cross-encoder model from the same
sentence-transformers library already used for embeddings (rag/embeddings.py)
-- no new heavy dependency.
"""
from __future__ import annotations

import logging
import os
import time
from functools import lru_cache
from typing import List, Dict, Any

logger = logging.getLogger("interviewdna.rag.reranker")

DEFAULT_RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


@lru_cache(maxsize=1)
def _get_reranker():
    from sentence_transformers import CrossEncoder

    logger.info("Loading reranker model '%s' (first call only)", DEFAULT_RERANKER_MODEL)
    start = time.monotonic()
    model = CrossEncoder(DEFAULT_RERANKER_MODEL)
    logger.info("Reranker model loaded in %.1fs", time.monotonic() - start)
    return model


def rerank(query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """Re-sorts `candidates` (each a dict with a "text" field) by cross-encoder
    relevance to `query`, returning the top_k. Adds a `_rerank_score` field.

    Falls back to returning the input order (already fusion-ranked) if the
    reranker model can't be loaded -- reranking is a quality improvement,
    not a hard dependency the app should break without.
    """
    if not candidates:
        return []
    try:
        model = _get_reranker()
    except Exception as exc:
        logger.warning("Reranker unavailable (%s) -- falling back to fusion ranking without reranking", exc)
        return candidates[:top_k]

    start = time.monotonic()
    pairs = [(query, c.get("text", "")) for c in candidates]
    scores = model.predict(pairs)

    scored = list(zip(candidates, scores))
    scored.sort(key=lambda pair: pair[1], reverse=True)

    results = []
    for candidate, score in scored[:top_k]:
        item = dict(candidate)
        item["_rerank_score"] = float(score)
        results.append(item)

    logger.info(
        "Reranked %d candidate(s) -> top %d in %.2fs (top result found_by=%s)",
        len(candidates), len(results), time.monotonic() - start,
        results[0].get("_found_by") if results else None,
    )
    return results

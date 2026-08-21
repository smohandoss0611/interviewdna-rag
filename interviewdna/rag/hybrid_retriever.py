"""
Hybrid retriever: vector search + BM25 keyword search, fused with
Reciprocal Rank Fusion, then optionally reranked with a cross-encoder.

    query
      |
      +----------------------> Pinecone vector search  (rag/pinecone_store.py)
      |                              |
      +----------------------> BM25 keyword search      (rag/bm25_index.py)
                                     |
                          Reciprocal Rank Fusion         (rag/fusion.py)
                                     |
                          Cross-encoder rerank (optional) (rag/reranker.py)
                                     |
                              final ranked results

This is the retrieval-quality upgrade to the plain vector-only search in
rag/retriever.py. Use this wherever retrieval precision actually matters
(e.g. grounding a coaching explanation) -- plain vector search is still
fine for lighter-weight lookups.

MEMORY NOTE: the reranker loads a SECOND PyTorch model into memory (on top
of the embedding model every code path already needs). On memory-constrained
free-tier hosting (e.g. Render's 512MB free web service limit), this can be
enough to push the process over the ceiling and get OOM-killed mid-request.
Set ENABLE_RERANKER=false in that environment to skip loading it entirely --
hybrid search still runs (vector + BM25 + fusion), just without the final
reranking pass. Quality is somewhat lower but the app stays up.
"""
from __future__ import annotations

import logging
import os
import time
from typing import List, Dict, Any, Optional

from rag.pinecone_store import get_pinecone_store
from rag.bm25_index import get_bm25_store
from rag.fusion import reciprocal_rank_fusion
from rag.reranker import rerank as cross_encoder_rerank

logger = logging.getLogger("interviewdna.rag.hybrid")


def _reranker_enabled_by_default() -> bool:
    return os.getenv("ENABLE_RERANKER", "true").strip().lower() not in ("false", "0", "no")


def hybrid_retrieve(
    query: str,
    top_k: int = 5,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    document_type: Optional[str] = None,
    use_reranker: Optional[bool] = None,
    candidate_multiplier: int = 4,
) -> List[Dict[str, Any]]:
    """Retrieve the top_k most relevant chunks using hybrid search.

    candidate_multiplier controls how many candidates each individual method
    (vector, BM25) contributes BEFORE fusion/reranking narrow it down to
    top_k -- casting a wider net before fusion generally improves final
    quality, since a chunk that ranks #8 in vector search but #1 in BM25
    should still have a chance to win after fusion.

    use_reranker: pass True/False to force it either way for this call; leave
    as None (default) to use the ENABLE_RERANKER env var, which defaults to
    enabled. See the module docstring for why you'd disable this on
    memory-constrained hosting.
    """
    if use_reranker is None:
        use_reranker = _reranker_enabled_by_default()

    start = time.monotonic()
    candidate_k = top_k * candidate_multiplier

    vector_results = get_pinecone_store().query(
        query_text=query, top_k=candidate_k,
        user_id=user_id, session_id=session_id, document_type=document_type,
    )
    for r in vector_results:
        r["_source_list"] = "vector"

    bm25_results = get_bm25_store().search(
        query=query, top_k=candidate_k,
        user_id=user_id, session_id=session_id, document_type=document_type,
    )
    for r in bm25_results:
        r["_source_list"] = "bm25"

    fused = reciprocal_rank_fusion(vector_results, bm25_results)

    if use_reranker and fused:
        final = cross_encoder_rerank(query, fused, top_k=top_k)
    else:
        final = fused[:top_k]

    logger.info(
        "hybrid_retrieve(%r): vector=%d, bm25=%d, fused=%d, reranked=%s -> "
        "returned %d in %.2fs",
        query[:60], len(vector_results), len(bm25_results), len(fused),
        use_reranker, len(final), time.monotonic() - start,
    )
    return final

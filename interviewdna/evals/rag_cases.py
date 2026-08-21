"""
RAG evaluation cases -- checking retrieval and generation QUALITY against
InterviewDNA's actual hybrid search pipeline (rag/hybrid_retriever.py) and
tool-use-grounded coaching (agents/coaching_agent.py).

Each case seeds a small, controlled, self-contained corpus into a FRESH,
uniquely-named session (so this never collides with real user data already
in your Pinecone index), then checks a known-answer query against it. Same
"golden dataset" idea as evals/cases.py, applied to retrieval and
generation-faithfulness instead of extraction.

Requires a real, configured LLM AND real Pinecone/BM25 access (same
requirements as the rest of the app) -- these are NOT unit tests, they
exercise your actual retrieval infrastructure.
"""
from __future__ import annotations

import time
import uuid

from evals.scorers import EvalResult
from evals.rag_scorers import relevant_chunk_found, relevant_chunk_ranked_above, judge_faithfulness
from llm.base import LLMService
from rag.ingestion import ingest_text
from rag.hybrid_retriever import hybrid_retrieve
from rag.pinecone_store import get_pinecone_store
from rag.bm25_index import get_bm25_store
from agents.coaching_agent import coach_with_retrieved_context


# A small, controlled reference corpus -- deliberately pairs one clearly
# RELEVANT chunk with one clearly IRRELEVANT-but-similar-sounding chunk
# (both are about "partitioning" in a backend-systems context), so
# retrieval actually has to discriminate on meaning, not just topic
# adjacency.
_EVAL_USER_ID = "rag-eval"
_RELEVANT_CHUNK = (
    "Apache Kafka partitions allow a topic's data to be split across "
    "multiple brokers for parallel consumption. Increasing partition count "
    "increases consumer parallelism but also increases coordination "
    "overhead and rebalance time."
)
_IRRELEVANT_CHUNK = (
    "PostgreSQL table partitioning splits a large table into smaller "
    "physical pieces based on a partition key, improving query performance "
    "on very large tables without changing application-level SQL."
)


def _seed_eval_corpus() -> str:
    """Ingests the controlled corpus into a FRESH session scope so this eval
    never collides with real user data. Returns the session_id used -- pass
    this to every retrieval call in the same case, matching what was used
    at ingestion time (both Pinecone and BM25 scope by session_id).

    Waits briefly after ingestion before returning: Pinecone doesn't
    guarantee a just-written vector is INSTANTLY queryable -- there's a
    small indexing delay. We hit this for real: one eval case took long
    enough (loading the reranker model for the first time) that Pinecone
    had caught up by query time and passed; the next case ran fast enough
    to query before its own freshly-seeded data was indexed, and got back
    an empty result with no error. This settle delay closes that race
    deterministically instead of getting lucky/unlucky on incidental
    latency elsewhere in the pipeline."""
    session_id = f"eval-{uuid.uuid4()}"
    ingest_text(_RELEVANT_CHUNK, user_id=_EVAL_USER_ID, session_id=session_id,
                document_type="reference", source="kafka_docs")
    ingest_text(_IRRELEVANT_CHUNK, user_id=_EVAL_USER_ID, session_id=session_id,
                document_type="reference", source="postgres_docs")
    time.sleep(2)  # let Pinecone's indexing catch up before anything queries this scope
    return session_id


def _diagnose_empty_retrieval(query: str, session_id: str) -> str:
    """Called only when hybrid_retrieve() comes back empty -- separately
    queries the two underlying stores directly, so the failure detail says
    WHICH one was empty (Pinecone indexing lag vs. a real BM25 bug vs. both)
    instead of just an unhelpful empty list."""
    try:
        vector_hits = get_pinecone_store().query(
            query_text=query, top_k=3, user_id=_EVAL_USER_ID, session_id=session_id, document_type="reference"
        )
        bm25_hits = get_bm25_store().search(
            query=query, top_k=3, user_id=_EVAL_USER_ID, session_id=session_id, document_type="reference"
        )
        return f"diagnostic: vector={len(vector_hits)} hit(s), bm25={len(bm25_hits)} hit(s)"
    except Exception as exc:
        return f"diagnostic query itself failed: {exc}"


# --------------------------------------------------------------------------- #
# Case 1 -- retrieval precision: does hybrid search find the actually
# relevant chunk for a query, not just something topically adjacent?
# --------------------------------------------------------------------------- #
def eval_hybrid_retrieval_finds_relevant_chunk(llm: LLMService) -> EvalResult:
    start = time.monotonic()
    try:
        session_id = _seed_eval_corpus()
        results = hybrid_retrieve(
            "How does partitioning affect Kafka consumer parallelism?",
            top_k=3, user_id=_EVAL_USER_ID, session_id=session_id, document_type="reference",
        )
        found = relevant_chunk_found(results, "consumer parallelism")
        detail = f"retrieved {len(results)} chunk(s), found expected content: {found}"
        if not results:
            detail += f" -- {_diagnose_empty_retrieval('How does partitioning affect Kafka consumer parallelism?', session_id)}"
        return EvalResult("hybrid_retrieval_finds_relevant_chunk", found, detail, time.monotonic() - start)
    except Exception as exc:
        return EvalResult("hybrid_retrieval_finds_relevant_chunk", False, f"EXCEPTION: {exc}", time.monotonic() - start)


# --------------------------------------------------------------------------- #
# Case 2 -- ranking quality: is the relevant chunk ranked ABOVE the
# similar-sounding-but-irrelevant one, not just present somewhere in results?
# --------------------------------------------------------------------------- #
def eval_hybrid_retrieval_ranks_relevant_above_similar_sounding(llm: LLMService) -> EvalResult:
    start = time.monotonic()
    try:
        session_id = _seed_eval_corpus()
        results = hybrid_retrieve(
            "How does partitioning affect Kafka consumer parallelism?",
            top_k=3, user_id=_EVAL_USER_ID, session_id=session_id, document_type="reference",
        )
        ranked_correctly = relevant_chunk_ranked_above(results, "consumer parallelism", "PostgreSQL table")
        detail = f"order: {[r.get('text', '')[:40] for r in results]}"
        if not results:
            detail += f" -- {_diagnose_empty_retrieval('How does partitioning affect Kafka consumer parallelism?', session_id)}"
        return EvalResult(
            "hybrid_retrieval_ranks_relevant_above_similar_sounding", ranked_correctly, detail, time.monotonic() - start
        )
    except Exception as exc:
        return EvalResult(
            "hybrid_retrieval_ranks_relevant_above_similar_sounding", False, f"EXCEPTION: {exc}", time.monotonic() - start
        )


# --------------------------------------------------------------------------- #
# Case 3 -- generation faithfulness: does coaching text stay grounded in
# what was actually retrieved, without adding unsupported claims?
# --------------------------------------------------------------------------- #
def eval_coaching_is_faithful_to_retrieved_context(llm: LLMService) -> EvalResult:
    start = time.monotonic()
    try:
        result = coach_with_retrieved_context(
            llm,
            competency="Kafka",
            detected_gap="consumer parallelism tradeoffs",
            original_question="How would you scale Kafka consumers?",
            original_answer="I'm not totally sure how partitions relate to consumer scaling.",
            retrieved_chunks=[{"text": _RELEVANT_CHUNK, "source": "kafka_docs"}],
        )
        judgment = judge_faithfulness(llm, context=_RELEVANT_CHUNK, generated_text=result.coaching_text)
        detail = f"faithful={judgment.faithful}, unsupported_claims={judgment.unsupported_claims}"
        return EvalResult("coaching_is_faithful_to_retrieved_context", judgment.faithful, detail, time.monotonic() - start)
    except Exception as exc:
        return EvalResult("coaching_is_faithful_to_retrieved_context", False, f"EXCEPTION: {exc}", time.monotonic() - start)


RAG_CASES = [
    eval_hybrid_retrieval_finds_relevant_chunk,
    eval_hybrid_retrieval_ranks_relevant_above_similar_sounding,
    eval_coaching_is_faithful_to_retrieved_context,
]

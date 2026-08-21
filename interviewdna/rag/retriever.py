"""
Retriever used by the agentic interview loop.

    LangGraph node needs context
              |
              v
       rag/retriever.py  (this file)
              |
              v
     rag/pinecone_store.py  ->  Pinecone

Kept intentionally simple (direct Pinecone query via the embedding model) so
the retrieval path used DURING the live interview loop has minimal latency.
LlamaIndex is still the ingestion/indexing engine (rag/ingestion.py); this
module is the retrieval-time counterpart the agents call.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional

from rag.pinecone_store import get_pinecone_store


def retrieve_context(
    query: str,
    top_k: int = 4,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    document_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieve relevant chunks for a query, scoped optionally by user/session/type.

    Returns a list of dicts: {id, score, text, source, skill, topic, document_type}
    """
    store = get_pinecone_store()
    return store.query(
        query_text=query,
        top_k=top_k,
        user_id=user_id,
        session_id=session_id,
        document_type=document_type,
    )


def retrieve_resume_evidence(query: str, user_id: str, session_id: str, top_k: int = 3) -> str:
    """Convenience helper: retrieve resume-scoped evidence and join into a short string
    for prompt injection (used by LLM CALL #5 - question generation)."""
    matches = retrieve_context(
        query, top_k=top_k, user_id=user_id, session_id=session_id, document_type="resume"
    )
    return "\n".join(f"- {m['text']}" for m in matches)


def retrieve_reference_context(query: str, top_k: int = 4) -> List[Dict[str, Any]]:
    """Retrieve from reference/learning-resource material (not scoped to a session),
    used for coaching (LLM CALL #10)."""
    return retrieve_context(query, top_k=top_k, document_type="reference")

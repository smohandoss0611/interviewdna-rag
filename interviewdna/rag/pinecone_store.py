"""
Pinecone access is encapsulated ENTIRELY inside this module, per spec section 18/19.
No other module should import the `pinecone` package directly.

Uses a single Pinecone Starter index for the hackathon, with metadata used to
scope queries by user_id / session_id / document_type / skill / topic / source.
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from typing import List, Dict, Any, Optional

from rag.embeddings import embed_texts, get_embedding_dim

logger = logging.getLogger("interviewdna.rag.pinecone")

INDEX_NAME = os.getenv("PINECONE_INDEX", "interviewdna")


class PineconeStore:
    def __init__(self):
        self._pc = None
        self._index = None

    # ------------------------------------------------------------------ #
    def _client(self):
        if self._pc is None:
            from pinecone import Pinecone

            api_key = os.getenv("PINECONE_API_KEY")
            if not api_key:
                raise RuntimeError("PINECONE_API_KEY is not set (see .env.example)")
            self._pc = Pinecone(api_key=api_key)
        return self._pc

    def _get_index(self):
        if self._index is None:
            pc = self._client()
            existing = [i["name"] for i in pc.list_indexes()]
            if INDEX_NAME not in existing:
                logger.info(
                    "Pinecone index '%s' not found -- creating it (dim=%d). "
                    "This can take 30-60s on a fresh Starter account.",
                    INDEX_NAME, get_embedding_dim(),
                )
                from pinecone import ServerlessSpec

                start = time.monotonic()
                pc.create_index(
                    name=INDEX_NAME,
                    dimension=get_embedding_dim(),
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud=os.getenv("PINECONE_CLOUD", "aws"),
                        region=os.getenv("PINECONE_REGION", "us-east-1"),
                    ),
                )
                logger.info("Pinecone index '%s' created in %.1fs", INDEX_NAME, time.monotonic() - start)
            self._index = pc.Index(INDEX_NAME)
        return self._index

    # ------------------------------------------------------------------ #
    def upsert_chunks(
        self,
        chunks: List[str],
        user_id: str,
        session_id: str,
        document_type: str,
        skill: Optional[str] = None,
        topic: Optional[str] = None,
        source: Optional[str] = None,
    ) -> int:
        """Embed and upsert text chunks with metadata. Returns count upserted."""
        if not chunks:
            return 0
        logger.info("Embedding %d chunk(s) for upsert (document_type=%s)", len(chunks), document_type)
        start = time.monotonic()
        vectors = embed_texts(chunks)
        logger.info("Embedding done in %.2fs", time.monotonic() - start)
        items = []
        for text, vec in zip(chunks, vectors):
            items.append(
                {
                    "id": str(uuid.uuid4()),
                    "values": vec,
                    "metadata": {
                        "text": text[:4000],
                        "user_id": user_id,
                        "session_id": session_id,
                        "document_type": document_type,
                        "skill": skill or "",
                        "topic": topic or "",
                        "source": source or document_type,
                    },
                }
            )
        index = self._get_index()
        index.upsert(vectors=items)
        logger.info("Upserted %d vector(s) to Pinecone index '%s'", len(items), INDEX_NAME)
        return len(items)

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        document_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        start = time.monotonic()
        vec = embed_texts([query_text])[0]
        flt: Dict[str, Any] = {}
        if user_id:
            flt["user_id"] = {"$eq": user_id}
        if session_id:
            flt["session_id"] = {"$eq": session_id}
        if document_type:
            flt["document_type"] = {"$eq": document_type}

        index = self._get_index()
        result = index.query(
            vector=vec, top_k=top_k, include_metadata=True, filter=flt or None
        )
        matches = []
        for m in result.get("matches", []):
            md = m.get("metadata", {}) or {}
            matches.append(
                {
                    "id": m.get("id"),
                    "score": m.get("score"),
                    "text": md.get("text", ""),
                    "source": md.get("source", ""),
                    "skill": md.get("skill", ""),
                    "topic": md.get("topic", ""),
                    "document_type": md.get("document_type", ""),
                }
            )
        logger.info(
            "Pinecone query done in %.2fs -- %d match(es) for %r (filter=%s)",
            time.monotonic() - start, len(matches), query_text[:60], flt or "none",
        )
        return matches


# Module-level singleton, mirrors llm service pattern: agents/services get a
# ready-to-use store without managing Pinecone client lifecycle themselves.
_store: Optional[PineconeStore] = None


def get_pinecone_store() -> PineconeStore:
    global _store
    if _store is None:
        _store = PineconeStore()
    return _store

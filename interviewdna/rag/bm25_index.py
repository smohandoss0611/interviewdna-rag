"""
BM25 keyword search index.

Pinecone (rag/pinecone_store.py) gives us SEMANTIC search -- "find chunks
that MEAN something similar to the query." That's powerful, but it can miss
exact keyword matches: a query for "Kafka" might not rank a chunk highly if
that chunk phrases things very differently, even though it literally
contains the word "Kafka".

BM25 is a classic, fast, well-understood KEYWORD ranking algorithm (the same
family of algorithm search engines used before embeddings existed, and
still use alongside them). It scores documents by term overlap and term
rarity -- rare words that appear in both the query and a document score
much higher than common words.

Combining the two (see rag/fusion.py) is "hybrid search": you get both
semantic understanding AND exact keyword precision.

This index is deliberately simple and in-memory, scoped by the same
(user_id, session_id, document_type) metadata Pinecone uses, so the two
indexes stay conceptually aligned. For a real production system you'd
likely use a proper search engine (Elasticsearch/OpenSearch/Postgres
full-text search) instead of an in-memory index, but the ALGORITHM and the
hybrid-fusion pattern are the same regardless of scale.
"""
from __future__ import annotations

import logging
import re
import threading
from typing import Dict, List, Any, Optional, Tuple

from rank_bm25 import BM25Okapi

logger = logging.getLogger("interviewdna.rag.bm25")

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    """Deliberately dumb tokenizer (lowercase + split on non-alphanumeric).
    BM25 doesn't need anything fancier than this to work well."""
    return _TOKEN_RE.findall(text.lower())


class _ScopedCorpus:
    """One BM25 index + the raw documents for a single scope key."""

    def __init__(self):
        self.doc_ids: List[str] = []
        self.texts: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self._bm25: Optional[BM25Okapi] = None

    def add(self, doc_id: str, text: str, metadata: Dict[str, Any]) -> None:
        self.doc_ids.append(doc_id)
        self.texts.append(text)
        self.metadatas.append(metadata)
        self._bm25 = None  # invalidate cache; rebuilt lazily on next search

    def _ensure_index(self) -> Optional[BM25Okapi]:
        if not self.texts:
            return None
        if self._bm25 is None:
            tokenized = [_tokenize(t) for t in self.texts]
            self._bm25 = BM25Okapi(tokenized)
        return self._bm25

    def search(self, query: str, top_k: int) -> List[Tuple[float, int]]:
        bm25 = self._ensure_index()
        if bm25 is None:
            return []
        scores = bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(float(scores[i]), i) for i in ranked[:top_k] if scores[i] > 0]


class BM25Store:
    """Module-level keyword index, mirrors the shape of PineconeStore so the
    two can be used interchangeably in hybrid retrieval code."""

    def __init__(self):
        self._scopes: Dict[str, _ScopedCorpus] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _scope_key(user_id: Optional[str], session_id: Optional[str], document_type: Optional[str]) -> str:
        return f"{user_id or '*'}::{session_id or '*'}::{document_type or '*'}"

    def add_documents(
        self,
        chunks: List[str],
        user_id: str,
        session_id: str,
        document_type: str,
        source: Optional[str] = None,
    ) -> int:
        if not chunks:
            return 0
        key = self._scope_key(user_id, session_id, document_type)
        with self._lock:
            corpus = self._scopes.setdefault(key, _ScopedCorpus())
            start_idx = len(corpus.texts)
            for i, chunk in enumerate(chunks):
                corpus.add(
                    doc_id=f"{key}::{start_idx + i}",
                    text=chunk,
                    metadata={
                        "text": chunk,
                        "user_id": user_id,
                        "session_id": session_id,
                        "document_type": document_type,
                        "source": source or document_type,
                    },
                )
        logger.info("BM25: indexed %d chunk(s) into scope '%s'", len(chunks), key)
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        document_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        key = self._scope_key(user_id, session_id, document_type)
        with self._lock:
            corpus = self._scopes.get(key)
            if corpus is None:
                return []
            hits = corpus.search(query, top_k)
            results = []
            for score, idx in hits:
                md = corpus.metadatas[idx]
                results.append(
                    {
                        "id": corpus.doc_ids[idx],
                        "score": score,
                        "text": md["text"],
                        "source": md.get("source", ""),
                        "document_type": md.get("document_type", ""),
                    }
                )
            return results


_store: Optional[BM25Store] = None


def get_bm25_store() -> BM25Store:
    global _store
    if _store is None:
        _store = BM25Store()
    return _store

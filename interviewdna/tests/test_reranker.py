from unittest.mock import patch

from rag.reranker import rerank


class _FakeCrossEncoder:
    """Stands in for a real sentence-transformers CrossEncoder -- returns a
    fixed relevance score per (query, text) pair based on simple keyword
    overlap, so we can test the SORTING/TRUNCATION logic in rerank()
    without downloading real model weights."""

    def predict(self, pairs):
        scores = []
        for query, text in pairs:
            query_words = set(query.lower().split())
            text_words = set(text.lower().split())
            overlap = len(query_words & text_words)
            scores.append(float(overlap))
        return scores


def test_rerank_sorts_by_relevance_score():
    candidates = [
        {"id": "a", "text": "The weather today is sunny."},
        {"id": "b", "text": "Kafka is used for event streaming pipelines."},
        {"id": "c", "text": "Streaming Kafka event pipelines are common."},
    ]
    with patch("rag.reranker._get_reranker", return_value=_FakeCrossEncoder()):
        results = rerank("Kafka streaming pipelines", candidates, top_k=3)

    # The two Kafka-relevant chunks should outrank the irrelevant weather one.
    ids_in_order = [r["id"] for r in results]
    assert ids_in_order[0] in ("b", "c")
    assert ids_in_order[-1] == "a"
    assert all("_rerank_score" in r for r in results)


def test_rerank_respects_top_k():
    candidates = [{"id": str(i), "text": f"chunk {i} Kafka"} for i in range(10)]
    with patch("rag.reranker._get_reranker", return_value=_FakeCrossEncoder()):
        results = rerank("Kafka", candidates, top_k=3)
    assert len(results) == 3


def test_rerank_falls_back_gracefully_when_model_unavailable():
    candidates = [{"id": "a", "text": "x"}, {"id": "b", "text": "y"}]
    with patch("rag.reranker._get_reranker", side_effect=RuntimeError("no internet / model not cached")):
        results = rerank("query", candidates, top_k=5)
    # Falls back to the input order rather than crashing.
    assert results == candidates[:5]


def test_rerank_handles_empty_candidates():
    with patch("rag.reranker._get_reranker", return_value=_FakeCrossEncoder()):
        assert rerank("query", [], top_k=5) == []

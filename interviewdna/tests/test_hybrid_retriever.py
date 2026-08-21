from unittest.mock import patch, MagicMock

from rag.hybrid_retriever import hybrid_retrieve


def test_hybrid_retrieve_queries_both_stores_and_fuses():
    fake_pinecone = MagicMock()
    fake_pinecone.query.return_value = [
        {"id": "v1", "text": "vector result one", "score": 0.9},
        {"id": "shared", "text": "found by both methods", "score": 0.8},
    ]
    fake_bm25 = MagicMock()
    fake_bm25.search.return_value = [
        {"id": "shared", "text": "found by both methods", "score": 5.0},
        {"id": "k1", "text": "keyword result one", "score": 3.0},
    ]

    with patch("rag.hybrid_retriever.get_pinecone_store", return_value=fake_pinecone), \
         patch("rag.hybrid_retriever.get_bm25_store", return_value=fake_bm25):
        results = hybrid_retrieve("test query", top_k=5, use_reranker=False)

    fake_pinecone.query.assert_called_once()
    fake_bm25.search.assert_called_once()

    ids = [r["id"] for r in results]
    assert "shared" in ids
    # The item found by BOTH retrieval methods should rank first.
    assert results[0]["id"] == "shared"
    assert sorted(results[0]["_found_by"]) == ["bm25", "vector"]


def test_hybrid_retrieve_requests_wider_candidate_pool_than_top_k():
    fake_pinecone = MagicMock()
    fake_pinecone.query.return_value = []
    fake_bm25 = MagicMock()
    fake_bm25.search.return_value = []

    with patch("rag.hybrid_retriever.get_pinecone_store", return_value=fake_pinecone), \
         patch("rag.hybrid_retriever.get_bm25_store", return_value=fake_bm25):
        hybrid_retrieve("q", top_k=3, candidate_multiplier=4, use_reranker=False)

    # Should ask each underlying store for MORE than top_k candidates
    # (top_k * candidate_multiplier), so fusion has real material to work with.
    _, kwargs = fake_pinecone.query.call_args
    assert kwargs["top_k"] == 12


def test_hybrid_retrieve_applies_reranker_when_enabled():
    fake_pinecone = MagicMock()
    fake_pinecone.query.return_value = [{"id": "a", "text": "x"}]
    fake_bm25 = MagicMock()
    fake_bm25.search.return_value = []

    fake_reranked = [{"id": "a", "text": "x", "_rerank_score": 0.99}]
    with patch("rag.hybrid_retriever.get_pinecone_store", return_value=fake_pinecone), \
         patch("rag.hybrid_retriever.get_bm25_store", return_value=fake_bm25), \
         patch("rag.hybrid_retriever.cross_encoder_rerank", return_value=fake_reranked) as mock_rerank:
        results = hybrid_retrieve("q", top_k=1, use_reranker=True)

    mock_rerank.assert_called_once()
    assert results == fake_reranked


def test_hybrid_retrieve_skips_reranker_when_disabled():
    fake_pinecone = MagicMock()
    fake_pinecone.query.return_value = [{"id": "a", "text": "x"}]
    fake_bm25 = MagicMock()
    fake_bm25.search.return_value = []

    with patch("rag.hybrid_retriever.get_pinecone_store", return_value=fake_pinecone), \
         patch("rag.hybrid_retriever.get_bm25_store", return_value=fake_bm25), \
         patch("rag.hybrid_retriever.cross_encoder_rerank") as mock_rerank:
        hybrid_retrieve("q", top_k=1, use_reranker=False)

    mock_rerank.assert_not_called()

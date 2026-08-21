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


def test_reranker_default_controlled_by_env_var(monkeypatch):
    """Regression test: this is the fix for a real production OOM crash on
    Render's free tier (512MB limit) -- loading the reranker's second
    PyTorch model pushed memory over the ceiling. ENABLE_RERANKER=false
    must skip it WITHOUT the caller needing to pass use_reranker explicitly
    everywhere, since that default is what actually gets deployed."""
    fake_pinecone = MagicMock()
    fake_pinecone.query.return_value = [{"id": "a", "text": "x"}]
    fake_bm25 = MagicMock()
    fake_bm25.search.return_value = []

    monkeypatch.setenv("ENABLE_RERANKER", "false")
    with patch("rag.hybrid_retriever.get_pinecone_store", return_value=fake_pinecone), \
         patch("rag.hybrid_retriever.get_bm25_store", return_value=fake_bm25), \
         patch("rag.hybrid_retriever.cross_encoder_rerank") as mock_rerank:
        hybrid_retrieve("q", top_k=1)  # use_reranker not passed -- should read env var

    mock_rerank.assert_not_called()


def test_reranker_enabled_by_default_when_env_var_unset(monkeypatch):
    fake_pinecone = MagicMock()
    fake_pinecone.query.return_value = [{"id": "a", "text": "x"}]
    fake_bm25 = MagicMock()
    fake_bm25.search.return_value = []

    monkeypatch.delenv("ENABLE_RERANKER", raising=False)
    with patch("rag.hybrid_retriever.get_pinecone_store", return_value=fake_pinecone), \
         patch("rag.hybrid_retriever.get_bm25_store", return_value=fake_bm25), \
         patch("rag.hybrid_retriever.cross_encoder_rerank", return_value=[]) as mock_rerank:
        hybrid_retrieve("q", top_k=1)

    mock_rerank.assert_called_once()


def test_explicit_use_reranker_overrides_env_var(monkeypatch):
    """Passing use_reranker explicitly should always win over the env var,
    so call sites that genuinely need it off/on for a specific reason
    aren't at the mercy of global config."""
    fake_pinecone = MagicMock()
    fake_pinecone.query.return_value = [{"id": "a", "text": "x"}]
    fake_bm25 = MagicMock()
    fake_bm25.search.return_value = []

    monkeypatch.setenv("ENABLE_RERANKER", "true")
    with patch("rag.hybrid_retriever.get_pinecone_store", return_value=fake_pinecone), \
         patch("rag.hybrid_retriever.get_bm25_store", return_value=fake_bm25), \
         patch("rag.hybrid_retriever.cross_encoder_rerank") as mock_rerank:
        hybrid_retrieve("q", top_k=1, use_reranker=False)  # explicit False overrides env=true

    mock_rerank.assert_not_called()

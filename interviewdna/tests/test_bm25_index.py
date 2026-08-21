from rag.bm25_index import BM25Store


def test_bm25_finds_exact_keyword_match():
    store = BM25Store()
    store.add_documents(
        chunks=[
            "We use Apache Kafka for real-time event streaming.",
            "Our data warehouse is built on Snowflake.",
            "The API gateway handles rate limiting and auth.",
        ],
        user_id="u1", session_id="s1", document_type="reference",
    )
    results = store.search("Kafka streaming", top_k=3, user_id="u1", session_id="s1", document_type="reference")
    assert results, "BM25 should return at least one match"
    assert "Kafka" in results[0]["text"]


def test_bm25_scopes_by_session_and_document_type():
    store = BM25Store()
    store.add_documents(["Kubernetes deployment guide"], user_id="u1", session_id="s1", document_type="reference")
    store.add_documents(["Unrelated resume content"], user_id="u1", session_id="s2", document_type="resume")

    # Searching a DIFFERENT session/doc_type scope should not see session s1's docs.
    results = store.search("Kubernetes", top_k=5, user_id="u1", session_id="s2", document_type="resume")
    assert all("Kubernetes" not in r["text"] for r in results)


def test_bm25_returns_empty_for_unknown_scope():
    store = BM25Store()
    results = store.search("anything", top_k=5, user_id="nobody", session_id="nothing", document_type="none")
    assert results == []


def test_bm25_returns_empty_for_no_overlap_query():
    store = BM25Store()
    store.add_documents(["Apache Airflow orchestrates data pipelines."], user_id="u1", session_id="s1", document_type="reference")
    # A query with zero token overlap should score 0 and be filtered out.
    results = store.search("giraffe pancake umbrella", top_k=5, user_id="u1", session_id="s1", document_type="reference")
    assert results == []

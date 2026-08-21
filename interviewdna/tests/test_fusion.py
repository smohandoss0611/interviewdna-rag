from rag.fusion import reciprocal_rank_fusion


def test_item_ranked_well_in_both_lists_wins():
    vector_results = [
        {"id": "a", "text": "chunk A"},
        {"id": "b", "text": "chunk B"},
        {"id": "c", "text": "chunk C"},
    ]
    bm25_results = [
        {"id": "a", "text": "chunk A"},
        {"id": "d", "text": "chunk D"},
        {"id": "e", "text": "chunk E"},
    ]
    fused = reciprocal_rank_fusion(vector_results, bm25_results)
    # "a" ranks #1 in BOTH lists -> should be the top fused result.
    assert fused[0]["id"] == "a"
    assert sorted(fused[0]["_found_by"]) == ["list_0", "list_1"]


def test_item_found_by_only_one_method_still_included():
    vector_results = [{"id": "a", "text": "x"}]
    bm25_results = [{"id": "b", "text": "y"}]
    fused = reciprocal_rank_fusion(vector_results, bm25_results)
    ids = {item["id"] for item in fused}
    assert ids == {"a", "b"}


def test_fusion_deduplicates_same_item_across_lists():
    vector_results = [{"id": "a", "text": "x"}, {"id": "b", "text": "y"}]
    bm25_results = [{"id": "a", "text": "x"}, {"id": "c", "text": "z"}]
    fused = reciprocal_rank_fusion(vector_results, bm25_results)
    ids = [item["id"] for item in fused]
    assert len(ids) == len(set(ids)) == 3


def test_fusion_handles_empty_lists():
    fused = reciprocal_rank_fusion([], [])
    assert fused == []

    fused = reciprocal_rank_fusion([{"id": "a", "text": "x"}], [])
    assert len(fused) == 1


def test_fusion_falls_back_to_text_when_no_id():
    # BM25/vector results without a stable "id" should dedup on text content.
    list_a = [{"text": "same chunk text"}]
    list_b = [{"text": "same chunk text"}]
    fused = reciprocal_rank_fusion(list_a, list_b)
    assert len(fused) == 1

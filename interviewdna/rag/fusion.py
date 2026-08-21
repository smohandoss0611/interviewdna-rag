"""
Reciprocal Rank Fusion (RRF) -- the standard, simple algorithm for combining
two separately-ranked result lists (here: vector search results and BM25
keyword search results) into one merged ranking.

Why not just average the raw scores? Because vector similarity scores
(cosine similarity, 0-1) and BM25 scores (unbounded, can be 0-20+) are on
completely different scales -- averaging them directly is meaningless.

RRF sidesteps that entirely: it only looks at each item's RANK POSITION
(1st, 2nd, 3rd...) in each list, not its raw score. An item's fused score is:

    sum over each list it appears in of:  1 / (k + rank_in_that_list)

k is a small constant (60 is the standard default from the original RRF
paper) that dampens the impact of being ranked #1 vs #2 -- without it, small
rank differences at the top would swing the fused score too aggressively.

An item that ranks well in BOTH lists (found by both keyword and semantic
search) naturally floats to the top. An item found by only one method still
gets a fair shot, just a lower fused score.
"""
from __future__ import annotations

from typing import List, Dict, Any

RRF_K = 60


def reciprocal_rank_fusion(
    *ranked_lists: List[Dict[str, Any]],
    id_key: str = "id",
    k: int = RRF_K,
) -> List[Dict[str, Any]]:
    """Fuses N ranked lists of dicts (each already sorted best-first) into
    one ranked list, deduplicated by `id_key`. Each returned dict gets an
    extra `_rrf_score` field and a `_found_by` list naming which input
    list(s) it appeared in -- useful for logging/debugging which retrieval
    method actually found the winning result.
    """
    fused_scores: Dict[str, float] = {}
    fused_items: Dict[str, Dict[str, Any]] = {}
    found_by: Dict[str, List[str]] = {}

    for list_idx, ranked_list in enumerate(ranked_lists):
        list_name = ranked_list[0].get("_source_list", f"list_{list_idx}") if ranked_list else f"list_{list_idx}"
        for rank, item in enumerate(ranked_list, start=1):
            item_id = item.get(id_key)
            if item_id is None:
                # Fall back to text content as a dedup key if no stable id
                # is present (e.g. BM25 results built on the fly).
                item_id = item.get("text", "")[:200]
            fused_scores[item_id] = fused_scores.get(item_id, 0.0) + 1.0 / (k + rank)
            if item_id not in fused_items:
                fused_items[item_id] = item
                found_by[item_id] = []
            found_by[item_id].append(list_name)

    ranked_ids = sorted(fused_scores.keys(), key=lambda i: fused_scores[i], reverse=True)
    results = []
    for item_id in ranked_ids:
        item = dict(fused_items[item_id])
        item["_rrf_score"] = fused_scores[item_id]
        item["_found_by"] = found_by[item_id]
        results.append(item)
    return results

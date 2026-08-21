"""
Mem0-backed long-term candidate coaching memory.

IMPORTANT DISTINCTION (spec section 17):
    Pinecone = resume/JD/project/reference KNOWLEDGE (rag/pinecone_store.py)
    Mem0     = long-term candidate COACHING memory (this file)

Mem0 does NOT store every candidate answer. Only meaningful coaching signals
are written, via explicit rules in `derive_memory_writes()`.
"""
from __future__ import annotations

import os
from typing import List, Dict, Any, Optional

MEM0_API_KEY = os.getenv("MEM0_API_KEY")


class Mem0Service:
    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            from mem0 import MemoryClient, Memory

            if MEM0_API_KEY:
                self._client = MemoryClient(api_key=MEM0_API_KEY)
            else:
                # Local/self-hosted fallback so the hackathon app runs without
                # a Mem0 cloud API key.
                self._client = Memory()
        return self._client

    # ------------------------------------------------------------------ #
    def add_memory(self, user_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        client = self._get_client()
        client.add(text, user_id=user_id, metadata=metadata or {})

    def search_memory(self, user_id: str, query: str, limit: int = 5) -> List[str]:
        client = self._get_client()
        try:
            results = client.search(query, user_id=user_id, limit=limit)
        except TypeError:
            # local `Memory()` signature differs slightly across versions
            results = client.search(query=query, user_id=user_id)
        memories = results.get("results", results) if isinstance(results, dict) else results
        out = []
        for r in memories or []:
            if isinstance(r, dict):
                out.append(r.get("memory") or r.get("text") or str(r))
            else:
                out.append(str(r))
        return out


# --------------------------------------------------------------------------- #
# Explicit memory-write rules (Feature: "Use explicit memory-write rules")
# --------------------------------------------------------------------------- #
MIN_GAP_SEVERITY_SCORE = 5  # out of 10; below this on a dimension = worth remembering
STRONG_SCORE_THRESHOLD = 8  # at/above this = worth remembering as a strength


def derive_memory_writes(
    competency: str,
    evaluation: Dict[str, Any],
    agent_action: str,
) -> List[str]:
    """Given one turn's evaluation, decide which (if any) coaching-relevant
    facts are worth persisting to Mem0. Returns a list of short memory strings.

    Rules (deliberately conservative -- NOT every answer is stored):
      - A COACH action always produces a "needs practice" memory for the
        detected_gap/weak dimension.
      - A dimension scoring >= STRONG_SCORE_THRESHOLD across evaluation is
        recorded as a durable strength.
      - A dimension scoring <= MIN_GAP_SEVERITY_SCORE, when the action is
        PROBE or COACH, is recorded as a development area.
    """
    memories: List[str] = []
    scores: Dict[str, int] = evaluation.get("scores") or {
        k: v for k, v in evaluation.items() if isinstance(v, int)
    }

    if agent_action == "COACH":
        gap = evaluation.get("detected_gap") or evaluation.get("weakness")
        if gap:
            memories.append(f"{competency}: needs practice on {gap}")

    for dim, score in scores.items():
        if not isinstance(score, int):
            continue
        if score >= STRONG_SCORE_THRESHOLD:
            memories.append(f"{competency}: strong {dim} ({score}/10)")
        elif score <= MIN_GAP_SEVERITY_SCORE and agent_action in ("PROBE", "COACH"):
            memories.append(f"{competency}: {dim} needs practice ({score}/10)")

    return memories


_service: Optional[Mem0Service] = None


def get_mem0_service() -> Mem0Service:
    global _service
    if _service is None:
        _service = Mem0Service()
    return _service

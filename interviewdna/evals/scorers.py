"""
Scorers: small, boring, deterministic functions that answer "is this AI
output actually good?"

Why deterministic and not "ask another AI to judge it" (an "LLM judge")?
Because deterministic checks are:
  - Free (no extra API/model call)
  - Instant
  - 100% reproducible (same input -> same pass/fail, always)
  - Easy to explain when they fail

LLM-as-judge scoring is a real, valuable technique too (useful for fuzzy
things like "is this coaching text well-written?"), but it's a second step,
not the first one. Start here.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List
from dataclasses import dataclass


@dataclass
class EvalResult:
    name: str
    passed: bool
    detail: str
    elapsed_s: float = 0.0


def contains_any(haystack: Iterable[str], needles: Iterable[str]) -> bool:
    """True if any `needle` appears (case-insensitive, substring) in any
    string in `haystack`. Used for "did the extraction find X?" checks."""
    haystack_lower = " | ".join(str(h) for h in haystack).lower()
    return any(needle.lower() in haystack_lower for needle in needles)


def all_scores_at_most(scores: Dict[str, int], threshold: int) -> bool:
    """True if every value in `scores` is <= threshold. Used for "did the
    evaluator correctly recognize this answer as garbage?" checks."""
    return all(v <= threshold for v in scores.values())


def none_contain(strings: List[str], forbidden_substrings: Iterable[str]) -> List[str]:
    """Returns the list of forbidden substrings that DID leak into `strings`
    (empty list = pass). Used for "did untested JD boilerplate leak into the
    improvement plan?" checks."""
    joined = " | ".join(strings).lower()
    return [f for f in forbidden_substrings if f.lower() in joined]

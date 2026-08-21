"""
Coverage Agent (Feature 9).

Maintains structured coverage state: {competency: NOT_TESTED|PARTIAL|TESTED}
and decides the NEXT competency to test.

Per spec: "The coverage decision can use deterministic rules first. Only use
an LLM if semantic reasoning is genuinely needed. Avoid unnecessary LLM calls."
-> This module is intentionally 100% deterministic / rule-based. It makes NO
LLM calls.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Any

NOT_TESTED = "NOT_TESTED"
PARTIAL = "PARTIAL"
TESTED = "TESTED"

_PRIORITY_RANK = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


def init_coverage(strategy: Dict[str, Any]) -> Dict[str, str]:
    """Seed coverage map from the strategy's priority_competencies list."""
    coverage = {}
    for pc in strategy.get("priority_competencies", []):
        coverage[pc["name"]] = NOT_TESTED
    return coverage


def update_coverage_after_turn(
    coverage: Dict[str, str],
    competency: str,
    agent_action: str,
) -> Dict[str, str]:
    """Deterministic rule: MOVE_ON => TESTED, PROBE/CLARIFY/CHALLENGE/COACH => PARTIAL."""
    if agent_action == "MOVE_ON":
        coverage[competency] = TESTED
    elif competency in coverage:
        coverage[competency] = PARTIAL
    else:
        coverage[competency] = PARTIAL
    return coverage


def select_next_competency(
    strategy: Dict[str, Any],
    coverage: Dict[str, str],
    questions_asked: int,
    question_budget: int,
) -> Optional[str]:
    """Deterministic next-competency selection.

    Rules (spec section 15):
      1. Respect remaining question budget (None if exhausted).
      2. Prefer NOT_TESTED over PARTIAL over TESTED.
      3. Within a tier, prefer higher JD priority (HIGH > MEDIUM > LOW).
      4. Stable tie-break: original strategy order.
    """
    if questions_asked >= question_budget:
        return None

    priorities: List[Dict[str, Any]] = strategy.get("priority_competencies", [])
    if not priorities:
        return None

    def sort_key(pc: Dict[str, Any], idx: int):
        status = coverage.get(pc["name"], NOT_TESTED)
        status_rank = {NOT_TESTED: 0, PARTIAL: 1, TESTED: 2}[status]
        priority_rank = _PRIORITY_RANK.get(pc.get("priority", "MEDIUM"), 1)
        return (status_rank, priority_rank, idx)

    ranked = sorted(
        list(enumerate(priorities)), key=lambda pair: sort_key(pair[1], pair[0])
    )

    for _, pc in ranked:
        if coverage.get(pc["name"], NOT_TESTED) != TESTED:
            return pc["name"]
    return None


def coverage_is_complete(strategy: Dict[str, Any], coverage: Dict[str, str]) -> bool:
    names = [pc["name"] for pc in strategy.get("priority_competencies", [])]
    if not names:
        return True
    return all(coverage.get(n) == TESTED for n in names)

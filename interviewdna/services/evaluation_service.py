"""
Evaluation Service
    - Aggregates all evaluation signals collected across the interview
    - Makes LLM CALL #12 (Personalized Improvement Plan)

Called once at interview completion, from the LangGraph `improvement_plan`
node (agents/interview_graph.py), NOT from FastAPI directly -- this keeps
LangGraph as the single controller of *when* the plan is generated, while
this service owns *how* the data is summarized and the LLM call itself.
"""
from __future__ import annotations

import json
import logging
from typing import Dict, Any, List

from llm.base import LLMService
from models.schemas import ImprovementPlan
from prompts.coaching_prompts import build_improvement_plan_messages

logger = logging.getLogger("interviewdna.services.evaluation")


def _summarize_technical(transcript: List[Dict[str, Any]]) -> str:
    entries = [t for t in transcript if t.get("mode") == "TECHNICAL" and t.get("evaluation")]
    if not entries:
        return "(no technical questions answered)"
    lines = []
    for e in entries:
        scores = e["evaluation"].get("scores", e["evaluation"])
        lines.append(f"- {e.get('competency')}: {scores}")
    return "\n".join(lines)


def _summarize_star(transcript: List[Dict[str, Any]]) -> str:
    entries = [t for t in transcript if t.get("mode") == "BEHAVIORAL_STAR" and t.get("evaluation")]
    if not entries:
        return "(no behavioral STAR questions answered)"
    lines = []
    for e in entries:
        lines.append(f"- {e.get('competency')}: {e['evaluation']}")
    return "\n".join(lines)


def _summarize_technical_star(transcript: List[Dict[str, Any]]) -> str:
    entries = [t for t in transcript if t.get("mode") == "TECHNICAL_STAR" and t.get("evaluation")]
    if not entries:
        return "(no technical STAR deep-dive answered)"
    lines = []
    for e in entries:
        lines.append(f"- {e.get('competency')}: {e['evaluation']}")
    return "\n".join(lines)


def _summarize_retry(before: Dict[str, int] | None, retry: Dict[str, int] | None) -> str:
    if not before or not retry:
        return "(no coach/retry cycle occurred)"
    lines = ["dimension | before | retry"]
    for dim in before:
        lines.append(f"{dim} | {before.get(dim)} | {retry.get(dim, '-')}")
    return "\n".join(lines)


def _filter_alignment_to_covered(alignment: Dict[str, Any], coverage: Dict[str, str]) -> Dict[str, Any]:
    """Restrict the alignment/JD-gap data to only competencies that were
    actually touched (TESTED or PARTIAL) during THIS interview.

    Without this, the improvement plan prompt received the FULL resume<->JD
    alignment -- every requirement in the job description, including things
    like education requirements or soft-skill bullets that were never part
    of the interview at all. A model taking the path of least resistance
    will happily just echo that clean, well-formatted requirement list back
    as "development areas" instead of synthesizing genuine feedback from the
    (sparser, harder to summarize) actual interview transcript -- which is
    exactly what was observed: JD bullets like "Bachelor's or advanced
    degree..." showing up verbatim as a "development area" despite never
    being asked about.

    This is a structural fix, not just a prompt instruction: items for
    competencies that were never touched in this session are removed BEFORE
    the LLM ever sees them, so there's nothing to copy from.
    """
    covered = {name for name, status in coverage.items() if status in ("TESTED", "PARTIAL")}
    if not covered:
        return {"items": []}

    def is_covered(requirement: str) -> bool:
        req_lower = requirement.lower()
        return any(c.lower() in req_lower or req_lower in c.lower() for c in covered)

    items = alignment.get("items", [])
    filtered = [item for item in items if is_covered(item.get("requirement", ""))]
    dropped = len(items) - len(filtered)
    if dropped:
        logger.info(
            "Filtered improvement-plan alignment context: kept %d/%d requirement(s) "
            "actually covered in the interview, dropped %d untested JD-only item(s)",
            len(filtered), len(items), dropped,
        )
    return {"items": filtered}


def generate_improvement_plan(
    llm: LLMService,
    alignment: Dict[str, Any],
    transcript: List[Dict[str, Any]],
    coverage: Dict[str, str],
    before_retry_scores: Dict[str, int] | None,
    retry_scores: Dict[str, int] | None,
    memory_context: List[str],
) -> ImprovementPlan:
    """LLM CALL #12 - Improvement Plan Generation.

    Node/Call site: agents/interview_graph.py `improvement_plan_node`.
    Why: synthesizes every prior signal (alignment gaps, technical/STAR/
    technical-STAR performance, coverage, retry deltas, Mem0 history) into a
    coaching-oriented improvement plan for the candidate.
    """
    scoped_alignment = _filter_alignment_to_covered(alignment, coverage)

    messages = build_improvement_plan_messages(
        alignment_json=json.dumps(scoped_alignment),
        technical_scores_summary=_summarize_technical(transcript),
        star_scores_summary=_summarize_star(transcript),
        technical_star_summary=_summarize_technical_star(transcript),
        coverage_json=json.dumps(coverage),
        retry_summary=_summarize_retry(before_retry_scores, retry_scores),
        memory_context=memory_context,
    )
    plan = llm.invoke_structured(messages, schema=ImprovementPlan)

    logger.info(
        "LLM CALL #12: improvement plan generated -- %d strength(s), %d development area(s), "
        "%d next-practice item(s)",
        len(plan.strengths), len(plan.development_areas), len(plan.next_practice),
    )
    return plan

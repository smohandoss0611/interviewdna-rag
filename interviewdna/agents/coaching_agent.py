"""
Coaching Agent (Feature 7) - Coach -> Retrieve -> Retry.

    Detected Knowledge Gap
          |
          v
    LangGraph COACH node (agents/interview_graph.py)
          |
          v
    Tool-Use Agent decides: knowledge base or web search? (agents/tool_agent.py)
          |
          v
    Selected tool executed -> Relevant Knowledge
          |
          v
    LLM CALL #10 - Grounded Coaching (this module)
          |
          v
    Generate Related Retry Question (part of the same call)
          |
          v
    Candidate Retry
          |
          v
    LLM CALL #11 - Retry Evaluation (this module)
          |
          v
    Compare Performance (before/after)
"""
from __future__ import annotations

from typing import Dict, Any, List

from llm.base import LLMService
from models.schemas import CoachingResult, AnswerEvaluation
from prompts.coaching_prompts import build_coaching_messages, build_retry_eval_messages


def coach_with_retrieved_context(
    llm: LLMService,
    competency: str,
    detected_gap: str,
    original_question: str,
    original_answer: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> CoachingResult:
    """LLM CALL #10 - RAG-Grounded Coaching.

    Node/Call site: agents/interview_graph.py `coach_node`, invoked when the
    routing decision (from answer_quality_agent) is COACH. The node calls
    agents/tool_agent.py FIRST to decide which source to ground this
    explanation in (its own indexed knowledge base, or a live web search)
    and fetch `retrieved_chunks`, then calls this function.
    Why: generates concise coaching text (with preserved source citations)
    plus exactly one related retry question -- all grounded in whatever the
    chosen tool actually returned, never hallucinated.
    """
    messages = build_coaching_messages(
        detected_gap=detected_gap,
        original_question=original_question,
        original_answer=original_answer,
        retrieved_chunks=retrieved_chunks,
    )
    result = llm.invoke_structured(messages, schema=CoachingResult)

    # Defensive: if the LLM didn't populate sources but we did retrieve
    # context, attach the top sources so citations are preserved (spec:
    # "Preserve citations/source metadata").
    if not result.sources and retrieved_chunks:
        from models.schemas import CoachingSource

        result.sources = [
            CoachingSource(source=c.get("source", "reference"), snippet=c.get("text", "")[:200])
            for c in retrieved_chunks[:2]
        ]
    return result


def evaluate_retry(
    llm: LLMService, retry_question: str, retry_answer: str, competency: str
) -> AnswerEvaluation:
    """LLM CALL #11 - Retry Evaluation.

    Node/Call site: agents/interview_graph.py `retry_evaluate_node`, invoked
    after the candidate answers the coaching-generated retry question.
    Why: re-scores using the SAME rubric as LLM CALL #6 so before/after can
    be directly compared in the Streamlit "Results" view.
    """
    messages = build_retry_eval_messages(
        retry_question=retry_question, retry_answer=retry_answer, competency=competency
    )
    return llm.invoke_structured(messages, schema=AnswerEvaluation)


def compare_before_retry(before: Dict[str, Any], retry: AnswerEvaluation) -> Dict[str, Dict[str, int]]:
    """Deterministic comparison -- no LLM call. Produces the BEFORE/RETRY table
    shown in the Streamlit UI (spec section 13)."""
    retry_scores = {
        "correctness": retry.correctness,
        "depth": retry.depth,
        "clarity": retry.clarity,
        "evidence": retry.evidence,
        "tradeoffs": retry.tradeoffs,
        "completeness": retry.completeness,
    }
    before_scores = before.get("scores", before)
    return {"before": before_scores, "retry": retry_scores}

"""
Answer Quality Agent (Feature 8) - wraps LLM CALL #6 (Candidate Answer Evaluation).

Per spec: "The Answer Quality Agent must NOT simply return a score. It must
produce structured evidence that LangGraph can use for routing." This module
returns an AnswerQualityResult (scores + strength + weakness +
recommended_action) that agents/interview_graph.py's routing function reads
directly to pick CLARIFY/PROBE/CHALLENGE/COACH/MOVE_ON.
"""
from __future__ import annotations

from llm.base import LLMService
from models.schemas import AnswerEvaluation, AnswerQualityResult
from prompts.interview_prompts import build_answer_eval_messages

# A dimension only gets called out as a genuine "strength" if it actually
# clears this bar. Without this, "strength" was just whichever dimension
# scored highest -- so a uniformly weak answer (e.g. 2/10 across the board)
# would get told its "strength" was "correctness (2/10)", which is false and
# actively bad coaching feedback.
STRENGTH_THRESHOLD = 6


def evaluate_answer(llm: LLMService, question: str, answer: str, competency: str) -> AnswerQualityResult:
    """LLM CALL #6 - Candidate Answer Evaluation.

    Node/Call site: agents/interview_graph.py `evaluate_answer_node`, invoked
    every time the candidate submits an answer to a TECHNICAL question.
    Why: produces the scored, structured evidence (correctness/depth/clarity/
    evidence/tradeoffs/completeness + recommended_action) that the
    orchestrator's conditional edge uses to route to CLARIFY/PROBE/CHALLENGE/
    COACH/MOVE_ON.
    """
    messages = build_answer_eval_messages(question=question, answer=answer, competency=competency)
    raw: AnswerEvaluation = llm.invoke_structured(messages, schema=AnswerEvaluation)

    scores = {
        "correctness": raw.correctness,
        "depth": raw.depth,
        "clarity": raw.clarity,
        "evidence": raw.evidence,
        "tradeoffs": raw.tradeoffs,
        "completeness": raw.completeness,
    }
    strongest_dim = max(scores, key=scores.get)
    weakest_dim = min(scores, key=scores.get)

    if scores[strongest_dim] >= STRENGTH_THRESHOLD:
        strength_text = f"Strong {strongest_dim} ({scores[strongest_dim]}/10)"
    else:
        # Honest framing: nothing here actually clears the bar for a
        # "strength" -- say so, rather than dressing up the least-bad score.
        strength_text = (
            f"No clear strength in this answer \u2014 comparatively highest was "
            f"{strongest_dim} ({scores[strongest_dim]}/10)"
        )

    return AnswerQualityResult(
        scores=scores,
        strength=strength_text,
        weakness=(
            f"{raw.detected_gap} ({weakest_dim}: {scores[weakest_dim]}/10)"
            if raw.detected_gap
            else f"Relatively weaker {weakest_dim} ({scores[weakest_dim]}/10)"
        ),
        recommended_action=raw.recommended_action,
    )

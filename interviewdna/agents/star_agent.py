"""
STAR Agent (Feature 5) - Behavioral interview questions + LLM CALL #8 (STAR Evaluation).
"""
from __future__ import annotations

from llm.base import LLMService
from models.schemas import GeneratedQuestion, STAREvaluation
from prompts.star_prompts import build_star_question_messages, build_star_eval_messages


def generate_star_question(
    llm: LLMService, competency: str, resume_evidence: str, difficulty: str
) -> GeneratedQuestion:
    """Behavioral question generation for a STAR competency (part of the same
    'question generation' family as LLM CALL #5, scoped to behavioral mode).

    Node/Call site: agents/interview_graph.py `generate_star_question_node`.
    """
    messages = build_star_question_messages(
        competency=competency, resume_evidence=resume_evidence, difficulty=difficulty
    )
    return llm.invoke_structured(messages, schema=GeneratedQuestion)


def evaluate_star_answer(llm: LLMService, question: str, answer: str, competency: str) -> STAREvaluation:
    """LLM CALL #8 - STAR Evaluation.

    Node/Call site: agents/interview_graph.py `evaluate_star_answer_node`,
    invoked whenever the candidate answers a BEHAVIORAL_STAR question.
    Why: scores Situation/Task/Action/Result independently and identifies
    the weakest_component so the orchestrator can decide which part of the
    story to probe deeper (PROBE_SITUATION/TASK/ACTION/RESULT) or MOVE_ON.
    """
    messages = build_star_eval_messages(question=question, answer=answer, competency=competency)
    return llm.invoke_structured(messages, schema=STAREvaluation)

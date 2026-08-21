"""
Technical STAR Agent (Feature 6) - Deep-dive questions grounded in a specific
resume claim/project/achievement, plus LLM CALL #9 (Technical STAR Evaluation).
"""
from __future__ import annotations

from typing import Optional, List, Dict, Any

from llm.base import LLMService
from models.schemas import GeneratedQuestion, TechnicalSTAREvaluation
from prompts.star_prompts import (
    build_technical_star_question_messages,
    build_technical_star_eval_messages,
)


def select_resume_claim(resume_dna: Dict[str, Any]) -> Optional[str]:
    """Deterministic selection of a concrete, real claim to probe -- prefers a
    quantifiable accomplishment (e.g. 'Reduced API latency by 40%'), falling
    back to a project description. Never invents a claim; returns None if the
    resume has nothing usable."""
    quantifiable = resume_dna.get("quantifiable_accomplishments") or []
    if quantifiable:
        return quantifiable[0]
    projects = resume_dna.get("projects") or []
    if projects:
        p = projects[0]
        if isinstance(p, dict):
            desc = p.get("description") or p.get("name")
            if desc:
                return desc
        elif isinstance(p, str):
            return p
    claims = resume_dna.get("technical_claims") or []
    if claims:
        return claims[0]
    return None


def generate_technical_star_question(
    llm: LLMService, competency: str, resume_claim: str, difficulty: str
) -> GeneratedQuestion:
    """Node/Call site: agents/interview_graph.py `generate_technical_star_question_node`."""
    messages = build_technical_star_question_messages(
        competency=competency, resume_claim=resume_claim, difficulty=difficulty
    )
    return llm.invoke_structured(messages, schema=GeneratedQuestion)


def evaluate_technical_star_answer(
    llm: LLMService, question: str, answer: str, resume_claim: str
) -> TechnicalSTAREvaluation:
    """LLM CALL #9 - Technical STAR Evaluation.

    Node/Call site: agents/interview_graph.py `evaluate_technical_star_node`.
    Why: scores both the STAR narrative AND technical depth dimensions
    (architecture, decisions, tradeoffs, scalability, metrics) so the
    orchestrator can dynamically select the next probe dimension.
    """
    messages = build_technical_star_eval_messages(
        question=question, answer=answer, resume_claim=resume_claim
    )
    return llm.invoke_structured(messages, schema=TechnicalSTAREvaluation)

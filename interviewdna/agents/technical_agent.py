"""
Technical Agent - LangGraph nodes wrapping:
    LLM CALL #5 - Interview Question Generation
    LLM CALL #7 - Adaptive Follow-Up Generation

Retrieval (LlamaIndex -> Pinecone via rag/retriever.py) happens BEFORE these
calls; the retrieved context is passed in so this module stays focused on
prompting/parsing the LLM response.
"""
from __future__ import annotations

from typing import Optional

from llm.base import LLMService
from models.schemas import GeneratedQuestion
from prompts.interview_prompts import build_question_messages, build_followup_messages


def generate_question(
    llm: LLMService,
    competency: str,
    job_requirement: str,
    resume_evidence: str,
    retrieved_context: str,
    difficulty: str,
    previous_question: Optional[str] = None,
) -> GeneratedQuestion:
    """LLM CALL #5 - Interview Question Generation.

    Node/Call site: agents/interview_graph.py `generate_question_node`,
    invoked after `retrieve_context_node` selects and fetches relevant
    resume/reference chunks for `competency`.
    Why: generates exactly one adaptive question grounded in the JD
    requirement + retrieved evidence + current difficulty.
    """
    messages = build_question_messages(
        competency=competency,
        job_requirement=job_requirement,
        resume_evidence=resume_evidence,
        retrieved_context=retrieved_context,
        difficulty=difficulty,
        previous_question=previous_question,
    )
    return llm.invoke_structured(messages, schema=GeneratedQuestion)


def generate_followup(
    llm: LLMService,
    action: str,
    competency: str,
    previous_question: str,
    previous_answer: str,
    detected_gap: Optional[str],
    difficulty: str,
) -> GeneratedQuestion:
    """LLM CALL #7 - Adaptive Follow-Up Generation.

    Node/Call site: agents/interview_graph.py `followup_node`, invoked when
    the routing decision (from answer_quality_agent) is CLARIFY, PROBE, or
    CHALLENGE.
    Why: LangGraph has already DECIDED the follow-up type; this call only
    generates the natural-language question for that decision.
    """
    messages = build_followup_messages(
        action=action,
        competency=competency,
        previous_question=previous_question,
        previous_answer=previous_answer,
        detected_gap=detected_gap,
        difficulty=difficulty,
    )
    return llm.invoke_structured(messages, schema=GeneratedQuestion)


def escalate_difficulty(current_difficulty: str) -> str:
    order = ["EASY", "MEDIUM", "HARD"]
    idx = order.index(current_difficulty) if current_difficulty in order else 1
    return order[min(idx + 1, len(order) - 1)]


def de_escalate_difficulty(current_difficulty: str) -> str:
    order = ["EASY", "MEDIUM", "HARD"]
    idx = order.index(current_difficulty) if current_difficulty in order else 1
    return order[max(idx - 1, 0)]

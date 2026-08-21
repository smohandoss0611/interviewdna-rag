"""
InterviewState is the single shared state object that flows through the
LangGraph graph defined in agents/interview_graph.py.

LangGraph nodes read from and write partial updates to this state. Nothing
outside LangGraph (FastAPI, Streamlit) is allowed to mutate interview logic
directly -- they only ever read/write state through the graph.
"""
from __future__ import annotations

import logging
from typing import TypedDict, List, Dict, Optional, Any

_activity_logger = logging.getLogger("interviewdna.agent")


class TranscriptEntry(TypedDict, total=False):
    competency: str
    mode: str                     # TECHNICAL | BEHAVIORAL_STAR | TECHNICAL_STAR
    question: str
    answer: str
    evaluation: Dict[str, Any]
    agent_action: str
    difficulty: str
    is_followup: bool
    is_retry: bool


class InterviewState(TypedDict, total=False):
    # --- identity -----------------------------------------------------
    session_id: str
    user_id: str

    # --- ingested source material --------------------------------------
    resume_text: str
    jd_text: str

    # --- DNA / alignment / strategy (LLM CALLS #1-#4) -------------------
    resume_dna: Dict[str, Any]
    job_dna: Dict[str, Any]
    alignment: Dict[str, Any]
    strategy: Dict[str, Any]

    # --- interview configuration -----------------------------------------
    mode: str                       # InterviewMode value
    question_budget: int
    questions_asked: int

    # --- coverage state (Feature 9) --------------------------------------
    coverage: Dict[str, str]        # competency -> CoverageStatus value

    # --- current turn -----------------------------------------------------
    current_competency: Optional[str]
    current_sub_mode: Optional[str]   # TECHNICAL | STAR | TECHNICAL_STAR for this turn
    current_question: Optional[str]
    current_difficulty: str
    current_retrieved_context: List[Dict[str, Any]]
    current_answer: Optional[str]

    # --- evaluation results for the current turn --------------------------
    last_evaluation: Optional[Dict[str, Any]]        # AnswerQualityResult
    last_star_evaluation: Optional[Dict[str, Any]]   # STAREvaluation
    last_technical_star_evaluation: Optional[Dict[str, Any]]
    agent_action: Optional[str]       # CLARIFY|PROBE|CHALLENGE|COACH|MOVE_ON|...

    # --- coach -> retrieve -> retry (Feature 7) ---------------------------
    coaching_result: Optional[Dict[str, Any]]
    before_retry_scores: Optional[Dict[str, int]]
    retry_scores: Optional[Dict[str, int]]

    # --- transcript / memory / results -------------------------------------
    transcript: List[TranscriptEntry]
    memory_context: List[str]           # Mem0 recalled memories relevant to session
    improvement_plan: Optional[Dict[str, Any]]

    # --- UI-facing agentic activity feed (Feature: "Agent Activity") -------
    activity_log: List[Dict[str, str]]

    # --- control flags -------------------------------------------------------
    interview_complete: bool
    awaiting_answer: bool
    awaiting_retry: bool


def new_state(session_id: str, user_id: str = "candidate") -> InterviewState:
    """Factory for a fresh InterviewState."""
    return InterviewState(
        session_id=session_id,
        user_id=user_id,
        resume_text="",
        jd_text="",
        resume_dna={},
        job_dna={},
        alignment={},
        strategy={},
        mode="MIXED",
        question_budget=8,
        questions_asked=0,
        coverage={},
        current_competency=None,
        current_sub_mode=None,
        current_question=None,
        current_difficulty="MEDIUM",
        current_retrieved_context=[],
        current_answer=None,
        last_evaluation=None,
        last_star_evaluation=None,
        last_technical_star_evaluation=None,
        agent_action=None,
        coaching_result=None,
        before_retry_scores=None,
        retry_scores=None,
        transcript=[],
        memory_context=[],
        improvement_plan=None,
        activity_log=[],
        interview_complete=False,
        awaiting_answer=False,
        awaiting_retry=False,
    )


def log_activity(state: InterviewState, label: str, detail: str = "") -> None:
    """Append a concise, non-chain-of-thought event to the activity feed
    (shown in the Streamlit "Agent Activity" panel), AND emit it to the
    backend logger so it's visible live in the terminal/log file too --
    this is what lets you watch the agentic loop's decisions in real time
    while an interview is running.
    """
    state.setdefault("activity_log", []).append({"label": label, "detail": detail})
    session_id = state.get("session_id", "?")
    _activity_logger.info("[session=%s] %s | %s", session_id, label, detail)

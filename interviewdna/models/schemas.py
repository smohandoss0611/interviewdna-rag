"""
Pydantic schemas for InterviewDNA.

These models are used in two places:
1. As the `schema` argument to LLMService.invoke_structured(...) so that every
   LLM call in the system (see agents/*.py, services/*.py) returns validated,
   structured data instead of free text.
2. As FastAPI request/response models (api/routes/*.py).
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional, Dict, Literal
from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Resume DNA  (LLM CALL #1 - services/resume_service.py)
# --------------------------------------------------------------------------- #
class ResumeProject(BaseModel):
    name: str
    description: str = ""
    technologies: List[str] = Field(default_factory=list)
    quantified_impact: Optional[str] = None


class ResumeDNA(BaseModel):
    skills: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    frameworks: List[str] = Field(default_factory=list)
    databases: List[str] = Field(default_factory=list)
    cloud_technologies: List[str] = Field(default_factory=list)
    work_experience: List[str] = Field(default_factory=list)
    projects: List[ResumeProject] = Field(default_factory=list)
    achievements: List[str] = Field(default_factory=list)
    leadership_evidence: List[str] = Field(default_factory=list)
    quantifiable_accomplishments: List[str] = Field(default_factory=list)
    technical_claims: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Job DNA  (LLM CALL #2 - services/job_service.py)
# --------------------------------------------------------------------------- #
class SeniorityLevel(str, Enum):
    JUNIOR = "JUNIOR"
    MID = "MID"
    SENIOR = "SENIOR"
    STAFF_PLUS = "STAFF_PLUS"


class JobDNA(BaseModel):
    required_skills: List[str] = Field(default_factory=list)
    preferred_skills: List[str] = Field(default_factory=list)
    responsibilities: List[str] = Field(default_factory=list)
    technical_competencies: List[str] = Field(default_factory=list)
    behavioral_competencies: List[str] = Field(default_factory=list)
    technologies: List[str] = Field(default_factory=list)
    seniority_expectation: SeniorityLevel = SeniorityLevel.MID


# --------------------------------------------------------------------------- #
# Alignment  (LLM CALL #3 - services/job_service.py)
# --------------------------------------------------------------------------- #
class EvidenceLevel(str, Enum):
    STRONG_EVIDENCE = "STRONG_EVIDENCE"
    PARTIAL_EVIDENCE = "PARTIAL_EVIDENCE"
    NOT_DEMONSTRATED = "NOT_DEMONSTRATED"


class AlignmentItem(BaseModel):
    requirement: str
    evidence_level: EvidenceLevel
    supporting_resume_evidence: Optional[str] = None
    rationale: str = ""


class AlignmentResult(BaseModel):
    items: List[AlignmentItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Interview Strategy  (LLM CALL #4 - agents/strategy_agent.py)
# --------------------------------------------------------------------------- #
class PriorityLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class PriorityCompetency(BaseModel):
    name: str
    priority: PriorityLevel
    reason: str


class InterviewMode(str, Enum):
    TECHNICAL = "TECHNICAL"
    BEHAVIORAL_STAR = "BEHAVIORAL_STAR"
    TECHNICAL_STAR = "TECHNICAL_STAR"
    MIXED = "MIXED"


class InterviewStrategy(BaseModel):
    technical: int = Field(ge=0, le=100)
    behavioral: int = Field(ge=0, le=100)
    priority_competencies: List[PriorityCompetency] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Question generation (LLM CALL #5, #7 - agents/technical_agent.py)
# --------------------------------------------------------------------------- #
class GeneratedQuestion(BaseModel):
    competency: str
    question: str
    difficulty: Literal["EASY", "MEDIUM", "HARD"] = "MEDIUM"
    rationale: str = ""


# --------------------------------------------------------------------------- #
# Answer evaluation (LLM CALL #6, #11 - agents/answer_quality_agent.py)
# --------------------------------------------------------------------------- #
class RecommendedAction(str, Enum):
    CLARIFY = "CLARIFY"
    PROBE = "PROBE"
    CHALLENGE = "CHALLENGE"
    COACH = "COACH"
    MOVE_ON = "MOVE_ON"


class AnswerEvaluation(BaseModel):
    correctness: int = Field(ge=0, le=10)
    depth: int = Field(ge=0, le=10)
    clarity: int = Field(ge=0, le=10)
    evidence: int = Field(ge=0, le=10)
    tradeoffs: int = Field(ge=0, le=10)
    completeness: int = Field(ge=0, le=10)
    detected_gap: Optional[str] = None
    recommended_action: RecommendedAction


# --------------------------------------------------------------------------- #
# Answer Quality Agent output (Feature 8) - wraps AnswerEvaluation w/ narrative
# --------------------------------------------------------------------------- #
class AnswerQualityResult(BaseModel):
    scores: Dict[str, int]
    strength: str
    weakness: str
    recommended_action: RecommendedAction


# --------------------------------------------------------------------------- #
# STAR evaluation (LLM CALL #8 - agents/star_agent.py)
# --------------------------------------------------------------------------- #
class STARWeakest(str, Enum):
    SITUATION = "SITUATION"
    TASK = "TASK"
    ACTION = "ACTION"
    RESULT = "RESULT"


class STARAction(str, Enum):
    PROBE_SITUATION = "PROBE_SITUATION"
    PROBE_TASK = "PROBE_TASK"
    PROBE_ACTION = "PROBE_ACTION"
    PROBE_RESULT = "PROBE_RESULT"
    MOVE_ON = "MOVE_ON"


class STAREvaluation(BaseModel):
    situation: int = Field(ge=0, le=10)
    task: int = Field(ge=0, le=10)
    action: int = Field(ge=0, le=10)
    result: int = Field(ge=0, le=10)
    weakest_component: STARWeakest
    recommended_action: STARAction


# --------------------------------------------------------------------------- #
# Technical STAR evaluation (LLM CALL #9 - agents/technical_star_agent.py)
# --------------------------------------------------------------------------- #
class TechnicalSTARScores(BaseModel):
    situation: int = Field(ge=0, le=10)
    task: int = Field(ge=0, le=10)
    action: int = Field(ge=0, le=10)
    result: int = Field(ge=0, le=10)


class TechnicalDimensionScores(BaseModel):
    architecture: int = Field(ge=0, le=10)
    decisions: int = Field(ge=0, le=10)
    tradeoffs: int = Field(ge=0, le=10)
    scalability: int = Field(ge=0, le=10)
    metrics: int = Field(ge=0, le=10)


class TechnicalSTAREvaluation(BaseModel):
    star: TechnicalSTARScores
    technical: TechnicalDimensionScores
    weakest_dimension: str
    recommended_action: str


# --------------------------------------------------------------------------- #
# Tool-use agent (tools/, agents/tool_agent.py) -- NEW, extends LLM CALL #10
# --------------------------------------------------------------------------- #
class ToolDecision(BaseModel):
    """Structured output of the tool-selection LLM call: given a knowledge
    gap, which tool (if any) should be used to find grounding information,
    and what should be searched for. This is what makes it a genuine
    "agent decides" step rather than always calling the same retrieval
    function -- the LLM chooses based on the gap's nature."""
    tool_name: str  # must match a registered tool's `name`, or "none"
    query: str = ""  # the search query to send to the chosen tool
    reasoning: str = ""  # brief, one sentence -- shown in the Agent Activity feed


# --------------------------------------------------------------------------- #
# Coaching (LLM CALL #10 - agents/coaching_agent.py)
# --------------------------------------------------------------------------- #
class CoachingSource(BaseModel):
    source: str
    snippet: str


class CoachingResult(BaseModel):
    coaching_text: str
    sources: List[CoachingSource] = Field(default_factory=list)
    retry_question: str


# --------------------------------------------------------------------------- #
# Improvement Plan (LLM CALL #12 - services/evaluation_service.py)
# --------------------------------------------------------------------------- #
class ImprovementPlan(BaseModel):
    strengths: List[str] = Field(default_factory=list)
    development_areas: List[str] = Field(default_factory=list)
    next_practice: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Coverage
# --------------------------------------------------------------------------- #
class CoverageStatus(str, Enum):
    NOT_TESTED = "NOT_TESTED"
    PARTIAL = "PARTIAL"
    TESTED = "TESTED"


# --------------------------------------------------------------------------- #
# API request/response models
# --------------------------------------------------------------------------- #
class ResumeAnalyzeResponse(BaseModel):
    session_id: str
    resume_dna: ResumeDNA


class JobAnalyzeRequest(BaseModel):
    session_id: str
    job_description_text: str


class JobAnalyzeResponse(BaseModel):
    session_id: str
    job_dna: JobDNA


class MatchRequest(BaseModel):
    session_id: str


class MatchResponse(BaseModel):
    session_id: str
    alignment: AlignmentResult
    strategy: InterviewStrategy


class InterviewStartRequest(BaseModel):
    session_id: str
    mode: InterviewMode = InterviewMode.MIXED
    question_budget: int = 8


class ActivityEvent(BaseModel):
    label: str
    detail: str = ""


class InterviewStartResponse(BaseModel):
    session_id: str
    competency: str
    question: str
    difficulty: str
    activity_log: List[ActivityEvent] = Field(default_factory=list)


class InterviewAnswerRequest(BaseModel):
    session_id: str
    answer_text: str


class InterviewAnswerResponse(BaseModel):
    session_id: str
    evaluation: Optional[AnswerQualityResult] = None
    star_evaluation: Optional[STAREvaluation] = None
    technical_star_evaluation: Optional[TechnicalSTAREvaluation] = None
    agent_action: str
    next_question: Optional[str] = None
    next_competency: Optional[str] = None
    coaching: Optional[CoachingResult] = None
    interview_complete: bool = False
    activity_log: List[ActivityEvent] = Field(default_factory=list)


class InterviewRetryRequest(BaseModel):
    session_id: str
    retry_answer_text: str


class InterviewRetryResponse(BaseModel):
    session_id: str
    before: Dict[str, int]
    retry: Dict[str, int]
    # These were previously missing, which meant the graph's outcome after a
    # retry (finished vs. more questions remain) was silently dropped -- the
    # frontend had no way to know whether to show a new question or move to
    # Results, so it did neither.
    interview_complete: bool = False
    next_question: Optional[str] = None
    next_competency: Optional[str] = None
    activity_log: List[ActivityEvent] = Field(default_factory=list)


class CoverageResponse(BaseModel):
    session_id: str
    coverage: Dict[str, CoverageStatus]


class ResultsResponse(BaseModel):
    session_id: str
    improvement_plan: ImprovementPlan
    coverage: Dict[str, CoverageStatus]
    transcript_summary: List[Dict]

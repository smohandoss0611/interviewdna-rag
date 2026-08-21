"""
Eval cases.

Each function takes the REAL, configured LLMService (whatever's in your
.env -- Ollama, OpenAI, Groq, whatever) and runs one real call through it,
then checks the output with the boring deterministic scorers in scorers.py.

Every case here is modeled directly on a bug we found MANUALLY, by clicking
through the UI, during this project's build. That's the point of an eval
harness: these same 4 checks now run in ~30 seconds instead of "upload a
resume, click through 3 tabs, eyeball the JSON, notice something's off."
"""
from __future__ import annotations

import time
from typing import List

from evals.scorers import EvalResult, contains_any, all_scores_at_most, none_contain
from llm.base import LLMService
from models.schemas import ResumeDNA, JobDNA, AlignmentResult
from services.resume_service import extract_resume_dna
from services.job_service import extract_job_dna
from agents.answer_quality_agent import evaluate_answer
from services.evaluation_service import generate_improvement_plan


# --------------------------------------------------------------------------- #
# Self-contained sample data (no file I/O -- these evals should run anywhere,
# on a fresh clone, with no setup beyond a configured LLM).
# --------------------------------------------------------------------------- #
SAMPLE_RESUME_TEXT = """
Jordan Reyes -- Data Engineer, 4 years of experience.

Skills: Python, SQL, Apache Airflow, dbt, AWS (S3, Lambda, Glue), PostgreSQL, Snowflake.

Experience -- Data Engineer, Northwind Analytics, 2022-Present:
- Built and maintained ETL pipelines using Airflow and dbt.
- Migrated the analytics data warehouse to Snowflake.
- Reduced nightly pipeline runtime by 46% through parallelizing independent DAG branches.
"""

SAMPLE_JD_TEXT = """
Data Engineer -- Requirements:
- Strong Python and SQL skills.
- Experience with a modern ELT stack such as Airflow or dbt.
- Experience with Kubernetes and container orchestration.
- Bachelor's or advanced degree in Computer Science or a related field.
"""


def _flatten_resume(dna: ResumeDNA) -> List[str]:
    """Combine every list field into one bag of strings, so the eval doesn't
    care WHICH bucket the model filed a skill under -- only whether it
    extracted the fact at all."""
    parts = (
        list(dna.skills) + list(dna.languages) + list(dna.frameworks)
        + list(dna.databases) + list(dna.cloud_technologies) + list(dna.work_experience)
        + list(dna.achievements) + list(dna.technical_claims) + list(dna.quantifiable_accomplishments)
    )
    parts += [f"{p.name} {p.description}" for p in dna.projects]
    return parts


def _flatten_job(dna: JobDNA) -> List[str]:
    return (
        list(dna.required_skills) + list(dna.preferred_skills) + list(dna.responsibilities)
        + list(dna.technical_competencies) + list(dna.technologies)
    )


# --------------------------------------------------------------------------- #
# Case 1 -- modeled on: "Resume DNA came back completely empty"
# --------------------------------------------------------------------------- #
def eval_resume_extraction_finds_obvious_skills(llm: LLMService) -> EvalResult:
    start = time.monotonic()
    try:
        dna = extract_resume_dna(llm, SAMPLE_RESUME_TEXT)
        flat = _flatten_resume(dna)
        found = contains_any(flat, ["Python"]) and contains_any(flat, ["SQL"]) and contains_any(flat, ["AWS"])
        detail = f"skills={dna.skills}, languages={dna.languages}, cloud={dna.cloud_technologies}"
        return EvalResult("resume_extraction_finds_obvious_skills", found, detail, time.monotonic() - start)
    except Exception as exc:
        return EvalResult("resume_extraction_finds_obvious_skills", False, f"EXCEPTION: {exc}", time.monotonic() - start)


# --------------------------------------------------------------------------- #
# Case 2 -- same bug class, but for Job DNA extraction (LLM CALL #2)
# --------------------------------------------------------------------------- #
def eval_job_extraction_finds_required_skills(llm: LLMService) -> EvalResult:
    start = time.monotonic()
    try:
        dna = extract_job_dna(llm, SAMPLE_JD_TEXT)
        flat = _flatten_job(dna)
        found = contains_any(flat, ["Python"]) and contains_any(flat, ["Kubernetes"])
        detail = f"required_skills={dna.required_skills}, technologies={dna.technologies}"
        return EvalResult("job_extraction_finds_required_skills", found, detail, time.monotonic() - start)
    except Exception as exc:
        return EvalResult("job_extraction_finds_required_skills", False, f"EXCEPTION: {exc}", time.monotonic() - start)


# --------------------------------------------------------------------------- #
# Case 3 -- modeled on: "pasted JD text as an answer still got a positive score"
# --------------------------------------------------------------------------- #
def eval_offtopic_answer_scores_near_zero(llm: LLMService) -> EvalResult:
    start = time.monotonic()
    try:
        result = evaluate_answer(
            llm,
            question="Can you walk me through how you optimized a data pipeline using Python and SQL?",
            answer=SAMPLE_JD_TEXT,  # a real answer would never look like this -- it's copy-pasted JD text
            competency="Strong SQL and Python skills",
        )
        scores_ok = all_scores_at_most(result.scores, threshold=3)
        strength_honest = "strong" not in result.strength.lower()
        passed = scores_ok and strength_honest
        detail = f"scores={result.scores}, strength={result.strength!r}"
        return EvalResult("offtopic_answer_scores_near_zero", passed, detail, time.monotonic() - start)
    except Exception as exc:
        return EvalResult("offtopic_answer_scores_near_zero", False, f"EXCEPTION: {exc}", time.monotonic() - start)


# --------------------------------------------------------------------------- #
# Case 4 -- modeled on: "untested JD requirements (degree, Kubernetes) leaked
# into the improvement plan's strengths/development areas"
# --------------------------------------------------------------------------- #
def eval_improvement_plan_excludes_untested_requirements(llm: LLMService) -> EvalResult:
    start = time.monotonic()
    try:
        alignment = {
            "items": [
                {"requirement": "Strong Python and SQL skills", "evidence_level": "STRONG_EVIDENCE"},
                {"requirement": "Bachelor's or advanced degree in Computer Science", "evidence_level": "STRONG_EVIDENCE"},
                {"requirement": "Experience with Kubernetes and container orchestration", "evidence_level": "NOT_DEMONSTRATED"},
            ]
        }
        # Only the Python/SQL competency was actually touched this session.
        coverage = {"Strong Python and SQL skills": "TESTED"}
        transcript = [
            {
                "competency": "Strong Python and SQL skills",
                "mode": "TECHNICAL",
                "question": "Walk me through a pipeline you optimized with Python and SQL.",
                "answer": "I pushed joins into SQL and used Python for orchestration, cutting runtime by 40%.",
                "evaluation": {"scores": {"correctness": 8, "depth": 7, "clarity": 8,
                                            "evidence": 7, "tradeoffs": 6, "completeness": 7}},
                "agent_action": "MOVE_ON",
            }
        ]
        plan = generate_improvement_plan(
            llm, alignment=alignment, transcript=transcript, coverage=coverage,
            before_retry_scores=None, retry_scores=None, memory_context=[],
        )
        all_text = plan.strengths + plan.development_areas
        leaked = none_contain(all_text, ["degree", "kubernetes"])
        passed = len(leaked) == 0
        detail = f"strengths={plan.strengths}, development_areas={plan.development_areas}, leaked={leaked}"
        return EvalResult("improvement_plan_excludes_untested_requirements", passed, detail, time.monotonic() - start)
    except Exception as exc:
        return EvalResult(
            "improvement_plan_excludes_untested_requirements", False, f"EXCEPTION: {exc}", time.monotonic() - start
        )


ALL_CASES = [
    eval_resume_extraction_finds_obvious_skills,
    eval_job_extraction_finds_required_skills,
    eval_offtopic_answer_scores_near_zero,
    eval_improvement_plan_excludes_untested_requirements,
]

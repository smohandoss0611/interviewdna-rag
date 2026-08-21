"""
Job Service
    - Parses / accepts the target job description text
    - Makes LLM CALL #2 (Job DNA Extraction)
    - Makes LLM CALL #3 (Resume <-> Job Alignment Reasoning)
    - Indexes JD chunks into Pinecone for later retrieval
"""
from __future__ import annotations

import logging
import time

from llm.base import LLMService
from models.schemas import JobDNA, ResumeDNA, AlignmentResult
from prompts.job_prompts import build_job_dna_messages, build_alignment_messages
from rag.ingestion import ingest_text

logger = logging.getLogger("interviewdna.services.job")


def extract_job_dna(llm: LLMService, jd_text: str) -> JobDNA:
    """LLM CALL #2 - Job DNA Extraction.

    Node/Call site: services/job_service.py -> api/routes/job.py.
    Why: produces the structured requirement list that everything downstream
    (alignment, strategy, question targeting, coverage) is measured against.
    """
    logger.info("LLM CALL #2: Job DNA extraction starting (%d chars input)", len(jd_text))
    logger.debug("Job description text preview (first 300 chars): %r", jd_text[:300])
    start = time.monotonic()
    messages = build_job_dna_messages(jd_text)
    dna = llm.invoke_structured(messages, schema=JobDNA)
    logger.info(
        "LLM CALL #2: done in %.1fs -- %d required skills, %d responsibilities",
        time.monotonic() - start, len(dna.required_skills), len(dna.responsibilities),
    )
    return dna


def align_resume_to_job(llm: LLMService, job_dna: JobDNA, resume_dna: ResumeDNA) -> AlignmentResult:
    """LLM CALL #3 - Resume <-> Job Alignment Reasoning.

    Node/Call site: services/job_service.py -> api/routes/match.py.
    Why: classifies each JD requirement as STRONG_EVIDENCE / PARTIAL_EVIDENCE /
    NOT_DEMONSTRATED, which directly feeds LLM CALL #4 (Strategy) and the
    Coverage Agent's prioritization.
    """
    logger.info("LLM CALL #3: Alignment reasoning starting")
    start = time.monotonic()
    messages = build_alignment_messages(
        job_dna_json=job_dna.model_dump_json(),
        resume_dna_json=resume_dna.model_dump_json(),
    )
    result = llm.invoke_structured(messages, schema=AlignmentResult)
    logger.info(
        "LLM CALL #3: done in %.1fs -- %d requirement(s) classified",
        time.monotonic() - start, len(result.items),
    )
    return result


def index_job_description(jd_text: str, user_id: str, session_id: str) -> int:
    logger.info("Indexing job description into Pinecone (session=%s)", session_id)
    start = time.monotonic()
    count = ingest_text(
        text=jd_text,
        user_id=user_id,
        session_id=session_id,
        document_type="job_description",
        source="job_description",
    )
    logger.info("Indexed %d JD chunk(s) in %.1fs", count, time.monotonic() - start)
    return count

"""
Resume Service
    - Parses resume files (PDF/DOCX) via rag/ingestion.py
    - Makes LLM CALL #1 (Resume DNA Extraction)
    - Indexes resume chunks into Pinecone for later retrieval

This is a "service", not a LangGraph node, because Resume/Job DNA extraction
happens once up-front (before the adaptive interview loop begins) via the
FastAPI /resume/analyze and /job/analyze endpoints. The LangGraph graph
(agents/interview_graph.py) consumes the *results* of these services as
initial state.
"""
from __future__ import annotations

import logging
import time

from llm.base import LLMService
from models.schemas import ResumeDNA
from prompts.resume_prompts import build_resume_dna_messages
from rag.ingestion import parse_document, ingest_text

logger = logging.getLogger("interviewdna.services.resume")


def parse_resume_file(file_path: str) -> str:
    logger.info("Parsing resume file: %s", file_path)
    start = time.monotonic()
    text = parse_document(file_path)
    logger.info("Parsed resume: %d chars in %.2fs", len(text), time.monotonic() - start)
    logger.debug("Parsed resume text preview (first 300 chars): %r", text[:300])
    if len(text.strip()) < 50:
        logger.warning(
            "Parsed resume text is suspiciously short (%d chars) -- if this is a "
            "scanned/image-based PDF, PyMuPDF can't extract text from it (no OCR). "
            "Try a text-based PDF/DOCX, or check the preview above with LOG_LEVEL=DEBUG.",
            len(text.strip()),
        )
    return text


def extract_resume_dna(llm: LLMService, resume_text: str) -> ResumeDNA:
    """LLM CALL #1 - Resume DNA Extraction.

    Node/Call site: services/resume_service.py -> called from
    api/routes/resume.py before the interview graph starts.
    Why: produces the structured Resume DNA that seeds Alignment (LLM #3),
    Strategy (LLM #4), and every question-generation call thereafter.
    """
    logger.info("LLM CALL #1: Resume DNA extraction starting (%d chars input)", len(resume_text))
    start = time.monotonic()
    dna = llm.invoke_structured(build_resume_dna_messages(resume_text), schema=ResumeDNA)
    logger.info(
        "LLM CALL #1: done in %.1fs -- %d skills, %d projects, %d achievements",
        time.monotonic() - start, len(dna.skills), len(dna.projects), len(dna.achievements),
    )
    return dna


def index_resume(resume_text: str, user_id: str, session_id: str) -> int:
    """Chunk + embed + store resume text in Pinecone (document_type='resume')
    so later retrieval (rag/retriever.py) can surface resume evidence during
    question generation and coaching."""
    logger.info("Indexing resume into Pinecone (session=%s)", session_id)
    start = time.monotonic()
    count = ingest_text(
        text=resume_text,
        user_id=user_id,
        session_id=session_id,
        document_type="resume",
        source="resume",
    )
    logger.info("Indexed %d resume chunk(s) in %.1fs", count, time.monotonic() - start)
    return count


def analyze_resume(llm: LLMService, file_path: str, user_id: str, session_id: str) -> ResumeDNA:
    """End-to-end: parse -> LLM CALL #1 -> index."""
    text = parse_resume_file(file_path)
    dna = extract_resume_dna(llm, text)
    index_resume(text, user_id, session_id)
    return dna

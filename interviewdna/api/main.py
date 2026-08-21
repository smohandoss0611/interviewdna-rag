"""
FastAPI service boundary for InterviewDNA.

    STREAMLIT -> FastAPI (this app) -> LangGraph Interview Orchestrator
                                     -> LLM Service / Mem0 / LlamaIndex+Pinecone

Run locally:
    uvicorn api.main:app --reload --port 8000

Every request is logged with method/path/status/duration. Set LOG_LEVEL=DEBUG
in .env for verbose per-node agent tracing (see logging_config.py).
"""
from __future__ import annotations

import logging
import time

from dotenv import load_dotenv

load_dotenv()

from logging_config import configure_logging

configure_logging()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from api.routes import resume, job, match, interview

logger = logging.getLogger("interviewdna.api")

app = FastAPI(
    title="InterviewDNA API",
    description="Personalized Agentic RAG AI Interview Coach",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local hackathon deployment only
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Logs every request with timing, so a stuck call (e.g. waiting on
    Ollama) is visible as 'still running' rather than silent."""
    start = time.monotonic()
    logger.info("--> %s %s", request.method, request.url.path)
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (time.monotonic() - start) * 1000
        logger.exception(
            "<-- %s %s FAILED after %.0fms", request.method, request.url.path, elapsed_ms
        )
        raise
    elapsed_ms = (time.monotonic() - start) * 1000
    logger.info(
        "<-- %s %s %s (%.0fms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    if elapsed_ms > 10_000:
        logger.warning(
            "%s %s took %.1fs -- likely an LLM call (Ollama cold start, long "
            "prompt, or structured-output retry loop)",
            request.method,
            request.url.path,
            elapsed_ms / 1000,
        )
    return response


app.include_router(resume.router)
app.include_router(job.router)
app.include_router(match.router)
app.include_router(interview.router)


@app.get("/health")
async def health():
    return {"status": "ok"}

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from api.session_store import get_session_store
from llm.factory import get_llm_service
from memory.mem0_service import get_mem0_service
from services.job_service import align_resume_to_job
from agents.strategy_agent import generate_strategy
from models.schemas import MatchRequest, MatchResponse, ResumeDNA, JobDNA

router = APIRouter(prefix="/match", tags=["match"])


@router.post("", response_model=MatchResponse)
async def match(payload: MatchRequest):
    store = get_session_store()
    session = store.get(payload.session_id)
    if session is None:
        raise HTTPException(404, "Unknown session_id")
    if "resume_dna" not in session:
        raise HTTPException(400, "Call /resume/analyze first")
    if "job_dna" not in session:
        raise HTTPException(400, "Call /job/analyze first")

    llm = get_llm_service()
    resume_dna = ResumeDNA.model_validate(session["resume_dna"])
    job_dna = JobDNA.model_validate(session["job_dna"])

    # See resume.py for why run_in_threadpool is required: these are
    # synchronous/blocking calls (LLM + Mem0 + Pinecone), and calling them
    # directly would freeze FastAPI's event loop for unrelated requests too.
    alignment = await run_in_threadpool(align_resume_to_job, llm, job_dna, resume_dna)

    memory_context = []
    try:
        memory_context = await run_in_threadpool(
            get_mem0_service().search_memory, "candidate", "interview coaching history"
        )
    except Exception:
        pass

    strategy = await run_in_threadpool(
        generate_strategy,
        llm,
        resume_dna.model_dump(),
        job_dna.model_dump(),
        alignment.model_dump(),
        "MIXED",
        memory_context,
    )

    store.update(
        payload.session_id,
        alignment=alignment.model_dump(),
        strategy=strategy.model_dump(),
    )
    return MatchResponse(session_id=payload.session_id, alignment=alignment, strategy=strategy)

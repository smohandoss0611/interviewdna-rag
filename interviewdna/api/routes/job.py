from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from api.session_store import get_session_store
from llm.factory import get_llm_service
from services.job_service import extract_job_dna, index_job_description
from models.schemas import JobAnalyzeRequest, JobAnalyzeResponse

router = APIRouter(prefix="/job", tags=["job"])


@router.post("/analyze", response_model=JobAnalyzeResponse)
async def analyze_job(payload: JobAnalyzeRequest):
    store = get_session_store()
    session = store.get(payload.session_id)
    if session is None:
        raise HTTPException(404, "Unknown session_id -- call /resume/analyze first")
    if not payload.job_description_text.strip():
        raise HTTPException(400, "job_description_text is empty")

    llm = get_llm_service()
    # See resume.py for why run_in_threadpool is required here: these calls
    # are synchronous/blocking and would otherwise freeze FastAPI's entire
    # event loop -- including unrelated requests like /health -- for the
    # full duration of the LLM call.
    job_dna = await run_in_threadpool(extract_job_dna, llm, payload.job_description_text)
    await run_in_threadpool(
        index_job_description, payload.job_description_text, "candidate", payload.session_id
    )

    store.update(
        payload.session_id,
        jd_text=payload.job_description_text,
        job_dna=job_dna.model_dump(),
    )
    return JobAnalyzeResponse(session_id=payload.session_id, job_dna=job_dna)

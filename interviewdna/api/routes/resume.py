from __future__ import annotations

import os
import tempfile

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool

from api.session_store import get_session_store
from llm.factory import get_llm_service
from services.resume_service import parse_resume_file, extract_resume_dna, index_resume
from models.schemas import ResumeAnalyzeResponse

router = APIRouter(prefix="/resume", tags=["resume"])


@router.post("/analyze", response_model=ResumeAnalyzeResponse)
async def analyze_resume(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".pdf", ".docx", ".txt")):
        raise HTTPException(400, "Only .pdf, .docx, or .txt resumes are supported")

    store = get_session_store()
    session_id = store.create_session()

    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # run_in_threadpool is essential here: parse_resume_file, the LLM
        # call, and the Pinecone upsert are all synchronous/blocking (they
        # use `requests` under the hood, not an async HTTP client). Calling
        # them directly inside an `async def` route would block FastAPI's
        # entire single-threaded event loop for the whole duration of the
        # LLM call -- meaning the server can't answer ANY other request,
        # including a simple /health check, until this one finishes. That
        # is exactly what "Backend unreachable... Read timed out" while a
        # question/answer is mid-flight looks like from the outside.
        resume_text = await run_in_threadpool(parse_resume_file, tmp_path)
        if not resume_text.strip():
            raise HTTPException(400, "Could not extract any text from the uploaded resume")

        llm = get_llm_service()
        resume_dna = await run_in_threadpool(extract_resume_dna, llm, resume_text)
        await run_in_threadpool(index_resume, resume_text, "candidate", session_id)

        store.update(
            session_id,
            resume_text=resume_text,
            resume_dna=resume_dna.model_dump(),
        )
        return ResumeAnalyzeResponse(session_id=session_id, resume_dna=resume_dna)
    finally:
        os.unlink(tmp_path)

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from api.session_store import get_session_store
from agents.interview_graph import start_interview, submit_answer, get_compiled_graph
from agents import coverage_agent
from models.interview_state import new_state
from models.schemas import (
    InterviewStartRequest, InterviewStartResponse,
    InterviewAnswerRequest, InterviewAnswerResponse,
    InterviewRetryRequest, InterviewRetryResponse,
    CoverageResponse, ResultsResponse, ActivityEvent,
)

router = APIRouter(prefix="/interview", tags=["interview"])

# Tracks how much of each session's activity_log has already been returned
# to the client, so each response only includes NEW agent activity events.
_activity_cursor: dict[str, int] = {}


def _new_activity(session_id: str, activity_log: list[dict]) -> list[ActivityEvent]:
    start = _activity_cursor.get(session_id, 0)
    fresh = activity_log[start:]
    _activity_cursor[session_id] = len(activity_log)
    return [ActivityEvent(label=a["label"], detail=a.get("detail", "")) for a in fresh]


@router.post("/start", response_model=InterviewStartResponse)
async def interview_start(payload: InterviewStartRequest):
    store = get_session_store()
    session = store.get(payload.session_id)
    if session is None:
        raise HTTPException(404, "Unknown session_id")
    for required in ("resume_dna", "job_dna", "alignment", "strategy"):
        if required not in session:
            raise HTTPException(400, f"Missing '{required}' -- run the full pipeline first")

    state = new_state(payload.session_id, user_id=session.get("user_id", "candidate"))
    state.update(
        resume_text=session.get("resume_text", ""),
        jd_text=session.get("jd_text", ""),
        resume_dna=session["resume_dna"],
        job_dna=session["job_dna"],
        alignment=session["alignment"],
        strategy=session["strategy"],
        mode=payload.mode.value,
        question_budget=payload.question_budget,
        questions_asked=0,
        coverage=coverage_agent.init_coverage(session["strategy"]),
    )

    # These are the single longest-running blocking calls in the app --
    # start_interview() can chain a competency selection, a Pinecone
    # retrieval, and an LLM question-generation call before returning. Run
    # in a thread pool so FastAPI's event loop stays free to answer other
    # requests (like the frontend's periodic /health check) while this runs.
    # Without this, the whole server appears "unreachable" for the entire
    # duration of every question/answer turn.
    result = await run_in_threadpool(start_interview, state)
    if result["complete"]:
        raise HTTPException(500, "Interview graph completed with no questions -- check strategy priorities")

    full_state = result["state"]
    return InterviewStartResponse(
        session_id=payload.session_id,
        competency=result["competency"],
        question=result["question"],
        difficulty=full_state.get("current_difficulty", "MEDIUM"),
        activity_log=_new_activity(payload.session_id, full_state.get("activity_log", [])),
    )


@router.post("/answer", response_model=InterviewAnswerResponse)
async def interview_answer(payload: InterviewAnswerRequest):
    _ensure_session(payload.session_id)
    result = await run_in_threadpool(submit_answer, payload.session_id, payload.answer_text)
    full_state = result["state"]
    activity = _new_activity(payload.session_id, full_state.get("activity_log", []))

    if result["complete"]:
        return InterviewAnswerResponse(
            session_id=payload.session_id,
            agent_action="INTERVIEW_COMPLETE",
            interview_complete=True,
            activity_log=activity,
        )

    return InterviewAnswerResponse(
        session_id=payload.session_id,
        evaluation=full_state.get("last_evaluation"),
        star_evaluation=full_state.get("last_star_evaluation"),
        technical_star_evaluation=full_state.get("last_technical_star_evaluation"),
        agent_action=full_state.get("agent_action") or ("COACH" if result.get("is_retry") else ""),
        next_question=result["question"] if not result.get("is_retry") else None,
        next_competency=result.get("competency"),
        coaching=full_state.get("coaching_result") if result.get("is_retry") else None,
        interview_complete=False,
        activity_log=activity,
    )


@router.post("/retry", response_model=InterviewRetryResponse)
async def interview_retry(payload: InterviewRetryRequest):
    _ensure_session(payload.session_id)
    result = await run_in_threadpool(submit_answer, payload.session_id, payload.retry_answer_text)
    full_state = result["state"]
    activity = _new_activity(payload.session_id, full_state.get("activity_log", []))

    return InterviewRetryResponse(
        session_id=payload.session_id,
        before=full_state.get("before_retry_scores", {}),
        retry=full_state.get("retry_scores", {}),
        interview_complete=result["complete"],
        next_question=result.get("question") if not result["complete"] else None,
        next_competency=result.get("competency") if not result["complete"] else None,
        activity_log=activity,
    )


@router.get("/{session_id}/coverage", response_model=CoverageResponse)
async def interview_coverage(session_id: str):
    state = _get_graph_state(session_id)
    return CoverageResponse(session_id=session_id, coverage=state.get("coverage", {}))


@router.get("/{session_id}/results", response_model=ResultsResponse)
async def interview_results(session_id: str):
    state = _get_graph_state(session_id)
    plan = state.get("improvement_plan")
    if plan is None:
        raise HTTPException(400, "Interview not yet complete")
    return ResultsResponse(
        session_id=session_id,
        improvement_plan=plan,
        coverage=state.get("coverage", {}),
        transcript_summary=state.get("transcript", []),
    )


def _ensure_session(session_id: str) -> None:
    if get_session_store().get(session_id) is None:
        raise HTTPException(404, "Unknown session_id")


def _get_graph_state(session_id: str) -> dict:
    app = get_compiled_graph()
    config = {"configurable": {"thread_id": session_id}}
    snapshot = app.get_state(config)
    if not snapshot.values:
        raise HTTPException(404, "No interview state for this session -- call /interview/start first")
    return snapshot.values

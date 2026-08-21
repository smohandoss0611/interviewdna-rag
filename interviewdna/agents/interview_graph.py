"""
InterviewDNA LangGraph Orchestrator.

This is the ONLY module allowed to decide *when* an LLM call happens and
*which* one. Every node below documents which LLM CALL (if any) it triggers.

    LangGraph = decides
    LLM       = understands, reasons, evaluates, generates
    LlamaIndex/Pinecone = retrieves
    Mem0      = remembers

Human-in-the-loop pattern
--------------------------
The candidate's answer is not available synchronously -- it arrives via a
separate FastAPI request. We use LangGraph's `interrupt()` / `Command(resume=...)`
pattern with a MemorySaver checkpointer so the graph can pause at
`await_answer_node`, return control to FastAPI (which shows the question to
Streamlit), and resume exactly where it left off once the candidate answers.

    /interview/start   -> graph.invoke(initial_state, config)      (runs to first interrupt)
    /interview/answer  -> graph.invoke(Command(resume=answer), config)
    /interview/retry   -> graph.invoke(Command(resume=retry_answer), config)
"""
from __future__ import annotations

from typing import Optional, Dict, Any

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

from models.interview_state import InterviewState, log_activity
from llm.factory import get_llm_service
from rag.retriever import retrieve_reference_context, retrieve_resume_evidence

from agents import coverage_agent
from agents import strategy_agent
from agents import technical_agent
from agents import star_agent
from agents import technical_star_agent
from agents import answer_quality_agent
from agents import coaching_agent
from agents import tool_agent
from services import evaluation_service
from memory.mem0_service import get_mem0_service, derive_memory_writes


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _classify_sub_mode(competency: str, job_dna: Dict[str, Any], interview_mode: str) -> str:
    """Deterministic: decide whether this competency turn should be TECHNICAL,
    BEHAVIORAL_STAR, or TECHNICAL_STAR. No LLM call needed."""
    if interview_mode == "TECHNICAL":
        return "TECHNICAL"
    if interview_mode == "BEHAVIORAL_STAR":
        return "BEHAVIORAL_STAR"
    if interview_mode == "TECHNICAL_STAR":
        return "TECHNICAL_STAR"

    # MIXED: use JD classification of the competency name
    behavioral = set(job_dna.get("behavioral_competencies", []))
    technical = set(job_dna.get("technical_competencies", []))
    if competency in behavioral:
        return "BEHAVIORAL_STAR"
    if competency in technical:
        return "TECHNICAL"
    return "TECHNICAL"


def _job_requirement_for(competency: str, job_dna: Dict[str, Any]) -> str:
    for req in job_dna.get("required_skills", []) + job_dna.get("technical_competencies", []):
        if competency.lower() in req.lower() or req.lower() in competency.lower():
            return req
    return competency


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
def select_competency_node(state: InterviewState) -> Dict[str, Any]:
    """Deterministic (Coverage Agent). No LLM call."""
    next_comp = coverage_agent.select_next_competency(
        strategy=state["strategy"],
        coverage=state["coverage"],
        questions_asked=state["questions_asked"],
        question_budget=state["question_budget"],
    )
    if next_comp is None:
        log_activity(state, "Coverage complete or budget exhausted", "Moving to improvement plan")
        return {"current_competency": None}

    sub_mode = _classify_sub_mode(next_comp, state["job_dna"], state["mode"])
    log_activity(state, "Coverage Agent selected competency", f"{next_comp} ({sub_mode})")
    return {
        "current_competency": next_comp,
        "current_sub_mode": sub_mode,
        "current_difficulty": "MEDIUM",
    }


def retrieve_context_node(state: InterviewState) -> Dict[str, Any]:
    """LlamaIndex -> Pinecone retrieval. No LLM call."""
    competency = state["current_competency"]
    sub_mode = state["current_sub_mode"]

    if sub_mode == "TECHNICAL_STAR":
        # Grounds on a real resume claim rather than generic semantic search.
        claim = technical_star_agent.select_resume_claim(state["resume_dna"])
        log_activity(state, "Selected resume claim for deep-dive", claim or "(none found)")
        return {"current_retrieved_context": [], "_resume_claim": claim}

    resume_evidence = retrieve_resume_evidence(
        query=competency, user_id=state["user_id"], session_id=state["session_id"]
    )
    reference_chunks = retrieve_reference_context(query=competency, top_k=3)
    log_activity(
        state,
        "Retrieved context",
        f"{len(reference_chunks)} reference chunk(s), resume evidence: "
        f"{'found' if resume_evidence else 'none'}",
    )
    return {
        "current_retrieved_context": reference_chunks,
        "_resume_evidence": resume_evidence,
    }


def generate_question_node(state: InterviewState) -> Dict[str, Any]:
    """Dispatches to the appropriate LLM CALL #5-family call based on sub_mode."""
    llm = get_llm_service()
    competency = state["current_competency"]
    sub_mode = state["current_sub_mode"]
    difficulty = state.get("current_difficulty", "MEDIUM")

    if sub_mode == "TECHNICAL":
        job_requirement = _job_requirement_for(competency, state["job_dna"])
        context_text = "\n".join(c["text"] for c in state.get("current_retrieved_context", []))
        q = technical_agent.generate_question(
            llm,
            competency=competency,
            job_requirement=job_requirement,
            resume_evidence=state.get("_resume_evidence", ""),
            retrieved_context=context_text,
            difficulty=difficulty,
            previous_question=None,
        )
        log_activity(state, "LLM CALL #5: Question generated", f"[{competency}] {q.question}")
    elif sub_mode == "BEHAVIORAL_STAR":
        q = star_agent.generate_star_question(
            llm, competency=competency, resume_evidence=state.get("_resume_evidence", ""), difficulty=difficulty
        )
        log_activity(state, "LLM CALL #5 (STAR variant): Question generated", f"[{competency}] {q.question}")
    else:  # TECHNICAL_STAR
        claim = state.get("_resume_claim") or "(no specific resume claim found; ask generally)"
        q = technical_star_agent.generate_technical_star_question(
            llm, competency=competency, resume_claim=claim, difficulty=difficulty
        )
        log_activity(state, "LLM CALL #5 (Technical STAR variant): Question generated", f"[{competency}] {q.question}")

    return {
        "current_question": q.question,
        "current_difficulty": q.difficulty,
        "questions_asked": state["questions_asked"] + 1,
        "awaiting_answer": True,
        "awaiting_retry": False,
    }


def await_answer_node(state: InterviewState) -> Dict[str, Any]:
    """Pauses graph execution until the candidate submits an answer via
    POST /interview/answer or /interview/retry."""
    answer = interrupt(
        {
            "type": "retry_answer" if state.get("awaiting_retry") else "answer",
            "question": state["current_question"],
            "competency": state["current_competency"],
        }
    )
    return {"current_answer": answer, "awaiting_answer": False}


def evaluate_answer_node(state: InterviewState) -> Dict[str, Any]:
    """Dispatches to the appropriate LLM CALL #6/#8/#9 based on sub_mode."""
    llm = get_llm_service()
    competency = state["current_competency"]
    sub_mode = state["current_sub_mode"]
    question = state["current_question"]
    answer = state["current_answer"]

    updates: Dict[str, Any] = {}

    if sub_mode == "TECHNICAL":
        result = answer_quality_agent.evaluate_answer(llm, question, answer, competency)
        action = result.recommended_action.value
        log_activity(state, "LLM CALL #6: Answer evaluated", f"action={action} | {result.weakness}")
        updates["last_evaluation"] = result.model_dump()
        updates["agent_action"] = action
        eval_for_transcript = result.model_dump()

    elif sub_mode == "BEHAVIORAL_STAR":
        result = star_agent.evaluate_star_answer(llm, question, answer, competency)
        action = result.recommended_action.value
        log_activity(
            state, "LLM CALL #8: STAR answer evaluated",
            f"weakest={result.weakest_component.value} | action={action}",
        )
        updates["last_star_evaluation"] = result.model_dump()
        updates["agent_action"] = action
        eval_for_transcript = result.model_dump()

    else:  # TECHNICAL_STAR
        claim = state.get("_resume_claim", "")
        result = technical_star_agent.evaluate_technical_star_answer(llm, question, answer, claim)
        action = result.recommended_action
        log_activity(
            state, "LLM CALL #9: Technical STAR evaluated",
            f"weakest={result.weakest_dimension} | action={action}",
        )
        updates["last_technical_star_evaluation"] = result.model_dump()
        updates["agent_action"] = action
        eval_for_transcript = result.model_dump()

    # Explicit Mem0 memory-write rules (Feature: long-term coaching memory)
    mem_writes = derive_memory_writes(competency, eval_for_transcript, updates["agent_action"])
    if mem_writes:
        mem0 = get_mem0_service()
        for m in mem_writes:
            try:
                mem0.add_memory(state["user_id"], m, metadata={"competency": competency})
            except Exception:
                pass  # non-fatal for the hackathon build if Mem0 backend isn't configured
        log_activity(state, "Mem0: stored coaching signal(s)", "; ".join(mem_writes))

    transcript_entry = {
        "competency": competency,
        "mode": sub_mode,
        "question": question,
        "answer": answer,
        "evaluation": eval_for_transcript,
        "agent_action": updates["agent_action"],
        "difficulty": state.get("current_difficulty", "MEDIUM"),
        "is_followup": False,
        "is_retry": False,
    }
    updates["transcript"] = state["transcript"] + [transcript_entry]
    return updates


def route_after_evaluation(state: InterviewState) -> str:
    """Conditional edge: LangGraph DECISION (spec section 5/9/11/12).

    IMPORTANT: question_budget is enforced HERE, not just in
    select_competency_node. Without this check, a competency that keeps
    drawing CLARIFY/PROBE/CHALLENGE from the LLM never routes back through
    select_competency_node (where the budget is normally checked) -- it just
    loops followup -> await_answer -> evaluate -> route_after_evaluation
    indefinitely, so a budget of 3 could silently turn into 6, 10, or more
    follow-ups on a single competency before the LLM ever says MOVE_ON. This
    check is a hard ceiling: once the budget is spent, wrap up the current
    competency regardless of what the LLM recommended.
    """
    if state["questions_asked"] >= state["question_budget"]:
        return "after_move_on"

    action = state["agent_action"]
    if action == "MOVE_ON":
        return "after_move_on"
    if action == "COACH":
        return "coach"
    # everything else (CLARIFY/PROBE/CHALLENGE, PROBE_SITUATION/TASK/ACTION/RESULT,
    # PROBE_<dimension> for technical STAR) is a follow-up
    return "followup"


def followup_node(state: InterviewState) -> Dict[str, Any]:
    """Dispatches to LLM CALL #7 (technical) or a targeted STAR/technical-STAR
    probe question (same 'generate one grounded question' family)."""
    llm = get_llm_service()
    competency = state["current_competency"]
    sub_mode = state["current_sub_mode"]
    action = state["agent_action"]
    difficulty = state.get("current_difficulty", "MEDIUM")

    if sub_mode == "TECHNICAL":
        eval_data = state["last_evaluation"]
        new_difficulty = (
            technical_agent.escalate_difficulty(difficulty)
            if action == "CHALLENGE"
            else difficulty
        )
        q = technical_agent.generate_followup(
            llm,
            action=action,
            competency=competency,
            previous_question=state["current_question"],
            previous_answer=state["current_answer"],
            detected_gap=eval_data.get("detected_gap"),
            difficulty=new_difficulty,
        )
        log_activity(state, "LLM CALL #7: Adaptive follow-up generated", f"[{action}] {q.question}")
        difficulty = new_difficulty
    elif sub_mode == "BEHAVIORAL_STAR":
        # Reuse the STAR question generator, nudged toward the weak component.
        weak = action.replace("PROBE_", "").title()
        hint_competency = f"{competency} (probe deeper on the {weak})"
        q = star_agent.generate_star_question(
            llm, competency=hint_competency, resume_evidence="", difficulty=difficulty
        )
        log_activity(state, "STAR follow-up generated", f"[{action}] {q.question}")
    else:  # TECHNICAL_STAR
        claim = state.get("_resume_claim") or ""
        weak = state.get("last_technical_star_evaluation", {}).get("weakest_dimension", "")
        hint_competency = f"{competency} (probe deeper on {weak})"
        q = technical_star_agent.generate_technical_star_question(
            llm, competency=hint_competency, resume_claim=claim, difficulty=difficulty
        )
        log_activity(state, "Technical STAR follow-up generated", f"[{action}] {q.question}")

    return {
        "current_question": q.question,
        "current_difficulty": difficulty,
        "questions_asked": state["questions_asked"] + 1,
        "awaiting_answer": True,
        "awaiting_retry": False,
    }


def after_move_on_node(state: InterviewState) -> Dict[str, Any]:
    """Deterministic: update coverage, loop back to competency selection."""
    coverage = dict(state["coverage"])
    coverage_agent.update_coverage_after_turn(coverage, state["current_competency"], "MOVE_ON")
    if state["agent_action"] != "MOVE_ON" and state["questions_asked"] >= state["question_budget"]:
        # We got routed here by route_after_evaluation's hard budget cap,
        # not because the LLM actually recommended MOVE_ON -- say so, so
        # it's clear from the activity feed why this competency ended.
        log_activity(
            state, "Question budget reached",
            f"{state['questions_asked']}/{state['question_budget']} -- wrapping up "
            f"'{state['current_competency']}' now rather than continuing follow-ups",
        )
    else:
        log_activity(state, "Coverage Agent updated", f"{state['current_competency']} -> TESTED")
    return {"coverage": coverage}


def coach_node(state: InterviewState) -> Dict[str, Any]:
    """Tool-Use Agent decision, then LLM CALL #10 - RAG-Grounded Coaching (Feature 7)."""
    llm = get_llm_service()
    eval_data = state["last_evaluation"]
    gap = eval_data.get("detected_gap") or eval_data.get("weakness", "")
    competency = state["current_competency"]

    tool_result = tool_agent.select_and_run_tool(llm, gap=gap, competency=competency)
    if tool_result.tool_name == "none":
        log_activity(state, "Tool Agent: no tool needed", "answering from the model's own knowledge")
    else:
        log_activity(
            state, f"Tool Agent selected: {tool_result.tool_name}",
            f"{len(tool_result.chunks)} result(s) found" if tool_result.success
            else f"failed: {tool_result.error}",
        )

    result = coaching_agent.coach_with_retrieved_context(
        llm,
        competency=competency,
        detected_gap=gap,
        original_question=state["current_question"],
        original_answer=state["current_answer"],
        retrieved_chunks=tool_result.chunks,
    )
    log_activity(state, "LLM CALL #10: Grounded coaching generated", result.coaching_text[:120])

    coverage = dict(state["coverage"])
    coverage_agent.update_coverage_after_turn(coverage, state["current_competency"], "PARTIAL")

    return {
        "coaching_result": result.model_dump(),
        "before_retry_scores": eval_data.get("scores", eval_data),
        "current_question": result.retry_question,
        "awaiting_answer": True,
        "awaiting_retry": True,
        "coverage": coverage,
    }


def retry_evaluate_node(state: InterviewState) -> Dict[str, Any]:
    """LLM CALL #11 - Retry Evaluation, then deterministic comparison."""
    llm = get_llm_service()
    competency = state["current_competency"]
    retry_answer = state["current_answer"]

    retry_eval = coaching_agent.evaluate_retry(
        llm, retry_question=state["current_question"], retry_answer=retry_answer, competency=competency
    )
    comparison = coaching_agent.compare_before_retry(state["before_retry_scores"], retry_eval)
    log_activity(
        state, "LLM CALL #11: Retry evaluated",
        f"before={comparison['before']} retry={comparison['retry']}",
    )

    coverage = dict(state["coverage"])
    coverage_agent.update_coverage_after_turn(coverage, competency, "MOVE_ON")

    transcript_entry = {
        "competency": competency,
        "mode": state["current_sub_mode"],
        "question": state["current_question"],
        "answer": retry_answer,
        "evaluation": retry_eval.model_dump(),
        "agent_action": "MOVE_ON",
        "difficulty": state.get("current_difficulty", "MEDIUM"),
        "is_followup": False,
        "is_retry": True,
    }

    return {
        "retry_scores": comparison["retry"],
        "coverage": coverage,
        "awaiting_retry": False,
        "transcript": state["transcript"] + [transcript_entry],
    }


def improvement_plan_node(state: InterviewState) -> Dict[str, Any]:
    """LLM CALL #12 - Personalized Improvement Plan (Feature 10)."""
    llm = get_llm_service()

    memory_context = state.get("memory_context", [])
    try:
        mem0 = get_mem0_service()
        memory_context = mem0.search_memory(state["user_id"], query="interview coaching history")
    except Exception:
        pass

    plan = evaluation_service.generate_improvement_plan(
        llm,
        alignment=state["alignment"],
        transcript=state["transcript"],
        coverage=state["coverage"],
        before_retry_scores=state.get("before_retry_scores"),
        retry_scores=state.get("retry_scores"),
        memory_context=memory_context,
    )
    log_activity(state, "LLM CALL #12: Improvement plan generated", "; ".join(plan.strengths))
    return {"improvement_plan": plan.model_dump(), "interview_complete": True}


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #
def build_interview_graph():
    graph = StateGraph(InterviewState)

    graph.add_node("select_competency", select_competency_node)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("generate_question", generate_question_node)
    graph.add_node("await_answer", await_answer_node)
    graph.add_node("evaluate_answer", evaluate_answer_node)
    graph.add_node("followup", followup_node)
    graph.add_node("after_move_on", after_move_on_node)
    graph.add_node("coach", coach_node)
    graph.add_node("retry_evaluate", retry_evaluate_node)
    graph.add_node("improvement_plan", improvement_plan_node)

    graph.set_entry_point("select_competency")

    graph.add_conditional_edges(
        "select_competency",
        lambda s: "done" if s.get("current_competency") is None else "continue",
        {"done": "improvement_plan", "continue": "retrieve_context"},
    )
    graph.add_edge("retrieve_context", "generate_question")
    graph.add_edge("generate_question", "await_answer")

    graph.add_conditional_edges(
        "await_answer",
        lambda s: "retry" if s.get("awaiting_retry") else "normal",
        {"retry": "retry_evaluate", "normal": "evaluate_answer"},
    )

    graph.add_conditional_edges(
        "evaluate_answer",
        route_after_evaluation,
        {"followup": "followup", "coach": "coach", "after_move_on": "after_move_on"},
    )

    graph.add_edge("followup", "await_answer")
    graph.add_edge("coach", "await_answer")
    graph.add_edge("retry_evaluate", "select_competency")
    graph.add_edge("after_move_on", "select_competency")
    graph.add_edge("improvement_plan", END)

    return graph


_checkpointer = MemorySaver()
_compiled_graph = None


def get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_interview_graph().compile(checkpointer=_checkpointer)
    return _compiled_graph


def start_interview(state: InterviewState) -> Dict[str, Any]:
    """Runs the graph from the beginning up to the first interrupt (a question)."""
    app = get_compiled_graph()
    config = {"configurable": {"thread_id": state["session_id"]}}
    result = app.invoke(state, config=config)
    return _extract_response(app, config, result)


def submit_answer(session_id: str, answer_text: str) -> Dict[str, Any]:
    """Resumes the graph with the candidate's answer."""
    app = get_compiled_graph()
    config = {"configurable": {"thread_id": session_id}}
    result = app.invoke(Command(resume=answer_text), config=config)
    return _extract_response(app, config, result)


def _extract_response(app, config, result: Dict[str, Any]) -> Dict[str, Any]:
    """Normalizes graph output: either paused at an interrupt (next question)
    or finished (interview complete + improvement plan)."""
    snapshot = app.get_state(config)
    if snapshot.next:
        # Graph is paused at await_answer -- pull the interrupt payload.
        interrupts = snapshot.tasks[0].interrupts if snapshot.tasks else []
        payload = interrupts[0].value if interrupts else {}
        full_state = snapshot.values
        return {
            "complete": False,
            "question": payload.get("question", full_state.get("current_question")),
            "competency": payload.get("competency", full_state.get("current_competency")),
            "is_retry": payload.get("type") == "retry_answer",
            "state": full_state,
        }
    return {"complete": True, "state": result}

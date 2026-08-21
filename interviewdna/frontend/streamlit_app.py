"""
InterviewDNA Streamlit frontend.

    STREAMLIT (this file)
          |
          v
       FastAPI  (http://localhost:8000, or your deployed backend)

Run locally:
    streamlit run frontend/streamlit_app.py

On Streamlit Community Cloud, set INTERVIEWDNA_API_BASE via the app's
"Secrets" panel (as INTERVIEWDNA_API_BASE = "https://your-backend-host:8000")
rather than an OS env var -- see _resolve_api_base() below.
"""
from __future__ import annotations

import os
import sys
import requests
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logging_config import configure_logging

configure_logging()


def _resolve_api_base() -> str:
    """Streamlit Cloud config lives in st.secrets, not os.environ. Check
    secrets first, fall back to an env var (local/Docker), then localhost
    (local dev default)."""
    try:
        if "INTERVIEWDNA_API_BASE" in st.secrets:
            return st.secrets["INTERVIEWDNA_API_BASE"]
    except Exception:
        pass  # no secrets.toml present -- fine for local dev
    return os.getenv("INTERVIEWDNA_API_BASE", "http://localhost:8000")


API_BASE = _resolve_api_base()

# LLM calls (Ollama, especially on CPU, or a cold first-request model load)
# can genuinely take 60-90+ seconds, and invoke_structured() retries up to
# 3x on malformed JSON -- so this needs to be generous, not the default
# "just hang forever" behavior of requests with no timeout at all.
BACKEND_TIMEOUT = 180


def backend_post(path: str, **kwargs):
    """POST to the backend with a real timeout and a visible, specific error
    instead of an infinite spinner."""
    try:
        resp = requests.post(f"{API_BASE}{path}", timeout=BACKEND_TIMEOUT, **kwargs)
        resp.raise_for_status()
        return resp
    except requests.exceptions.Timeout:
        st.error(
            f"Backend didn't respond within {BACKEND_TIMEOUT}s calling `{path}`. "
            "This usually means Ollama is stuck or still loading the model -- "
            "check the terminal running `uvicorn` and the terminal running "
            "`ollama serve` for what's actually happening."
        )
        st.stop()
    except requests.exceptions.ConnectionError as e:
        st.error(f"Could not reach the backend at `{API_BASE}{path}`. Is FastAPI running? ({e})")
        st.stop()
    except requests.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = e.response.text[:500]
        except Exception:
            pass
        st.error(f"Backend returned an error for `{path}`: {e}\n\n{detail}")
        st.stop()


def backend_get(path: str, **kwargs):
    try:
        resp = requests.get(f"{API_BASE}{path}", timeout=BACKEND_TIMEOUT, **kwargs)
        resp.raise_for_status()
        return resp
    except requests.exceptions.Timeout:
        st.error(f"Backend didn't respond within {BACKEND_TIMEOUT}s calling `{path}`.")
        st.stop()
    except requests.exceptions.ConnectionError as e:
        st.error(f"Could not reach the backend at `{API_BASE}{path}`. Is FastAPI running? ({e})")
        st.stop()
    except requests.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = e.response.text[:500]
        except Exception:
            pass
        st.error(f"Backend returned an error for `{path}`: {e}\n\n{detail}")
        st.stop()

st.set_page_config(page_title="InterviewDNA", page_icon="🧬", layout="wide")

with st.sidebar:
    st.caption(f"Backend: `{API_BASE}`")
    try:
        health = requests.get(f"{API_BASE}/health", timeout=4)
        if health.ok:
            st.success("Backend reachable")
        else:
            st.error(f"Backend returned {health.status_code}")
    except requests.RequestException as e:
        st.error("Backend unreachable")
        st.caption(str(e))


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def _init_state():
    defaults = {
        "session_id": None,
        "resume_dna": None,
        "job_dna": None,
        "alignment": None,
        "strategy": None,
        "interview_started": False,
        "current_question": None,
        "current_competency": None,
        "current_difficulty": None,
        "awaiting_retry": False,
        "last_answer_response": None,
        "activity_feed": [],
        "interview_complete": False,
        "results": None,
        # Purely a visible progress indicator -- makes it unmistakable
        # whether a new question actually loaded after Submit, rather than
        # relying on eyeballing whether the question text changed.
        "turn_number": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


def push_activity(events: list[dict]):
    st.session_state.activity_feed.extend(events)


def render_activity_feed():
    with st.expander("🤖 Agent Activity", expanded=True):
        if not st.session_state.activity_feed:
            st.caption("No agent activity yet.")
        for evt in st.session_state.activity_feed[-25:]:
            st.markdown(f"✓ **{evt['label']}** — {evt.get('detail', '')}")


# --------------------------------------------------------------------------- #
# Sidebar / nav
# --------------------------------------------------------------------------- #
st.title("🧬 InterviewDNA")
st.caption("A personalized, agentic RAG interview coach")

tab_resume, tab_interview, tab_results = st.tabs(["📄 Resume & Job", "🎤 Interview", "📊 Results"])


# --------------------------------------------------------------------------- #
# TAB 1: Resume & Job
# --------------------------------------------------------------------------- #
with tab_resume:
    st.subheader("Upload your resume and target job description")

    col1, col2 = st.columns(2)
    with col1:
        resume_file = st.file_uploader("Resume (PDF or DOCX)", type=["pdf", "docx", "txt"])
    with col2:
        jd_text = st.text_area("Target Job Description", height=220)

    if st.button("Analyze Resume & Job", type="primary", disabled=not (resume_file and jd_text)):
        with st.spinner("Extracting Resume DNA..."):
            resp = backend_post(
                "/resume/analyze",
                files={"file": (resume_file.name, resume_file.getvalue())},
            )
            data = resp.json()
            st.session_state.session_id = data["session_id"]
            st.session_state.resume_dna = data["resume_dna"]

        with st.spinner("Extracting Job DNA..."):
            resp = backend_post(
                "/job/analyze",
                json={"session_id": st.session_state.session_id, "job_description_text": jd_text},
            )
            st.session_state.job_dna = resp.json()["job_dna"]

        with st.spinner("Computing alignment & interview strategy..."):
            resp = backend_post("/match", json={"session_id": st.session_state.session_id})
            match_data = resp.json()
            st.session_state.alignment = match_data["alignment"]
            st.session_state.strategy = match_data["strategy"]

        st.success("Analysis complete — head to the Interview tab.")

    if st.session_state.resume_dna:
        st.divider()
        st.subheader("Resume DNA")
        rd = st.session_state.resume_dna
        c1, c2, c3 = st.columns(3)
        c1.metric("Skills", len(rd.get("skills", [])))
        c2.metric("Projects", len(rd.get("projects", [])))
        c3.metric("Achievements", len(rd.get("achievements", [])))
        with st.expander("Full Resume DNA"):
            st.json(rd)

    if st.session_state.job_dna:
        st.subheader("Job DNA")
        with st.expander("Full Job DNA"):
            st.json(st.session_state.job_dna)

    if st.session_state.alignment:
        st.subheader("Match / Alignment")
        for item in st.session_state.alignment["items"]:
            level = item["evidence_level"]
            color = {"STRONG_EVIDENCE": "🟢", "PARTIAL_EVIDENCE": "🟡", "NOT_DEMONSTRATED": "⚪"}.get(level, "⚪")
            st.markdown(f"{color} **{item['requirement']}** — {level}")
            if item.get("rationale"):
                st.caption(item["rationale"])

    if st.session_state.strategy:
        st.subheader("Interview Priorities")
        strat = st.session_state.strategy
        st.markdown(f"Technical **{strat['technical']}%** / Behavioral **{strat['behavioral']}%**")
        for pc in strat["priority_competencies"]:
            st.markdown(f"- **{pc['name']}** ({pc['priority']}) — {pc['reason']}")


# --------------------------------------------------------------------------- #
# TAB 2: Interview
# --------------------------------------------------------------------------- #
with tab_interview:
    if not st.session_state.session_id:
        st.info("Complete the Resume & Job tab first.")
    else:
        if not st.session_state.interview_started:
            mode = st.selectbox(
                "Interview mode", ["MIXED", "TECHNICAL", "BEHAVIORAL_STAR", "TECHNICAL_STAR"]
            )
            budget = st.slider("Question budget", min_value=3, max_value=15, value=8)
            if st.button("Start Interview", type="primary"):
                resp = backend_post(
                    "/interview/start",
                    json={"session_id": st.session_state.session_id, "mode": mode, "question_budget": budget},
                )
                data = resp.json()
                st.session_state.interview_started = True
                st.session_state.current_question = data["question"]
                st.session_state.current_competency = data["competency"]
                st.session_state.current_difficulty = data["difficulty"]
                st.session_state.turn_number = 1
                push_activity(data["activity_log"])
                st.rerun()
        else:
            render_activity_feed()
            st.divider()

            if st.session_state.interview_complete:
                st.success("Interview complete! See the Results tab.")
            else:
                st.markdown(f"**Competency:** {st.session_state.current_competency}  \n"
                            f"**Difficulty:** {st.session_state.current_difficulty}")
                label = "Retry" if st.session_state.awaiting_retry else f"Question #{st.session_state.turn_number}"
                st.subheader(label)
                st.info(st.session_state.current_question)

                answer_key = "retry_answer" if st.session_state.awaiting_retry else "answer"
                answer = st.text_area("Your answer", key=answer_key, height=150)

                if st.button("Submit", type="primary", disabled=not answer.strip()):
                    endpoint = "/interview/retry" if st.session_state.awaiting_retry else "/interview/answer"
                    body_key = "retry_answer_text" if st.session_state.awaiting_retry else "answer_text"
                    resp = backend_post(
                        endpoint,
                        json={"session_id": st.session_state.session_id, body_key: answer},
                    )
                    data = resp.json()
                    push_activity(data["activity_log"])

                    # Clear whichever text_area was just submitted -- otherwise
                    # Streamlit's widget state keeps the typed text around
                    # across the rerun, so the NEXT question shows up with the
                    # PREVIOUS answer still sitting in the box, which looks
                    # exactly like "nothing happened" even when the question
                    # itself did update.
                    # Clear BOTH textbox keys, not just the one just used --
                    # a COACH cycle switches between the "answer" and
                    # "retry_answer" widgets, and whichever one wasn't just
                    # submitted could still be holding stale text from an
                    # EARLIER turn in this same session (e.g. a previous
                    # retry cycle), which then reappears when that widget
                    # comes back into use.
                    st.session_state.pop("answer", None)
                    st.session_state.pop("retry_answer", None)

                    if st.session_state.awaiting_retry:
                        st.session_state.awaiting_retry = False
                        st.session_state.last_answer_response = {"retry_comparison": data}
                        if data.get("interview_complete"):
                            st.session_state.interview_complete = True
                            st.session_state.current_question = None
                        else:
                            # The graph moved on to a new competency after the
                            # retry -- show that next question now instead of
                            # leaving the UI with nothing to display.
                            st.session_state.current_question = data.get("next_question")
                            st.session_state.current_competency = (
                                data.get("next_competency") or st.session_state.current_competency
                            )
                            st.session_state.turn_number += 1
                        st.rerun()
                    else:
                        if data.get("interview_complete"):
                            st.session_state.interview_complete = True
                            st.rerun()
                        elif data.get("coaching"):
                            # COACH path: show coaching, then retry question --
                            # not a new turn number, this is still resolving
                            # the current competency.
                            st.session_state.last_answer_response = data
                            st.session_state.awaiting_retry = True
                            st.session_state.current_question = data["coaching"]["retry_question"]
                            st.rerun()
                        else:
                            st.session_state.last_answer_response = data
                            st.session_state.current_question = data.get("next_question")
                            st.session_state.current_competency = data.get("next_competency") or st.session_state.current_competency
                            st.session_state.turn_number += 1
                            st.rerun()

                # Show last turn's evaluation / decision (Agent Activity detail)
                last = st.session_state.last_answer_response
                if last and not st.session_state.awaiting_retry:
                    st.divider()
                    if last.get("evaluation"):
                        st.subheader("Answer Quality")
                        st.json(last["evaluation"])
                    if last.get("star_evaluation"):
                        st.subheader("STAR Evaluation")
                        se = last["star_evaluation"]
                        st.table(
                            {
                                "Component": ["Situation", "Task", "Action", "Result"],
                                "Score": [se["situation"], se["task"], se["action"], se["result"]],
                            }
                        )
                    if last.get("technical_star_evaluation"):
                        st.subheader("Technical STAR Evaluation")
                        tse = last["technical_star_evaluation"]
                        st.markdown("**STAR**")
                        st.table(tse["star"])
                        st.markdown("**Technical Dimensions**")
                        st.table(tse["technical"])
                    if last.get("agent_action"):
                        st.markdown(f"**Agent Decision:** `{last['agent_action']}`")

                if last and last.get("retry_comparison"):
                    rc = last["retry_comparison"]
                    st.subheader("Before / Retry Comparison")
                    st.table(
                        {
                            "Dimension": list(rc["before"].keys()),
                            "Before": list(rc["before"].values()),
                            "Retry": [rc["retry"].get(k, "-") for k in rc["before"].keys()],
                        }
                    )


# --------------------------------------------------------------------------- #
# TAB 3: Results
# --------------------------------------------------------------------------- #
with tab_results:
    if not st.session_state.session_id:
        st.info("Complete the interview first.")
    elif not st.session_state.interview_complete:
        st.info("Finish the interview to see your personalized improvement plan.")
    else:
        if st.button("Load Results"):
            resp = backend_get(f"/interview/{st.session_state.session_id}/results")
            st.session_state.results = resp.json()

        if st.session_state.results:
            results = st.session_state.results
            plan = results["improvement_plan"]

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("💪 Strengths")
                for s in plan["strengths"]:
                    st.markdown(f"- {s}")
            with col2:
                st.subheader("🎯 Development Areas")
                for d in plan["development_areas"]:
                    st.markdown(f"- {d}")

            st.subheader("📚 Next Practice")
            for n in plan["next_practice"]:
                st.markdown(f"- {n}")

            st.subheader("Coverage")
            coverage = results["coverage"]
            st.table({"Competency": list(coverage.keys()), "Status": list(coverage.values())})

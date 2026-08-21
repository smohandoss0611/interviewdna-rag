#!/usr/bin/env python3
"""
End-to-end smoke test for InterviewDNA -- drives the ENTIRE pipeline
(resume -> job -> match -> interview -> results) via the API directly,
skipping the Streamlit UI, so you can reach the Results tab in one command
instead of manually clicking through N slow LLM-backed questions.

Usage:
    python scripts/smoke_test.py path/to/resume.pdf
    python scripts/smoke_test.py path/to/resume.pdf --budget 3 --mode TECHNICAL
    python scripts/smoke_test.py path/to/resume.pdf --jd path/to/jd.txt

If --jd is omitted, a short built-in sample job description is used.
Answers to interview questions are canned placeholder text -- this script is
for verifying the PIPELINE works end-to-end (every LLM call fires, the graph
reaches improvement_plan, /results returns data), not for judging answer
quality. Use the real Streamlit UI for that.

Requires the backend running first:
    uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

import argparse
import sys
import time

import requests

DEFAULT_JD = """Data Engineer (Founding Team)
We're looking for a Data Engineer to own our data infrastructure end-to-end,
from ingestion pipelines to the metrics our exec team relies on daily.

Requirements:
- 3+ years of experience in data engineering or a closely related role.
- Strong SQL and Python skills.
- Hands-on experience with a modern ELT stack (e.g. Airflow, dbt, Fivetran).
- Experience with a cloud data warehouse (Snowflake, BigQuery, or Redshift).
- Experience with AWS or another major cloud provider.
- Ability to work in a founding environment: high ownership, comfort with
  ambiguity, collaboration, and end-to-end product thinking.

Preferred:
- Experience with streaming data systems (Kafka, Kinesis, or similar).
- Experience deploying and operating services on Kubernetes.
- Experience at an early-stage startup, or building zero-to-one data
  infrastructure.
"""

CANNED_ANSWER = (
    "In my last role I owned this end-to-end. I designed the pipeline using "
    "Airflow and dbt, made a deliberate trade-off to favor incremental "
    "processing over full-table scans to cut runtime, and validated the "
    "approach with data quality tests before rolling it out. It reduced "
    "pipeline runtime by roughly 40% and cut our time-to-detect failures "
    "significantly. If I were to do it again I'd invest earlier in "
    "observability around the DAG so failures surface faster."
)


def step(label: str):
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("resume_path", help="Path to a resume file (.pdf, .docx, or .txt)")
    parser.add_argument("--jd", help="Path to a job description .txt file (default: built-in sample)")
    parser.add_argument("--api-base", default="http://localhost:8000")
    parser.add_argument("--mode", default="TECHNICAL", choices=["TECHNICAL", "BEHAVIORAL_STAR", "TECHNICAL_STAR", "MIXED"])
    parser.add_argument("--budget", type=int, default=3, help="Question budget -- keep this LOW for fast testing")
    parser.add_argument("--max-turns", type=int, default=30, help="Safety cap on total answer/retry submissions")
    args = parser.parse_args()

    api = args.api_base.rstrip("/")
    jd_text = open(args.jd).read() if args.jd else DEFAULT_JD

    # -- health check first, fail fast with a clear message ------------
    try:
        requests.get(f"{api}/health", timeout=5).raise_for_status()
    except requests.RequestException as e:
        print(f"Backend not reachable at {api} -- is uvicorn running? ({e})")
        sys.exit(1)

    overall_start = time.monotonic()

    step("1/6  POST /resume/analyze")
    t0 = time.monotonic()
    with open(args.resume_path, "rb") as f:
        resp = requests.post(f"{api}/resume/analyze", files={"file": (args.resume_path, f)}, timeout=180)
    resp.raise_for_status()
    data = resp.json()
    session_id = data["session_id"]
    resume_dna = data["resume_dna"]
    print(f"session_id = {session_id}")
    print(f"skills extracted: {len(resume_dna.get('skills', []))}, "
          f"projects: {len(resume_dna.get('projects', []))}  ({time.monotonic() - t0:.1f}s)")
    if not resume_dna.get("skills") and not resume_dna.get("projects"):
        print("WARNING: Resume DNA came back empty -- extraction likely failed. "
              "Check backend logs / model quality before continuing.")

    step("2/6  POST /job/analyze")
    t0 = time.monotonic()
    resp = requests.post(f"{api}/job/analyze",
                          json={"session_id": session_id, "job_description_text": jd_text}, timeout=180)
    resp.raise_for_status()
    job_dna = resp.json()["job_dna"]
    print(f"required skills: {len(job_dna.get('required_skills', []))}  ({time.monotonic() - t0:.1f}s)")

    step("3/6  POST /match")
    t0 = time.monotonic()
    resp = requests.post(f"{api}/match", json={"session_id": session_id}, timeout=180)
    resp.raise_for_status()
    match_data = resp.json()
    strategy = match_data["strategy"]
    print(f"priority competencies: {[c['name'] for c in strategy['priority_competencies']]}  "
          f"({time.monotonic() - t0:.1f}s)")

    step(f"4/6  POST /interview/start  (mode={args.mode}, budget={args.budget})")
    t0 = time.monotonic()
    resp = requests.post(f"{api}/interview/start",
                          json={"session_id": session_id, "mode": args.mode, "question_budget": args.budget},
                          timeout=180)
    resp.raise_for_status()
    turn = resp.json()
    print(f"Q1 [{turn['competency']}]: {turn['question'][:100]}...  ({time.monotonic() - t0:.1f}s)")

    step("5/6  Answering questions until interview_complete...")
    complete = False
    awaiting_retry = False
    for i in range(args.max_turns):
        t0 = time.monotonic()
        if awaiting_retry:
            resp = requests.post(f"{api}/interview/retry",
                                  json={"session_id": session_id, "retry_answer_text": CANNED_ANSWER}, timeout=180)
            resp.raise_for_status()
            r = resp.json()
            print(f"  [retry {i+1}] before={r['before']} retry={r['retry']}  ({time.monotonic() - t0:.1f}s)")
            awaiting_retry = False
            if r.get("interview_complete"):
                print(f"  [retry {i+1}] interview_complete=True (right after this retry)")
                complete = True
                break
            continue

        resp = requests.post(f"{api}/interview/answer",
                              json={"session_id": session_id, "answer_text": CANNED_ANSWER}, timeout=180)
        resp.raise_for_status()
        r = resp.json()
        elapsed = time.monotonic() - t0

        if r.get("interview_complete"):
            print(f"  [turn {i+1}] interview_complete=True  ({elapsed:.1f}s)")
            complete = True
            break
        if r.get("coaching"):
            print(f"  [turn {i+1}] action=COACH -> retry queued: "
                  f"{r['coaching']['retry_question'][:80]}...  ({elapsed:.1f}s)")
            awaiting_retry = True
            continue

        action = r.get("agent_action", "?")
        next_q = (r.get("next_question") or "")[:80]
        print(f"  [turn {i+1}] action={action} -> next: {next_q}...  ({elapsed:.1f}s)")
    else:
        print(f"\nHit --max-turns ({args.max_turns}) without completing -- either raise "
              f"--budget's effective ceiling isn't being reached, or there's a loop bug. "
              f"Check backend logs.")
        sys.exit(1)

    if not complete:
        print("\nLoop ended without interview_complete=True -- something's off. Check backend logs.")
        sys.exit(1)

    step("6/6  GET /interview/{session_id}/results")
    t0 = time.monotonic()
    resp = requests.get(f"{api}/interview/{session_id}/results", timeout=60)
    resp.raise_for_status()
    results = resp.json()
    plan = results["improvement_plan"]
    print(f"({time.monotonic() - t0:.1f}s)\n")
    print("Strengths:")
    for s in plan["strengths"]:
        print(f"  - {s}")
    print("Development areas:")
    for d in plan["development_areas"]:
        print(f"  - {d}")
    print("Next practice:")
    for n in plan["next_practice"]:
        print(f"  - {n}")
    print(f"\nCoverage: {results['coverage']}")

    print(f"\n{'=' * 70}")
    print(f"ALL STEPS PASSED in {time.monotonic() - overall_start:.1f}s total")
    print(f"session_id = {session_id}  (paste this into Streamlit's Results tab flow if you want the UI too)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

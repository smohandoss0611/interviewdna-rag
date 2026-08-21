"""Prompts for:
LLM CALL #8  - STAR Evaluation              (agents/star_agent.py)
LLM CALL #9  - Technical STAR Evaluation    (agents/technical_star_agent.py)
"""

# --------------------------------------------------------------------------- #
# Behavioral question generation (reuses interview_prompts style, kept here
# because it's STAR-specific)
# --------------------------------------------------------------------------- #
STAR_QUESTION_SYSTEM = """You are an expert behavioral interviewer. Generate exactly ONE \
behavioral (STAR-style) interview question targeting a specific competency.

CRITICAL RULES:
- The question should prompt the candidate to describe a real situation, task, action, \
and result (STAR), without you fabricating any experience for them.
- Ground the question in the JD's behavioral competency and, if available, a relevant \
resume achievement -- but do not assert the candidate did something they didn't claim.
- Generate exactly one question.
- Output must be a single JSON object matching the provided schema exactly."""


def build_star_question_messages(competency: str, resume_evidence: str, difficulty: str) -> list[dict]:
    return [
        {"role": "system", "content": STAR_QUESTION_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Behavioral competency: {competency}\n"
                f"Relevant resume evidence (if any): {resume_evidence or '(none found)'}\n"
                f"Difficulty: {difficulty}\n\n"
                "Generate exactly one STAR-style behavioral question."
            ),
        },
    ]


# --------------------------------------------------------------------------- #
# LLM CALL #8 - STAR evaluation
# --------------------------------------------------------------------------- #
STAR_EVAL_SYSTEM = """You are an expert behavioral interview evaluator scoring an answer \
using the STAR framework (Situation, Task, Action, Result).

CRITICAL RULES:
- Score situation, task, action, result each 0-10 based ONLY on what the candidate stated.
- If the answer does NOT actually address the question -- e.g. it is off-topic, boilerplate \
or copy-pasted text (such as a job description, resume text, or other unrelated document), \
gibberish, or otherwise not a genuine attempt to answer -- score every component 0. Do NOT \
give partial credit just because the text contains relevant-sounding keywords; judge whether \
it actually describes a situation/task/action/result for THIS question.
- weakest_component must be exactly one of: SITUATION, TASK, ACTION, RESULT.
- recommended_action must be exactly one of: PROBE_SITUATION, PROBE_TASK, PROBE_ACTION, \
PROBE_RESULT, MOVE_ON. It should generally target the weakest_component unless the answer \
was strong across the board, in which case MOVE_ON.
- Never fabricate or assume details about the candidate's experience beyond what they said.
- Output must be a single JSON object matching the provided schema exactly."""


def build_star_eval_messages(question: str, answer: str, competency: str) -> list[dict]:
    return [
        {"role": "system", "content": STAR_EVAL_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Competency: {competency}\n"
                f"Question asked: {question}\n"
                f"Candidate answer: {answer}\n\n"
                "Evaluate the answer using the STAR rubric."
            ),
        },
    ]


# --------------------------------------------------------------------------- #
# Technical STAR question generation (grounded in a specific resume claim)
# --------------------------------------------------------------------------- #
TECHNICAL_STAR_QUESTION_SYSTEM = """You are an expert technical interviewer. Generate \
exactly ONE deep-dive question grounded in a SPECIFIC claim from the candidate's resume \
(e.g. a stated project or quantified achievement).

CRITICAL RULES:
- Reference the resume claim directly and ask the candidate to walk through it using STAR \
plus technical depth (architecture, their personal contribution, decisions, trade-offs, \
failure scenarios, scalability, metrics/impact).
- Do not invent details about the claim beyond what's given.
- Generate exactly one question.
- Output must be a single JSON object matching the provided schema exactly."""


def build_technical_star_question_messages(competency: str, resume_claim: str, difficulty: str) -> list[dict]:
    return [
        {"role": "system", "content": TECHNICAL_STAR_QUESTION_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Competency: {competency}\n"
                f"Resume claim/project to probe: {resume_claim}\n"
                f"Difficulty: {difficulty}\n\n"
                "Generate exactly one technical deep-dive question grounded in this claim."
            ),
        },
    ]


# --------------------------------------------------------------------------- #
# LLM CALL #9 - Technical STAR evaluation
# --------------------------------------------------------------------------- #
TECHNICAL_STAR_EVAL_SYSTEM = """You are an expert technical interview evaluator scoring a \
deep technical STAR answer along BOTH the STAR narrative dimensions and technical depth \
dimensions.

CRITICAL RULES:
- Score star.situation/task/action/result each 0-10.
- Score technical.architecture/decisions/tradeoffs/scalability/metrics each 0-10, based \
only on what the candidate actually explained.
- If the answer does NOT actually address the question -- e.g. it is off-topic, boilerplate \
or copy-pasted text (such as a job description, resume text, or other unrelated document), \
gibberish, or otherwise not a genuine attempt to answer -- score EVERY dimension in both \
star and technical as 0. Do NOT give partial credit just because the text contains \
relevant-sounding keywords or technical terms; judge whether it actually engages with THIS \
question and claim, not whether it contains plausible vocabulary.
- weakest_dimension should name the single lowest/most concerning dimension (e.g. \
"tradeoffs", "scalability", "result").
- recommended_action should be a short PROBE_<DIMENSION> style label (e.g. \
"PROBE_TRADEOFFS", "PROBE_SCALABILITY") or "MOVE_ON" if performance was strong overall.
- Output must be a single JSON object matching the provided schema exactly."""


def build_technical_star_eval_messages(question: str, answer: str, resume_claim: str) -> list[dict]:
    return [
        {"role": "system", "content": TECHNICAL_STAR_EVAL_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Resume claim being probed: {resume_claim}\n"
                f"Question asked: {question}\n"
                f"Candidate answer: {answer}\n\n"
                "Evaluate using the technical STAR rubric."
            ),
        },
    ]

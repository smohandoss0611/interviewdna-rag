"""Prompts for:
LLM CALL #4  - Interview Strategy Generation      (agents/strategy_agent.py)
LLM CALL #5  - Interview Question Generation       (agents/technical_agent.py)
LLM CALL #6  - Candidate Answer Evaluation         (agents/answer_quality_agent.py)
LLM CALL #7  - Adaptive Follow-Up Generation       (agents/technical_agent.py)
"""

# --------------------------------------------------------------------------- #
# LLM CALL #4 - Strategy
# --------------------------------------------------------------------------- #
STRATEGY_SYSTEM = """You are an interview strategy planner. You DO NOT run the \
interview yourself -- a separate orchestrator (LangGraph) controls the actual flow. \
Your job is only to PROPOSE priorities the orchestrator can use.

CRITICAL RULES:
- Base priorities on JD requirements, resume alignment gaps, and any prior coaching memory.
- Weight technical/behavioral split (must sum to 100) to fit the requested interview mode.
- List priority_competencies ordered by importance, each with a HIGH/MEDIUM/LOW priority \
and a one-sentence reason grounded in the alignment data.
- Output must be a single JSON object matching the provided schema exactly."""


def build_strategy_messages(
    resume_dna_json: str,
    job_dna_json: str,
    alignment_json: str,
    mode: str,
    memory_context: list[str],
) -> list[dict]:
    memory_block = "\n".join(f"- {m}" for m in memory_context) or "(none)"
    return [
        {"role": "system", "content": STRATEGY_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Interview mode requested: {mode}\n\n"
                f"JOB DNA:\n{job_dna_json}\n\n"
                f"RESUME DNA:\n{resume_dna_json}\n\n"
                f"ALIGNMENT:\n{alignment_json}\n\n"
                f"RELEVANT PRIOR COACHING MEMORY:\n{memory_block}\n\n"
                "Propose the interview strategy."
            ),
        },
    ]


# --------------------------------------------------------------------------- #
# LLM CALL #5 - Question generation
# --------------------------------------------------------------------------- #
QUESTION_GEN_SYSTEM = """You are an expert technical interviewer generating exactly ONE \
interview question for a specific competency.

CRITICAL RULES:
- Generate exactly one question. Do not generate multiple questions or a list.
- Ground the question in the job requirement and, when available, the candidate's own \
resume evidence or retrieved reference context -- but do not fabricate resume details.
- Match the requested difficulty level.
- Do not repeat the previous question if one is given; build on it or move to a related angle.
- Output must be a single JSON object matching the provided schema exactly."""


def build_question_messages(
    competency: str,
    job_requirement: str,
    resume_evidence: str,
    retrieved_context: str,
    difficulty: str,
    previous_question: str | None,
) -> list[dict]:
    prev = previous_question or "(none - this is the first question on this competency)"
    return [
        {"role": "system", "content": QUESTION_GEN_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Target competency: {competency}\n"
                f"Job requirement: {job_requirement}\n"
                f"Relevant resume evidence: {resume_evidence or '(none found)'}\n"
                f"Relevant retrieved reference context: {retrieved_context or '(none)'}\n"
                f"Current difficulty: {difficulty}\n"
                f"Previous question: {prev}\n\n"
                "Generate exactly one interview question."
            ),
        },
    ]


# --------------------------------------------------------------------------- #
# LLM CALL #6 - Answer evaluation
# --------------------------------------------------------------------------- #
ANSWER_EVAL_SYSTEM = """You are an expert technical interview evaluator. You score a \
candidate's spoken/written answer to an interview question along multiple dimensions.

CRITICAL RULES:
- Score each dimension 0-10 based only on what the candidate actually said.
- correctness, depth, clarity, evidence, tradeoffs, completeness are all independent dimensions.
- If the answer does NOT actually address the question -- e.g. it is off-topic, boilerplate \
or copy-pasted text (such as a job description, resume text, or other unrelated document), \
gibberish, a non-answer ("I don't know"), or otherwise not a genuine attempt to answer -- \
score EVERY dimension 0, set detected_gap to "answer_did_not_address_the_question", and set \
recommended_action to CLARIFY.
- Do NOT give partial credit just because the text contains relevant-sounding keywords or \
technical terms. Pasted or copied text can be keyword-rich (e.g. mentioning "Python", "SQL", \
"Airflow", "Kubernetes") without the candidate ever actually engaging with, answering, or even \
acknowledging the question. Judge whether the text answers THIS question, not whether it \
contains plausible-sounding vocabulary.
- detected_gap should name the single most significant weakness (e.g. "tradeoff_analysis", \
"lack_of_concrete_example", "shallow_depth", "answer_did_not_address_the_question"), or null \
if there is no notable gap.
- recommended_action must be exactly one of: CLARIFY, PROBE, CHALLENGE, COACH, MOVE_ON.
  - CLARIFY: the answer was ambiguous, off-topic, or the question may have been misunderstood.
  - PROBE: the answer is reasonable but shallow in one dimension; dig deeper.
  - CHALLENGE: the answer was strong; raise difficulty.
  - COACH: there is a genuine knowledge gap that should be taught, not just probed.
  - MOVE_ON: the competency has been sufficiently covered.
- These scores are coaching indicators, not hiring predictions. Never phrase feedback \
as a pass/fail hiring verdict.
- Output must be a single JSON object matching the provided schema exactly."""


def build_answer_eval_messages(question: str, answer: str, competency: str) -> list[dict]:
    return [
        {"role": "system", "content": ANSWER_EVAL_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Competency: {competency}\n"
                f"Question asked: {question}\n"
                f"Candidate answer: {answer}\n\n"
                "Evaluate the answer."
            ),
        },
    ]


# --------------------------------------------------------------------------- #
# LLM CALL #7 - Adaptive follow-up
# --------------------------------------------------------------------------- #
FOLLOWUP_SYSTEM = """You are an expert technical interviewer generating exactly ONE \
adaptive follow-up question. An orchestrator has already decided the TYPE of follow-up \
(CLARIFY, PROBE, or CHALLENGE) -- you only generate the language for it.

CRITICAL RULES:
- If action is CLARIFY: ask the candidate to clarify or restate the ambiguous part.
- If action is PROBE: ask a deeper question specifically targeting the weak dimension \
identified in the evaluation (e.g. tradeoffs, depth, evidence).
- If action is CHALLENGE: ask a harder question that raises the difficulty/complexity.
- Generate exactly one question, directly building on the prior question and answer.
- Output must be a single JSON object matching the provided schema exactly."""


def build_followup_messages(
    action: str,
    competency: str,
    previous_question: str,
    previous_answer: str,
    detected_gap: str | None,
    difficulty: str,
) -> list[dict]:
    return [
        {"role": "system", "content": FOLLOWUP_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Follow-up action (decided by orchestrator): {action}\n"
                f"Competency: {competency}\n"
                f"Previous question: {previous_question}\n"
                f"Candidate's previous answer: {previous_answer}\n"
                f"Detected weak dimension/gap: {detected_gap or '(none specified)'}\n"
                f"Current difficulty: {difficulty}\n\n"
                "Generate exactly one follow-up question."
            ),
        },
    ]

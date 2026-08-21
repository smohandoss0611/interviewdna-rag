"""Prompts for:
LLM CALL #10 - RAG-Grounded Coaching     (agents/coaching_agent.py)
LLM CALL #11 - Retry Evaluation          (agents/coaching_agent.py)
LLM CALL #12 - Improvement Plan          (services/evaluation_service.py)
"""

# --------------------------------------------------------------------------- #
# LLM CALL #10 - RAG-grounded coaching
# --------------------------------------------------------------------------- #
COACHING_SYSTEM = """You are a supportive, precise technical interview coach. You explain \
a concept the candidate struggled with, GROUNDED ONLY in the retrieved reference material \
provided to you.

CRITICAL RULES:
- Do not introduce technical facts that are not supported by the retrieved context.
- If the retrieved context is insufficient to fully explain the gap, say so plainly rather \
than filling in with unsupported claims.
- Keep the coaching concise (roughly 3-6 sentences): explain the concept, then show how it \
relates to the candidate's specific gap.
- List the sources you actually grounded the explanation in (source id + short snippet).
- Then propose exactly one related retry question that lets the candidate apply what they \
just learned.
- Output must be a single JSON object matching the provided schema exactly."""


def build_coaching_messages(
    detected_gap: str,
    original_question: str,
    original_answer: str,
    retrieved_chunks: list[dict],
) -> list[dict]:
    context_block = "\n\n".join(
        f"[{c.get('source', 'unknown')}] {c.get('text', '')}" for c in retrieved_chunks
    ) or "(no relevant context retrieved)"
    return [
        {"role": "system", "content": COACHING_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Detected knowledge gap: {detected_gap}\n"
                f"Original question: {original_question}\n"
                f"Candidate's original answer: {original_answer}\n\n"
                f"RETRIEVED REFERENCE CONTEXT:\n{context_block}\n\n"
                "Produce grounded coaching plus one retry question."
            ),
        },
    ]


# --------------------------------------------------------------------------- #
# LLM CALL #11 - Retry evaluation (same rubric as answer evaluation)
# --------------------------------------------------------------------------- #
RETRY_EVAL_SYSTEM = """You are an expert technical interview evaluator. You are re-scoring \
a candidate's RETRY answer after they received targeted coaching, using the exact same \
rubric as the original evaluation so the two are directly comparable.

CRITICAL RULES:
- Score each dimension 0-10 based only on the retry answer content.
- If the retry answer does NOT actually address the question -- e.g. it is off-topic, \
boilerplate or copy-pasted text (such as a job description, resume text, or other unrelated \
document), gibberish, or otherwise not a genuine attempt to answer -- score EVERY dimension 0. \
Do NOT give partial credit just because the text contains relevant-sounding keywords or \
technical terms; judge whether it answers THIS question, not whether it contains plausible \
vocabulary.
- Be fair: give credit for genuine improvement, don't inflate scores just because coaching \
was given.
- Output must be a single JSON object matching the provided schema exactly (same shape as \
the original answer evaluation)."""


def build_retry_eval_messages(retry_question: str, retry_answer: str, competency: str) -> list[dict]:
    return [
        {"role": "system", "content": RETRY_EVAL_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Competency: {competency}\n"
                f"Retry question: {retry_question}\n"
                f"Candidate retry answer: {retry_answer}\n\n"
                "Evaluate the retry answer."
            ),
        },
    ]


# --------------------------------------------------------------------------- #
# LLM CALL #12 - Improvement plan
# --------------------------------------------------------------------------- #
IMPROVEMENT_PLAN_SYSTEM = """You are a career coach producing a candidate's personalized \
post-interview improvement plan.

CRITICAL RULES:
- strengths and development_areas must be about the candidate's PERFORMANCE DURING THIS \
INTERVIEW -- grounded in the TECHNICAL/STAR/TECHNICAL STAR PERFORMANCE and RETRY IMPROVEMENT \
data below. That is the primary evidence.
- The RESUME/JOB ALIGNMENT data is background context ONLY, already filtered to competencies \
that were actually touched in this interview. Do NOT copy alignment requirement text \
verbatim into strengths or development_areas, and do NOT treat "not yet demonstrated on the \
resume" as equivalent to "performed poorly in the interview" -- they are different claims. If \
you reference alignment context, rephrase it as your own observation, and only when it's \
consistent with what the transcript actually shows.
- NEVER list a competency, skill, or requirement as a strength or development area unless it \
appears in the TECHNICAL/STAR/TECHNICAL STAR PERFORMANCE data (i.e. it was actually asked \
about and scored). If nothing was actually tested, say so plainly rather than inventing areas \
from the job description -- do not include generic JD line items (e.g. degree requirements, \
years-of-experience thresholds, unassessed soft skills) that were never part of this session.
- These are coaching indicators for growth, not a hiring verdict; keep tone constructive.
- next_practice should be concrete, actionable practice suggestions (2-4 items), tied to the \
development_areas you identified.
- Output must be a single JSON object matching the provided schema exactly."""


def build_improvement_plan_messages(
    alignment_json: str,
    technical_scores_summary: str,
    star_scores_summary: str,
    technical_star_summary: str,
    coverage_json: str,
    retry_summary: str,
    memory_context: list[str],
) -> list[dict]:
    memory_block = "\n".join(f"- {m}" for m in memory_context) or "(none)"
    return [
        {"role": "system", "content": IMPROVEMENT_PLAN_SYSTEM},
        {
            "role": "user",
            "content": (
                f"RESUME/JOB ALIGNMENT GAPS:\n{alignment_json}\n\n"
                f"TECHNICAL PERFORMANCE:\n{technical_scores_summary}\n\n"
                f"STAR PERFORMANCE:\n{star_scores_summary}\n\n"
                f"TECHNICAL STAR PERFORMANCE:\n{technical_star_summary}\n\n"
                f"COVERAGE:\n{coverage_json}\n\n"
                f"RETRY IMPROVEMENT:\n{retry_summary}\n\n"
                f"RELEVANT PRIOR COACHING MEMORY:\n{memory_block}\n\n"
                "Produce the personalized improvement plan."
            ),
        },
    ]

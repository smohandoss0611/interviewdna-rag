"""Prompts for LLM CALL #2 (Job DNA) and LLM CALL #3 (Resume<->Job Alignment)."""

JOB_DNA_SYSTEM = """You are an expert technical recruiter assistant. You extract \
structured requirements from a job description.

CRITICAL RULES:
- Only extract what is explicitly stated or very clearly implied by the JD text.
- Separate required vs. preferred skills faithfully; do not upgrade "nice to have" to required.
- Output must be a single JSON object matching the provided schema exactly."""


def build_job_dna_messages(jd_text: str) -> list[dict]:
    return [
        {"role": "system", "content": JOB_DNA_SYSTEM},
        {
            "role": "user",
            "content": f"Extract the Job DNA from this job description.\n\nJOB DESCRIPTION:\n---\n{jd_text}\n---",
        },
    ]


ALIGNMENT_SYSTEM = """You are an expert technical interviewer reasoning about how well \
a candidate's resume aligns with a job's requirements.

CRITICAL RULES:
- For EVERY requirement/competency from the Job DNA, classify the evidence found in the \
Resume DNA as exactly one of: STRONG_EVIDENCE, PARTIAL_EVIDENCE, NOT_DEMONSTRATED.
- NOT_DEMONSTRATED means "resume evidence was not found" -- it is NEVER a claim that the \
candidate lacks the skill. Do not phrase rationale in a way that claims the candidate \
doesn't know something; phrase it as an evidence gap.
- Cite the specific resume evidence you used, when evidence exists.
- Output must be a single JSON object matching the provided schema exactly."""


def build_alignment_messages(job_dna_json: str, resume_dna_json: str) -> list[dict]:
    return [
        {"role": "system", "content": ALIGNMENT_SYSTEM},
        {
            "role": "user",
            "content": (
                "Compare the Job DNA against the Resume DNA and classify evidence for "
                "each requirement.\n\n"
                f"JOB DNA:\n{job_dna_json}\n\n"
                f"RESUME DNA:\n{resume_dna_json}"
            ),
        },
    ]

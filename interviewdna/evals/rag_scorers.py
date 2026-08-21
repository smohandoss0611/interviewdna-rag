"""
RAG-specific scorers -- checking retrieval and generation QUALITY, not just
whether the pipeline runs without crashing.

Two different kinds of check here, matching the two different questions RAG
evaluation actually asks:

1. RETRIEVAL quality ("did we find the right stuff, ranked correctly?") --
   deterministic, checked against a small controlled corpus where we know
   in advance which chunk SHOULD be relevant to a given query.

2. GENERATION faithfulness ("does the generated text only say things the
   retrieved context actually supports?") -- this genuinely needs semantic
   judgment, so it uses an LLM as a judge. This is the "second layer"
   evals/README.md already flags: once keyword/threshold checks can't tell
   you enough, a structured LLM-judge call is the next tool to reach for --
   still structured output, still validated against a schema, same pattern
   as every other LLM call in this app.
"""
from __future__ import annotations

from typing import List, Dict, Any

from llm.base import LLMService
from evals.schemas import FaithfulnessJudgment


def relevant_chunk_found(retrieved: List[Dict[str, Any]], expected_substring: str) -> bool:
    """Deterministic check: does ANY retrieved chunk contain the expected
    substring? A simple proxy for retrieval precision/recall when you know
    in advance which specific fact a query should surface."""
    return any(expected_substring.lower() in r.get("text", "").lower() for r in retrieved)


def relevant_chunk_ranked_above(
    retrieved: List[Dict[str, Any]], relevant_substring: str, irrelevant_substring: str
) -> bool:
    """Deterministic check: among the RANKED results, does the relevant
    chunk appear before the irrelevant one? Tests ranking quality, not just
    whether retrieval found the right thing somewhere in a big pile --
    a chunk buried at position #8 is much less useful than one at #1, even
    though a plain 'was it found at all' check can't tell the difference."""
    relevant_rank = next(
        (i for i, r in enumerate(retrieved) if relevant_substring.lower() in r.get("text", "").lower()), None
    )
    irrelevant_rank = next(
        (i for i, r in enumerate(retrieved) if irrelevant_substring.lower() in r.get("text", "").lower()), None
    )
    if relevant_rank is None:
        return False  # didn't even find it
    if irrelevant_rank is None:
        return True  # irrelevant chunk wasn't retrieved at all -- no ranking conflict
    return relevant_rank < irrelevant_rank


FAITHFULNESS_JUDGE_SYSTEM = """You are a strict fact-checker evaluating whether a piece of \
generated text is FAITHFUL to its source context -- meaning every factual claim in the text \
is actually supported by the context, with nothing added that isn't there.

CRITICAL RULES:
- Read the CONTEXT and the GENERATED TEXT carefully.
- List any claims in the generated text that are NOT supported by the context (information \
that was added, embellished, or invented beyond what the context actually states).
- faithful should be true ONLY if there are zero unsupported claims.
- Do not penalize reasonable paraphrasing or summarization -- only flag claims that \
introduce NEW factual content the context doesn't support.
- Output must be a single JSON object matching the provided schema exactly."""


def judge_faithfulness(llm: LLMService, context: str, generated_text: str) -> FaithfulnessJudgment:
    """Eval-only LLM call -- NOT part of the running application. Used
    solely by the eval harness to check whether RAG-grounded generation
    (e.g. coaching text) stays faithful to what was actually retrieved,
    rather than the model filling gaps with plausible-sounding invention."""
    messages = [
        {"role": "system", "content": FAITHFULNESS_JUDGE_SYSTEM},
        {
            "role": "user",
            "content": f"CONTEXT:\n{context}\n\nGENERATED TEXT:\n{generated_text}\n\nJudge faithfulness.",
        },
    ]
    return llm.invoke_structured(messages, schema=FaithfulnessJudgment)

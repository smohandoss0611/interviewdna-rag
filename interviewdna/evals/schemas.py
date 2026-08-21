"""
Schemas used ONLY by the eval harness (evals/), not by the running
application itself. Kept separate from models/schemas.py because these
aren't part of InterviewDNA's actual data model -- they exist purely to
give an LLM-as-judge call somewhere structured to write its verdict.
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class FaithfulnessJudgment(BaseModel):
    """LLM-as-judge output for RAG faithfulness: does a piece of generated
    text ONLY make claims supported by its source context, or does it add
    something the context doesn't actually support (hallucination)?"""
    faithful: bool
    unsupported_claims: List[str] = Field(default_factory=list)
    reasoning: str = ""

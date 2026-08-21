"""
Strategy Agent - LangGraph node wrapping LLM CALL #4 (Interview Strategy Generation).

    LangGraph `strategy_node`
              |
              v
        strategy_agent.generate_strategy()
              |
              v
          LLMService.invoke_structured()

LangGraph remains responsible for controlling the actual interview workflow;
this LLM call only *proposes* priorities and a technical/behavioral split.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, Any, List

from llm.base import LLMService
from models.schemas import InterviewStrategy
from prompts.interview_prompts import build_strategy_messages

logger = logging.getLogger("interviewdna.agents.strategy")


def generate_strategy(
    llm: LLMService,
    resume_dna: Dict[str, Any],
    job_dna: Dict[str, Any],
    alignment: Dict[str, Any],
    mode: str,
    memory_context: List[str],
) -> InterviewStrategy:
    """LLM CALL #4 - Interview Strategy Generation.

    Node/Call site: agents/interview_graph.py `strategy_node` (runs once,
    right after alignment, before the adaptive loop begins).
    Why: turns Resume DNA + Job DNA + Alignment + prior Mem0 coaching signals
    into a prioritized competency list the Coverage Agent (deterministic) and
    the adaptive loop use to decide what to ask next.
    """
    import json

    logger.info("LLM CALL #4: Strategy generation starting (mode=%s)", mode)
    start = time.monotonic()
    messages = build_strategy_messages(
        resume_dna_json=json.dumps(resume_dna),
        job_dna_json=json.dumps(job_dna),
        alignment_json=json.dumps(alignment),
        mode=mode,
        memory_context=memory_context,
    )
    strategy = llm.invoke_structured(messages, schema=InterviewStrategy)
    logger.info(
        "LLM CALL #4: done in %.1fs -- %d priority competencies, %d%%/%d%% technical/behavioral",
        time.monotonic() - start, len(strategy.priority_competencies),
        strategy.technical, strategy.behavioral,
    )
    return strategy

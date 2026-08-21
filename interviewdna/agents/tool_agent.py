"""
Tool-Use Agent -- extends LLM CALL #10 (RAG-Grounded Coaching) with a
genuine "the agent decides what to do" step.

Before this, coaching ALWAYS retrieved from the same fixed source (Pinecone
reference material) no matter what the gap was. This adds a decision layer:

    Detected Knowledge Gap
              |
              v
    LLM CALL: Tool Selection (this module)   <-- the LLM picks a tool
              |
              v
    Execute the chosen tool (this module)     <-- deterministic Python
              |
              v
    ToolResult (chunks) fed into LLM CALL #10 (coaching_agent.py)

This is the actual pattern behind "agentic tool use": the LLM doesn't just
generate text, it can choose an ACTION (which tool, what query) first, and
that choice determines what information the next LLM call gets grounded in.
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional

from llm.base import LLMService
from models.schemas import ToolDecision
from prompts.tool_prompts import build_tool_selection_messages
from tools.base import Tool, ToolResult
from tools.registry import get_default_tools, build_tool_menu, get_tool_by_name

logger = logging.getLogger("interviewdna.agents.tool_agent")


def select_and_run_tool(
    llm: LLMService,
    gap: str,
    competency: str,
    tools: Optional[List[Tool]] = None,
) -> ToolResult:
    """LLM CALL (NEW -- extends #10): Tool Selection.

    Node/Call site: agents/coaching_agent.py `coach_with_retrieved_context`,
    called BEFORE the actual coaching-generation LLM call, replacing what
    used to be a hardcoded Pinecone-only retrieval.
    Why: lets the agent choose the most appropriate information source for
    THIS specific gap, instead of always using the same one -- the
    definition of "tool-use" rather than "fixed retrieval."
    """
    tools = tools if tools is not None else get_default_tools()
    menu = build_tool_menu(tools)

    start = time.monotonic()
    messages = build_tool_selection_messages(gap=gap, competency=competency, tool_menu=menu)
    decision: ToolDecision = llm.invoke_structured(messages, schema=ToolDecision)
    logger.info(
        "Tool selection: %s (query=%r) -- %.1fs -- reasoning: %s",
        decision.tool_name, decision.query, time.monotonic() - start, decision.reasoning,
    )

    if decision.tool_name == "none" or not decision.tool_name:
        return ToolResult(tool_name="none", query="", success=True, chunks=[])

    tool = get_tool_by_name(tools, decision.tool_name)
    if tool is None:
        logger.warning(
            "LLM selected unknown tool '%s' (not in registry) -- treating as no tool used",
            decision.tool_name,
        )
        return ToolResult(tool_name="none", query="", success=False, error="unknown tool selected")

    query = decision.query or gap
    exec_start = time.monotonic()
    result = tool.run(query)
    logger.info(
        "Tool '%s' executed in %.1fs -- success=%s, %d chunk(s) returned",
        tool.name, time.monotonic() - exec_start, result.success, len(result.chunks),
    )
    return result

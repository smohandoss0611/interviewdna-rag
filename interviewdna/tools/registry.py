"""
Tool registry: the single place that knows which tools exist, so
agents/tool_agent.py doesn't need to hardcode tool instances, and so the
LLM's "menu" of available tools (shown in the prompt) always matches what's
actually registered.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from tools.base import Tool


def get_default_tools() -> List[Tool]:
    """Instantiates the tools actually available in this app. Add a new
    tool here (and give it a Tool subclass) to make it available to the
    tool-use agent -- nothing else needs to change."""
    from tools.knowledge_base_tool import KnowledgeBaseTool
    from tools.web_search_tool import WebSearchTool

    return [KnowledgeBaseTool(), WebSearchTool()]


def build_tool_menu(tools: List[Tool]) -> str:
    """Renders the available tools as text for the LLM's tool-selection
    prompt, e.g.:
        - search_local_knowledge_base: Search InterviewDNA's own indexed...
        - search_web: Search the live web for current or niche information...
    """
    return "\n".join(f"- {t.name}: {t.description}" for t in tools)


def get_tool_by_name(tools: List[Tool], name: str) -> Optional[Tool]:
    for t in tools:
        if t.name == name:
            return t
    return None

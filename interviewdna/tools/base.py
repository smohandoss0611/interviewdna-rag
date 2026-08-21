"""
Tool interface.

A "tool" is anything an agent can DECIDE to call to get information it
doesn't already have -- as opposed to just generating text from what the
model already knows. This is the core pattern behind "agentic" AI systems:
the LLM doesn't just answer, it can choose to ACT first, then answer using
what it found.

Every tool implements this same tiny interface, so the tool-use agent
(agents/tool_agent.py) can present a uniform menu of options to the LLM
regardless of what's actually behind each tool (a vector database, a web
search API, a calculator, anything).
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class ToolResult:
    tool_name: str
    query: str
    success: bool
    chunks: List[Dict[str, Any]] = field(default_factory=list)  # each: {"text":..., "source":...}
    error: str = ""


class Tool(abc.ABC):
    """Base class every tool implements."""

    name: str
    description: str  # shown to the LLM when it's deciding which tool to use

    @abc.abstractmethod
    def run(self, query: str) -> ToolResult:
        raise NotImplementedError

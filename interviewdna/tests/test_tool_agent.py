from unittest.mock import MagicMock

from agents.tool_agent import select_and_run_tool
from models.schemas import ToolDecision
from tools.base import Tool, ToolResult
from tests.conftest import FakeLLMService


class _FakeTool(Tool):
    def __init__(self, name: str, chunks=None, fail=False):
        self.name = name
        self.description = f"fake tool {name}"
        self._chunks = chunks or []
        self._fail = fail

    def run(self, query: str) -> ToolResult:
        if self._fail:
            return ToolResult(tool_name=self.name, query=query, success=False, error="simulated failure")
        return ToolResult(tool_name=self.name, query=query, success=True, chunks=self._chunks)


def test_select_and_run_tool_executes_the_chosen_tool():
    decision = ToolDecision(tool_name="tool_a", query="kubernetes networking", reasoning="niche topic")
    llm = FakeLLMService(structured_responses={"ToolDecision": decision})
    tool_a = _FakeTool("tool_a", chunks=[{"text": "k8s networking info", "source": "web"}])
    tool_b = _FakeTool("tool_b")

    result = select_and_run_tool(llm, gap="kubernetes networking", competency="Infra", tools=[tool_a, tool_b])

    assert result.tool_name == "tool_a"
    assert result.success is True
    assert result.chunks == [{"text": "k8s networking info", "source": "web"}]


def test_select_and_run_tool_respects_none_decision():
    decision = ToolDecision(tool_name="none", query="", reasoning="already well covered")
    llm = FakeLLMService(structured_responses={"ToolDecision": decision})
    tool_a = _FakeTool("tool_a", chunks=[{"text": "should not be used", "source": "x"}])

    result = select_and_run_tool(llm, gap="basic python syntax", competency="Python", tools=[tool_a])

    assert result.tool_name == "none"
    assert result.chunks == []


def test_select_and_run_tool_handles_unknown_tool_name_gracefully():
    """If the LLM hallucinates a tool name that isn't registered, don't crash --
    treat it as if no tool was used."""
    decision = ToolDecision(tool_name="made_up_tool_that_does_not_exist", query="x", reasoning="y")
    llm = FakeLLMService(structured_responses={"ToolDecision": decision})
    tool_a = _FakeTool("tool_a")

    result = select_and_run_tool(llm, gap="gap", competency="comp", tools=[tool_a])

    assert result.tool_name == "none"
    assert result.success is False


def test_select_and_run_tool_falls_back_to_gap_when_query_empty():
    decision = ToolDecision(tool_name="tool_a", query="", reasoning="y")
    llm = FakeLLMService(structured_responses={"ToolDecision": decision})
    tool_a = MagicMock()
    tool_a.name = "tool_a"
    tool_a.run.return_value = ToolResult(tool_name="tool_a", query="fallback", success=True, chunks=[])

    select_and_run_tool(llm, gap="the actual gap text", competency="comp", tools=[tool_a])

    tool_a.run.assert_called_once_with("the actual gap text")


def test_select_and_run_tool_surfaces_tool_failure():
    decision = ToolDecision(tool_name="tool_a", query="q", reasoning="y")
    llm = FakeLLMService(structured_responses={"ToolDecision": decision})
    tool_a = _FakeTool("tool_a", fail=True)

    result = select_and_run_tool(llm, gap="gap", competency="comp", tools=[tool_a])

    assert result.success is False
    assert result.error == "simulated failure"

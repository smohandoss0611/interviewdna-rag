"""Prompt for the tool-selection decision (tools/, agents/tool_agent.py).

This is the "agent decides which action to take" step: rather than always
retrieving from the same fixed source, the LLM is shown a menu of available
tools and picks one (or none) based on what's actually needed."""

TOOL_SELECTION_SYSTEM = """You are deciding which tool (if any) to use to find \
grounding information for a coaching explanation.

CRITICAL RULES:
- Pick exactly ONE tool from the menu below, by its exact name, or "none" if no \
tool is likely to help.
- Prefer the local knowledge base for well-established, likely-already-indexed technical \
concepts. Prefer web search for niche, very recent, or unusual topics unlikely to be \
pre-indexed.
- query should be a short, focused search query (not the full original question) -- \
extract just the core concept that needs grounding.
- reasoning should be ONE short sentence explaining your choice, in plain language \
suitable for showing directly to the person using this tool.
- Output must be a single JSON object matching the provided schema exactly."""


def build_tool_selection_messages(gap: str, competency: str, tool_menu: str) -> list[dict]:
    return [
        {"role": "system", "content": TOOL_SELECTION_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Competency: {competency}\n"
                f"Detected knowledge gap: {gap}\n\n"
                f"Available tools:\n{tool_menu}\n\n"
                "Decide which tool to use."
            ),
        },
    ]

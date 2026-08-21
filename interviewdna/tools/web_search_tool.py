"""
Web search tool -- lets the agent search the live web when its own indexed
knowledge base doesn't have good grounding for something.

Uses DuckDuckGo's HTML endpoint (https://html.duckduckgo.com/html/), which
returns plain server-rendered HTML and needs no API key -- good for a
zero-setup default. This is a genuinely fragile approach long-term (DDG can
change their markup at any time, and scraping search results may violate
their terms of service for anything beyond light personal/educational use)
-- for real production use, swap this out for a proper search API (Tavily,
Serper, Bing Search) behind the SAME Tool interface. That swap requires
touching only this one file, same pattern as llm/factory.py for LLM
providers.

The HTML parsing is split into its own pure function (_parse_ddg_html) so
it's testable without a live network call.
"""
from __future__ import annotations

import logging
import re
from typing import List, Dict

import requests

from tools.base import Tool, ToolResult

logger = logging.getLogger("interviewdna.tools.web_search")

DDG_HTML_URL = "https://html.duckduckgo.com/html/"

# Matches DuckDuckGo's HTML-lite result blocks: a result link (title + url)
# followed by a snippet, each wrapped in a result__body div. This is
# intentionally a lightweight regex, not a full HTML parser -- good enough
# for this well-known, stable-ish markup shape, and avoids adding a heavy
# HTML-parsing dependency for one scraper.
_RESULT_RE = re.compile(
    r'result__a"[^>]*>(?P<title>.*?)</a>.*?'
    r'result__snippet[^>]*>(?P<snippet>.*?)</a>',
    re.DOTALL,
)


def _strip_tags(html_fragment: str) -> str:
    return re.sub(r"<[^>]+>", "", html_fragment).strip()


def _parse_ddg_html(html: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Pure parsing function -- takes raw HTML, returns [{"title", "snippet"}].
    Kept separate from the network call so it's unit-testable offline."""
    results = []
    for match in _RESULT_RE.finditer(html):
        title = _strip_tags(match.group("title"))
        snippet = _strip_tags(match.group("snippet"))
        if title or snippet:
            results.append({"title": title, "snippet": snippet})
        if len(results) >= max_results:
            break
    return results


class WebSearchTool(Tool):
    name = "search_web"
    description = (
        "Search the live web for current or niche information not likely to "
        "be in a pre-indexed knowledge base -- e.g. recent tooling changes, "
        "very specific technical edge cases, or anything time-sensitive."
    )

    def __init__(self, max_results: int = 4, timeout: int = 10):
        self.max_results = max_results
        self.timeout = timeout

    def run(self, query: str) -> ToolResult:
        try:
            resp = requests.get(
                DDG_HTML_URL,
                params={"q": query},
                headers={"User-Agent": "Mozilla/5.0 (InterviewDNA educational tool)"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            parsed = _parse_ddg_html(resp.text, max_results=self.max_results)
            logger.info("Web search for %r returned %d result(s)", query, len(parsed))
            return ToolResult(
                tool_name=self.name,
                query=query,
                success=True,
                chunks=[{"text": f"{r['title']}: {r['snippet']}", "source": "web_search"} for r in parsed],
            )
        except requests.RequestException as exc:
            logger.warning("Web search failed for %r: %s", query, exc)
            return ToolResult(tool_name=self.name, query=query, success=False, error=str(exc))

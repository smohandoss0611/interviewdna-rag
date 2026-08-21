"""
Knowledge base tool -- wraps rag/hybrid_retriever.py (vector + BM25 +
rerank) as a Tool the agent can choose to call. This is the "search my own
indexed reference material" option.
"""
from __future__ import annotations

from tools.base import Tool, ToolResult
from rag.hybrid_retriever import hybrid_retrieve


class KnowledgeBaseTool(Tool):
    name = "search_local_knowledge_base"
    description = (
        "Search InterviewDNA's own indexed reference material (technical "
        "reference docs, previously ingested resume/JD content) using hybrid "
        "vector+keyword search. Best for well-established technical concepts "
        "likely already covered by indexed reference material."
    )

    def __init__(self, top_k: int = 4, document_type: str = "reference"):
        self.top_k = top_k
        self.document_type = document_type

    def run(self, query: str) -> ToolResult:
        try:
            results = hybrid_retrieve(query, top_k=self.top_k, document_type=self.document_type)
            return ToolResult(
                tool_name=self.name,
                query=query,
                success=True,
                chunks=[{"text": r.get("text", ""), "source": r.get("source", "knowledge_base")} for r in results],
            )
        except Exception as exc:
            return ToolResult(tool_name=self.name, query=query, success=False, error=str(exc))

from unittest.mock import patch

from tools.knowledge_base_tool import KnowledgeBaseTool


def test_knowledge_base_tool_wraps_hybrid_retrieve_results():
    fake_results = [
        {"text": "Kubernetes pods are the smallest deployable unit.", "source": "reference"},
    ]
    tool = KnowledgeBaseTool(top_k=3)
    with patch("tools.knowledge_base_tool.hybrid_retrieve", return_value=fake_results) as mock_retrieve:
        result = tool.run("Kubernetes pods")

    mock_retrieve.assert_called_once()
    assert result.success is True
    assert result.tool_name == "search_local_knowledge_base"
    assert len(result.chunks) == 1
    assert "smallest deployable unit" in result.chunks[0]["text"]


def test_knowledge_base_tool_handles_retrieval_failure():
    tool = KnowledgeBaseTool()
    with patch("tools.knowledge_base_tool.hybrid_retrieve", side_effect=RuntimeError("pinecone down")):
        result = tool.run("query")

    assert result.success is False
    assert "pinecone down" in result.error

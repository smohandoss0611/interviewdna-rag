from unittest.mock import patch, MagicMock

from tools.web_search_tool import _parse_ddg_html, WebSearchTool

SAMPLE_DDG_HTML = """
<div class="result results_links results_links_deep web-result">
  <div class="result__body links_main links_deep result__check">
    <h2 class="result__title">
      <a rel="nofollow" href="//duckduckgo.com/l/?uddg=x" class="result__a">Apache Kafka Documentation</a>
    </h2>
    <a class="result__snippet" href="//example.com">Apache Kafka is a distributed event streaming platform used for high-throughput pipelines.</a>
  </div>
</div>
<div class="result results_links results_links_deep web-result">
  <div class="result__body links_main links_deep result__check">
    <h2 class="result__title">
      <a rel="nofollow" href="//duckduckgo.com/l/?uddg=y" class="result__a">Kafka vs RabbitMQ</a>
    </h2>
    <a class="result__snippet" href="//example.com">A comparison of Kafka and RabbitMQ messaging systems.</a>
  </div>
</div>
"""


def test_parse_ddg_html_extracts_title_and_snippet():
    results = _parse_ddg_html(SAMPLE_DDG_HTML, max_results=5)
    assert len(results) == 2
    assert results[0]["title"] == "Apache Kafka Documentation"
    assert "distributed event streaming" in results[0]["snippet"]
    assert results[1]["title"] == "Kafka vs RabbitMQ"


def test_parse_ddg_html_respects_max_results():
    results = _parse_ddg_html(SAMPLE_DDG_HTML, max_results=1)
    assert len(results) == 1


def test_parse_ddg_html_handles_no_results():
    results = _parse_ddg_html("<html><body>no results here</body></html>", max_results=5)
    assert results == []


def test_web_search_tool_run_success():
    tool = WebSearchTool(max_results=5)
    fake_response = MagicMock()
    fake_response.text = SAMPLE_DDG_HTML
    fake_response.raise_for_status = MagicMock()

    with patch("tools.web_search_tool.requests.get", return_value=fake_response) as mock_get:
        result = tool.run("Kafka streaming")

    mock_get.assert_called_once()
    assert result.success is True
    assert len(result.chunks) == 2
    assert "Apache Kafka Documentation" in result.chunks[0]["text"]
    assert all(c["source"] == "web_search" for c in result.chunks)


def test_web_search_tool_run_handles_network_failure():
    import requests

    tool = WebSearchTool()
    with patch("tools.web_search_tool.requests.get", side_effect=requests.ConnectionError("no network")):
        result = tool.run("query")

    assert result.success is False
    assert "no network" in result.error

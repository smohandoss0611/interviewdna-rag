from unittest.mock import MagicMock, patch

from llm.openai_service import _rate_limit_wait_seconds, OpenAILLMService


def test_rate_limit_wait_detects_by_message_heuristic():
    """Providers that don't raise a real openai.RateLimitError (e.g. a
    generic requests exception with '429' in the message) should still be
    recognized as rate limiting via the text fallback."""
    assert _rate_limit_wait_seconds(RuntimeError("429 Too Many Requests")) is not None
    assert _rate_limit_wait_seconds(RuntimeError("You are being rate limited")) is not None


def test_rate_limit_wait_returns_none_for_unrelated_errors():
    """A JSON validation failure is NOT a rate limit -- must not be treated
    as one, or we'd wait pointlessly instead of correcting the prompt."""
    assert _rate_limit_wait_seconds(RuntimeError("json_validate_failed")) is None
    assert _rate_limit_wait_seconds(ValueError("invalid schema")) is None


def test_rate_limit_wait_respects_cap():
    """A malicious or buggy Retry-After header shouldn't stall the request
    for an unreasonable amount of time."""
    fake_response = MagicMock()
    fake_response.headers = {"retry-after": "99999"}

    class FakeRateLimitError(Exception):
        def __init__(self):
            self.response = fake_response

    exc = FakeRateLimitError()
    with patch("openai.RateLimitError", FakeRateLimitError):
        wait = _rate_limit_wait_seconds(exc, cap=30.0)
        assert wait == 30.0


def test_client_disables_sdk_automatic_retries():
    """Regression test: the openai SDK's own hidden automatic retry-on-429
    was stacking with our own retry loop, causing compounding, invisible
    delays (a single one of OUR attempts could silently block for 16+
    seconds inside the SDK). max_retries=0 on the client hands that
    control entirely to our own explicit, logged retry logic."""
    svc = OpenAILLMService(api_key="fake", base_url="https://api.groq.com/openai/v1")
    with patch("openai.OpenAI") as mock_openai_cls:
        svc._get_client()
        _, kwargs = mock_openai_cls.call_args
        assert kwargs.get("max_retries") == 0

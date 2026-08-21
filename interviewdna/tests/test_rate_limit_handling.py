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


def test_rate_limit_on_last_attempt_still_gets_a_real_retry():
    """Regression test for a real production bug: a rate limit hitting on
    the LAST content-attempt (e.g. attempt 3/3) would sleep for the full
    wait time and then immediately fail anyway, because there was no
    remaining loop iteration to actually use -- an 18-second wait that
    accomplished nothing. Rate limits must get their own separate retry
    budget that doesn't get exhausted by the content-attempt counter."""
    from models.schemas import ResumeDNA

    class FakeRateLimitError(Exception):
        def __init__(self):
            self.response = MagicMock(headers={})

    svc = OpenAILLMService(api_key="fake", base_url="https://api.groq.com/openai/v1")

    call_count = {"n": 0}

    def fake_create(**kwargs):
        call_count["n"] += 1
        if call_count["n"] <= 3:
            # Every "real" attempt hits a rate limit, including what would
            # be the last one under the OLD (buggy) attempt-counting logic.
            raise FakeRateLimitError()
        # On the 4th actual API call (made possible only because rate
        # limits now have their own separate budget), succeed.
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content='{"skills": ["Python"]}'))]
        return resp

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = fake_create
    del fake_client.chat.completions.parse

    with patch("openai.RateLimitError", FakeRateLimitError), \
         patch("time.sleep"):  # don't actually wait in the test
        result = svc._invoke_structured_json_mode(
            fake_client, [{"role": "user", "content": "x"}], ResumeDNA,
            temperature=0.2, max_retries=2, overall_start=0,
        )

    # Succeeded despite 3 consecutive rate limits -- proves rate-limit
    # retries don't get exhausted by the same counter as content attempts.
    assert result.skills == ["Python"]
    assert call_count["n"] == 4

from unittest.mock import MagicMock

from llm.openai_service import _extract_json_object, OpenAILLMService


def test_extract_json_object_handles_pure_json():
    text = '{"skills": ["Python"], "languages": []}'
    assert _extract_json_object(text) == text


def test_extract_json_object_strips_markdown_fences():
    text = '```json\n{"skills": ["Python"]}\n```'
    result = _extract_json_object(text)
    assert result.strip().startswith("{")
    assert result.strip().endswith("}")
    import json
    assert json.loads(result) == {"skills": ["Python"]}


def test_extract_json_object_handles_leaked_reasoning_text_before_and_after():
    """Regression test: this is the exact failure mode that caused the real
    Groq 'json_validate_failed' error -- a reasoning model (gpt-oss) mixing
    its internal thinking into the response around the actual JSON."""
    text = (
        "Let me think about this step by step. The candidate has Python "
        "and SQL experience...\n\n"
        '{"skills": ["Python", "SQL"], "languages": []}'
        "\n\nThat should be the correct extraction."
    )
    result = _extract_json_object(text)
    import json
    parsed = json.loads(result)
    assert parsed == {"skills": ["Python", "SQL"], "languages": []}


def test_extract_json_object_falls_back_to_original_when_no_braces_found():
    text = "no json here at all"
    assert _extract_json_object(text) == text


def test_reasoning_extra_body_added_for_groq():
    svc = OpenAILLMService(api_key="fake", base_url="https://api.groq.com/openai/v1")
    assert svc._reasoning_extra_body() == {"reasoning_format": "hidden"}


def test_reasoning_extra_body_not_added_for_plain_openai():
    svc = OpenAILLMService(api_key="fake", base_url=None)
    assert svc._reasoning_extra_body() == {}


def test_reasoning_extra_body_not_added_for_other_compatible_providers():
    svc = OpenAILLMService(api_key="fake", base_url="https://api.together.xyz/v1")
    assert svc._reasoning_extra_body() == {}


def test_json_mode_retry_actually_adapts_on_server_side_error():
    """Regression test: previously, a server-side error (like Groq's
    json_validate_failed) was caught by the generic `except Exception`
    branch but never added a corrective message -- so every retry sent the
    EXACT same request and failed identically 3 times in a row. Now it
    should add a corrective message so retries have a real chance."""
    from models.schemas import ResumeDNA

    svc = OpenAILLMService(api_key="fake", base_url="https://api.groq.com/openai/v1")

    message_snapshots = []

    def fake_create(**kwargs):
        # Snapshot the message list's LENGTH at call time -- kwargs["messages"]
        # is the same mutating list object across calls (it's appended to
        # in-place between retries), so we must record a length/copy now,
        # not a reference we'd read back later after further mutation.
        message_snapshots.append(len(kwargs["messages"]))
        raise RuntimeError("simulated json_validate_failed")

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = fake_create
    # Force the JSON-mode fallback path (skip native .parse()).
    del fake_client.chat.completions.parse

    try:
        svc._invoke_structured_json_mode(fake_client, [{"role": "user", "content": "x"}], ResumeDNA, 0.2, 2, 0)
    except Exception:
        pass

    assert len(message_snapshots) == 3  # max_retries=2 -> 3 total attempts
    # The important assertion: each successive attempt sent MORE messages
    # than the last (i.e. a corrective message was actually appended
    # between attempts, not just blindly repeating the same request).
    assert message_snapshots[0] < message_snapshots[1] < message_snapshots[2]

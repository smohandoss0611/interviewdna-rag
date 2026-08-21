from models.schemas import InterviewRetryResponse


def test_interview_retry_response_carries_completion_and_next_question():
    """Regression test: /interview/retry previously dropped whether the graph
    finished after the retry, and dropped the next question if it didn't --
    the frontend had no way to know whether to show Results or a new
    question, so it silently showed neither. This locks in the fields that
    fix that."""
    # Interview finished right after the retry.
    finished = InterviewRetryResponse(
        session_id="s1", before={"correctness": 4}, retry={"correctness": 8},
        interview_complete=True,
    )
    assert finished.interview_complete is True
    assert finished.next_question is None

    # Interview continues with a new competency after the retry.
    continuing = InterviewRetryResponse(
        session_id="s1", before={"correctness": 4}, retry={"correctness": 8},
        interview_complete=False,
        next_question="Tell me about a time you scaled a system.",
        next_competency="System Design",
    )
    assert continuing.interview_complete is False
    assert continuing.next_question is not None
    assert continuing.next_competency == "System Design"


def test_interview_retry_response_defaults_to_incomplete():
    """Backward-compat: omitting the new fields shouldn't break construction,
    and should default to 'not complete, no next question' rather than
    silently asserting completion."""
    resp = InterviewRetryResponse(session_id="s1", before={}, retry={})
    assert resp.interview_complete is False
    assert resp.next_question is None
    assert resp.next_competency is None

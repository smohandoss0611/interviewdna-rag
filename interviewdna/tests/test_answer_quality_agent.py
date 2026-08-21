from agents.answer_quality_agent import evaluate_answer
from models.schemas import AnswerEvaluation, RecommendedAction
from tests.conftest import FakeLLMService


def test_evaluate_answer_wraps_llm_output_with_structured_evidence():
    fake_eval = AnswerEvaluation(
        correctness=8,
        depth=4,
        clarity=7,
        evidence=3,
        tradeoffs=2,
        completeness=6,
        detected_gap="tradeoff_analysis",
        recommended_action=RecommendedAction.PROBE,
    )
    llm = FakeLLMService(structured_responses={"AnswerEvaluation": fake_eval})

    result = evaluate_answer(llm, question="Explain X", answer="It's Y", competency="System Design")

    assert result.recommended_action == RecommendedAction.PROBE
    assert result.scores["correctness"] == 8
    assert "tradeoff_analysis" in result.weakness
    assert "correctness" in result.strength
    # Must not just return a bare score -- structured evidence required
    assert result.scores and result.strength and result.weakness


def test_evaluate_answer_does_not_call_a_weak_score_a_strength():
    """Regression test: a uniformly weak answer (nothing above the
    STRENGTH_THRESHOLD) must NOT be told its "strength" is whatever scored
    highest -- e.g. calling 2/10 correctness a "strength" is false and
    actively bad coaching feedback."""
    fake_eval = AnswerEvaluation(
        correctness=2, depth=1, clarity=1, evidence=1, tradeoffs=0, completeness=1,
        detected_gap="lack_of_concrete_example",
        recommended_action=RecommendedAction.CLARIFY,
    )
    llm = FakeLLMService(structured_responses={"AnswerEvaluation": fake_eval})

    result = evaluate_answer(llm, question="Explain X", answer="I don't know", competency="System Design")

    assert "strong" not in result.strength.lower(), (
        f"Should not claim a strength when the best score is only 2/10, got: {result.strength!r}"
    )
    assert "no clear strength" in result.strength.lower()
    assert "correctness" in result.strength  # still names the comparatively-best dimension


def test_evaluate_answer_does_call_a_genuinely_strong_score_a_strength():
    """A dimension that actually clears the bar (>= 6/10) should still be
    labeled a real strength."""
    fake_eval = AnswerEvaluation(
        correctness=9, depth=8, clarity=7, evidence=6, tradeoffs=5, completeness=6,
        recommended_action=RecommendedAction.MOVE_ON,
    )
    llm = FakeLLMService(structured_responses={"AnswerEvaluation": fake_eval})

    result = evaluate_answer(llm, question="Explain X", answer="Solid answer", competency="System Design")

    assert "strong correctness" in result.strength.lower()

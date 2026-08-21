"""
Tests for the eval harness ITSELF -- these prove the harness correctly
distinguishes good model behavior from the exact bad behavior we found
manually this session. Unlike evals/run_evals.py (which needs a real LLM
and takes real time/money), these use FakeLLMService so they run in the
normal `pytest` suite, in milliseconds, on every commit.

Think of it as: pytest checks "does the eval harness work correctly?",
while evals/run_evals.py checks "does the actual model behave correctly?"
Two different questions, two different tools.
"""
from evals.cases import eval_offtopic_answer_scores_near_zero, eval_resume_extraction_finds_obvious_skills
from models.schemas import AnswerEvaluation, RecommendedAction, ResumeDNA
from tests.conftest import FakeLLMService


def test_harness_catches_offtopic_answer_getting_partial_credit():
    """Regression: a copy-pasted JD 'answer' that gets non-trivial scores
    (the exact bug we found manually) must make this eval FAIL."""
    buggy_eval = AnswerEvaluation(
        correctness=6, depth=5, clarity=7, evidence=6, tradeoffs=4, completeness=5,
        recommended_action=RecommendedAction.PROBE,
    )
    llm = FakeLLMService(structured_responses={"AnswerEvaluation": buggy_eval})
    result = eval_offtopic_answer_scores_near_zero(llm)
    assert result.passed is False


def test_harness_passes_correct_offtopic_handling():
    """A model that correctly zeroes out an off-topic answer should PASS."""
    good_eval = AnswerEvaluation(
        correctness=0, depth=0, clarity=0, evidence=0, tradeoffs=0, completeness=0,
        detected_gap="answer_did_not_address_the_question",
        recommended_action=RecommendedAction.CLARIFY,
    )
    llm = FakeLLMService(structured_responses={"AnswerEvaluation": good_eval})
    result = eval_offtopic_answer_scores_near_zero(llm)
    assert result.passed is True


def test_harness_catches_empty_resume_extraction():
    """Regression: the exact 'every field empty' bug from earlier this
    session must make this eval FAIL, not silently pass."""
    empty_dna = ResumeDNA()
    llm = FakeLLMService(structured_responses={"ResumeDNA": empty_dna})
    result = eval_resume_extraction_finds_obvious_skills(llm)
    assert result.passed is False


def test_harness_passes_good_resume_extraction():
    good_dna = ResumeDNA(skills=["Python", "SQL"], cloud_technologies=["AWS"])
    llm = FakeLLMService(structured_responses={"ResumeDNA": good_dna})
    result = eval_resume_extraction_finds_obvious_skills(llm)
    assert result.passed is True

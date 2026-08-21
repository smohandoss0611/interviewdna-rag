import pytest
from pydantic import ValidationError

from models.schemas import AnswerEvaluation, RecommendedAction, EvidenceLevel, AlignmentItem


def test_answer_evaluation_rejects_out_of_range_scores():
    with pytest.raises(ValidationError):
        AnswerEvaluation(
            correctness=11,  # out of 0-10 range
            depth=5, clarity=5, evidence=5, tradeoffs=5, completeness=5,
            recommended_action=RecommendedAction.MOVE_ON,
        )


def test_answer_evaluation_accepts_valid_payload():
    ev = AnswerEvaluation(
        correctness=8, depth=6, clarity=8, evidence=5, tradeoffs=4, completeness=7,
        detected_gap="tradeoff_analysis",
        recommended_action=RecommendedAction.PROBE,
    )
    assert ev.recommended_action == "PROBE"


def test_alignment_item_not_demonstrated_is_a_valid_evidence_level():
    item = AlignmentItem(
        requirement="Kubernetes",
        evidence_level=EvidenceLevel.NOT_DEMONSTRATED,
        rationale="No resume evidence found for container orchestration experience.",
    )
    assert item.evidence_level == EvidenceLevel.NOT_DEMONSTRATED

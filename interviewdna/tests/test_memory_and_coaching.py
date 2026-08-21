from agents.coaching_agent import compare_before_retry
from memory.mem0_service import derive_memory_writes
from models.schemas import AnswerEvaluation, RecommendedAction

# NOTE: build_retrieval_query() was retired when coaching's retrieval step
# became a genuine tool-use decision (agents/tool_agent.py) instead of a
# fixed deterministic query -- that behavior is now covered by
# tests/test_tool_agent.py instead.


def test_derive_memory_writes_coach_action_creates_gap_memory():
    evaluation = {
        "scores": {"correctness": 4, "depth": 3, "clarity": 6, "evidence": 2, "tradeoffs": 2, "completeness": 5},
        "detected_gap": "vector_db_internals",
    }
    memories = derive_memory_writes("RAG", evaluation, agent_action="COACH")
    assert any("vector_db_internals" in m for m in memories)


def test_derive_memory_writes_does_not_store_every_answer():
    # A MOVE_ON with all mid-range scores should NOT trigger a strength or gap write.
    evaluation = {"scores": {"correctness": 7, "depth": 6, "clarity": 7, "evidence": 6, "tradeoffs": 6, "completeness": 7}}
    memories = derive_memory_writes("Python", evaluation, agent_action="MOVE_ON")
    assert memories == []


def test_compare_before_retry_produces_before_and_retry_tables():
    before = {"scores": {"correctness": 6, "depth": 5, "clarity": 6, "evidence": 3, "tradeoffs": 3, "completeness": 6}}
    retry = AnswerEvaluation(
        correctness=8, depth=8, clarity=8, evidence=7, tradeoffs=7, completeness=8,
        recommended_action=RecommendedAction.MOVE_ON,
    )
    comparison = compare_before_retry(before, retry)
    assert comparison["before"]["correctness"] == 6
    assert comparison["retry"]["correctness"] == 8

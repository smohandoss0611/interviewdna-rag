from evals.rag_scorers import relevant_chunk_found, relevant_chunk_ranked_above, judge_faithfulness
from evals.schemas import FaithfulnessJudgment
from tests.conftest import FakeLLMService


# --------------------------------------------------------------------------- #
# relevant_chunk_found
# --------------------------------------------------------------------------- #
def test_relevant_chunk_found_true_when_present():
    retrieved = [{"text": "Kafka partitions increase consumer parallelism."}]
    assert relevant_chunk_found(retrieved, "consumer parallelism") is True


def test_relevant_chunk_found_false_when_absent():
    retrieved = [{"text": "PostgreSQL table partitioning improves query performance."}]
    assert relevant_chunk_found(retrieved, "consumer parallelism") is False


def test_relevant_chunk_found_case_insensitive():
    retrieved = [{"text": "CONSUMER PARALLELISM is affected by partition count."}]
    assert relevant_chunk_found(retrieved, "consumer parallelism") is True


def test_relevant_chunk_found_empty_results():
    assert relevant_chunk_found([], "anything") is False


# --------------------------------------------------------------------------- #
# relevant_chunk_ranked_above
# --------------------------------------------------------------------------- #
def test_relevant_ranked_above_irrelevant_true_when_correctly_ordered():
    retrieved = [
        {"text": "Kafka partitions increase consumer parallelism."},
        {"text": "PostgreSQL table partitioning improves query performance."},
    ]
    assert relevant_chunk_ranked_above(retrieved, "consumer parallelism", "PostgreSQL table") is True


def test_relevant_ranked_above_irrelevant_false_when_backwards():
    """This is the actual regression case worth catching: a bad reranker or
    fusion bug could put the wrong chunk first even if both were retrieved."""
    retrieved = [
        {"text": "PostgreSQL table partitioning improves query performance."},
        {"text": "Kafka partitions increase consumer parallelism."},
    ]
    assert relevant_chunk_ranked_above(retrieved, "consumer parallelism", "PostgreSQL table") is False


def test_relevant_ranked_above_false_when_relevant_not_found_at_all():
    retrieved = [{"text": "PostgreSQL table partitioning improves query performance."}]
    assert relevant_chunk_ranked_above(retrieved, "consumer parallelism", "PostgreSQL table") is False


def test_relevant_ranked_above_true_when_irrelevant_wasnt_retrieved():
    """If the irrelevant chunk wasn't even retrieved, there's no ranking
    conflict to fail on -- the relevant one being present at all is fine."""
    retrieved = [{"text": "Kafka partitions increase consumer parallelism."}]
    assert relevant_chunk_ranked_above(retrieved, "consumer parallelism", "PostgreSQL table") is True


# --------------------------------------------------------------------------- #
# judge_faithfulness (LLM-as-judge, tested with FakeLLMService)
# --------------------------------------------------------------------------- #
def test_judge_faithfulness_returns_structured_verdict():
    fake_judgment = FaithfulnessJudgment(
        faithful=False,
        unsupported_claims=["The context never mentioned rebalance timing thresholds."],
        reasoning="Generated text invented a specific numeric threshold not present in context.",
    )
    llm = FakeLLMService(structured_responses={"FaithfulnessJudgment": fake_judgment})

    result = judge_faithfulness(llm, context="Kafka partitions affect parallelism.",
                                 generated_text="Kafka rebalances trigger after exactly 30 seconds.")

    assert result.faithful is False
    assert len(result.unsupported_claims) == 1


def test_judge_faithfulness_true_case():
    fake_judgment = FaithfulnessJudgment(faithful=True, unsupported_claims=[], reasoning="All claims supported.")
    llm = FakeLLMService(structured_responses={"FaithfulnessJudgment": fake_judgment})

    result = judge_faithfulness(llm, context="Kafka partitions affect parallelism.",
                                 generated_text="More partitions generally means more parallelism.")

    assert result.faithful is True
    assert result.unsupported_claims == []

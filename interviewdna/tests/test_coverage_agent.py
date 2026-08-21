from agents import coverage_agent


STRATEGY = {
    "technical": 60,
    "behavioral": 40,
    "priority_competencies": [
        {"name": "RAG", "priority": "HIGH", "reason": "critical"},
        {"name": "Python", "priority": "MEDIUM", "reason": "core skill"},
        {"name": "Leadership", "priority": "LOW", "reason": "nice to have"},
    ],
}


def test_init_coverage_seeds_not_tested():
    coverage = coverage_agent.init_coverage(STRATEGY)
    assert coverage == {"RAG": "NOT_TESTED", "Python": "NOT_TESTED", "Leadership": "NOT_TESTED"}


def test_select_next_competency_prefers_high_priority_not_tested():
    coverage = coverage_agent.init_coverage(STRATEGY)
    next_comp = coverage_agent.select_next_competency(STRATEGY, coverage, questions_asked=0, question_budget=8)
    assert next_comp == "RAG"


def test_select_next_competency_skips_tested():
    coverage = coverage_agent.init_coverage(STRATEGY)
    coverage["RAG"] = "TESTED"
    next_comp = coverage_agent.select_next_competency(STRATEGY, coverage, questions_asked=1, question_budget=8)
    assert next_comp == "Python"


def test_select_next_competency_respects_budget():
    coverage = coverage_agent.init_coverage(STRATEGY)
    next_comp = coverage_agent.select_next_competency(STRATEGY, coverage, questions_asked=8, question_budget=8)
    assert next_comp is None


def test_update_coverage_after_turn_move_on_marks_tested():
    coverage = {"RAG": "NOT_TESTED"}
    coverage_agent.update_coverage_after_turn(coverage, "RAG", "MOVE_ON")
    assert coverage["RAG"] == "TESTED"


def test_update_coverage_after_turn_probe_marks_partial():
    coverage = {"RAG": "NOT_TESTED"}
    coverage_agent.update_coverage_after_turn(coverage, "RAG", "PROBE")
    assert coverage["RAG"] == "PARTIAL"


def test_coverage_is_complete():
    coverage = {"RAG": "TESTED", "Python": "TESTED", "Leadership": "TESTED"}
    assert coverage_agent.coverage_is_complete(STRATEGY, coverage) is True

    coverage["Leadership"] = "PARTIAL"
    assert coverage_agent.coverage_is_complete(STRATEGY, coverage) is False

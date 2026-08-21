from agents.interview_graph import route_after_evaluation


def _state(agent_action: str, questions_asked: int, question_budget: int) -> dict:
    return {"agent_action": agent_action, "questions_asked": questions_asked, "question_budget": question_budget}


def test_route_respects_budget_even_when_llm_keeps_recommending_followups():
    """Regression test: previously, question_budget was only checked in
    select_competency_node -- which a run of CLARIFY/PROBE/CHALLENGE
    follow-ups on the SAME competency never passes through. That meant a
    budget of 3 could silently balloon to 6, 10, or more follow-ups before
    the LLM ever said MOVE_ON. route_after_evaluation must now enforce the
    budget as a hard ceiling regardless of the LLM's recommendation."""
    for action in ("PROBE", "CLARIFY", "CHALLENGE", "PROBE_RESULT", "PROBE_TRADEOFFS", "COACH"):
        state = _state(action, questions_asked=3, question_budget=3)
        assert route_after_evaluation(state) == "after_move_on", (
            f"action={action} at budget limit should force wrap-up, not continue looping"
        )


def test_route_allows_followups_while_under_budget():
    """Under budget, the LLM's recommendation should still be honored normally."""
    assert route_after_evaluation(_state("PROBE", questions_asked=1, question_budget=3)) == "followup"
    assert route_after_evaluation(_state("CLARIFY", questions_asked=2, question_budget=3)) == "followup"
    assert route_after_evaluation(_state("COACH", questions_asked=1, question_budget=3)) == "coach"


def test_route_honors_genuine_move_on_even_under_budget():
    """A real MOVE_ON decision from the LLM should still end the competency
    early, even with budget remaining -- the fix shouldn't force follow-ups
    to continue just because there's room left."""
    assert route_after_evaluation(_state("MOVE_ON", questions_asked=1, question_budget=10)) == "after_move_on"


def test_route_over_budget_forces_wrapup_regardless_of_action():
    """Defensive: even if questions_asked somehow exceeds the budget (not
    just equals it), the hard cap still applies."""
    state = _state("CHALLENGE", questions_asked=5, question_budget=3)
    assert route_after_evaluation(state) == "after_move_on"

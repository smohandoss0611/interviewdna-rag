# Evals: automated checks for "is the AI's output actually good?"

## The idea, in one sentence

You already write unit tests that check "does my code do the right thing?"
An **eval** is the same idea, but for the AI's *answers* — since you can't
write `assert result == expected` (the AI's wording changes every time),
evals check things like "did it find the obviously-correct skills?" or "did
it correctly give a near-zero score to a garbage answer?"

## Two different tools, two different questions

| | `pytest tests/` | `python evals/run_evals.py` |
|---|---|---|
| Question it answers | "Does my **code logic** work?" | "Does the **actual model's output** hold up?" |
| Uses | `FakeLLMService` (instant, free, deterministic) | Your real configured LLM (Ollama/OpenAI/Groq) |
| Speed | Milliseconds | Seconds to minutes (real model calls) |
| Cost | Free | Free (Ollama) or real API cost (OpenAI/Groq) |
| Catches | Bugs in your Python | Bugs in prompts, or the model just not being good enough |

You need **both**. `pytest` would never have caught "the model gave a
pasted job description a 6/10 for correctness" — that's not a code bug,
the code did exactly what it was told. Only an eval, running the real
model, catches that.

## Every case here is a real bug we found manually

That's not a coincidence — it's the point. Each case in `evals/cases.py`
is modeled directly on something we only discovered by clicking through
the UI during this project's build:

| Eval case | The bug it would have caught automatically |
|---|---|
| `resume_extraction_finds_obvious_skills` | Resume DNA coming back completely empty |
| `job_extraction_finds_required_skills` | Same bug, for job description extraction |
| `offtopic_answer_scores_near_zero` | A copy-pasted job description scored as a real answer |
| `improvement_plan_excludes_untested_requirements` | Untested JD requirements (degree, Kubernetes) leaking into your results |

If we'd had this harness from day one, every one of those would have
shown up as a red `FAIL` in a 30-second script run, instead of requiring
a manual click-through, a screenshot, and a conversation to diagnose.

## Running it

```bash
# needs the same .env setup as the rest of the app (LLM_PROVIDER, etc.)
python evals/run_evals.py
```

Output looks like:

```
Running 4 eval case(s) against OllamaLLMService...

  running eval_resume_extraction_finds_obvious_skills... PASS (12.3s)
  running eval_job_extraction_finds_required_skills... PASS (8.1s)
  running eval_offtopic_answer_scores_near_zero... FAIL (6.4s)
  running eval_improvement_plan_excludes_untested_requirements... PASS (18.9s)

==============================================================================
CASE                                               RESULT   TIME
------------------------------------------------------------------------------
resume_extraction_finds_obvious_skills             PASS     12.3s
job_extraction_finds_required_skills               PASS     8.1s
offtopic_answer_scores_near_zero                   FAIL     6.4s
improvement_plan_excludes_untested_requirements    PASS     18.9s
------------------------------------------------------------------------------
3/4 passed in 45.7s total

FAILURE DETAILS:

  [offtopic_answer_scores_near_zero]
  scores={'correctness': 4, ...}, strength='Strong clarity (6/10)'
```

Exit code is `0` if everything passes, `1` if anything fails — wire this
into CI (a GitHub Actions step, for example) and you'll catch a prompt
change or a model swap that quietly makes output quality worse, the same
way `pytest` catches a code change that breaks behavior.

## Adding your own eval case

1. Pick a bug you find manually (there will be more — that's normal).
2. Write it as a function in `evals/cases.py`: take `llm: LLMService`,
   call the real service/agent function, check the output with a scorer
   from `evals/scorers.py` (or write a new deterministic check), return
   an `EvalResult`.
3. Add it to `ALL_CASES`.
4. **Also** add a version using `FakeLLMService` in
   `tests/test_evals_harness.py` that proves your new case correctly
   fails on the bad behavior and passes on the good behavior — this is
   the harness testing itself, and it's what makes the eval trustworthy
   rather than just "a script that might work."

## What this deliberately does NOT do (yet)

This harness uses simple, deterministic checks (keyword matching, score
thresholds) — no "ask another AI to judge the answer" (an **LLM-as-judge**
scorer). That's a real, more advanced technique, useful for fuzzier
questions like "is this coaching text well-written and encouraging?" —
but it's worth mastering deterministic evals first, since they're free,
instant, and you can fully explain every pass/fail. Add LLM-judge scoring
as a second layer once you've outgrown what keyword/threshold checks can
tell you.

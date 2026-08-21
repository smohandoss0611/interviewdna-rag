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

## RAG evaluation (`evals/rag_cases.py`, `evals/rag_scorers.py`)

Everything above checks the 12 core LLM calls (extraction, scoring,
planning). RAG evaluation is a different, more specific question: **is
retrieval actually finding the right thing, and does generation stay
faithful to what it found?**

Two kinds of check, because they need genuinely different techniques:

| Question | Technique | Where |
|---|---|---|
| Did retrieval find the relevant chunk at all? | Deterministic (substring match against a known corpus) | `relevant_chunk_found` |
| Is the relevant chunk actually ranked ABOVE an irrelevant one? | Deterministic (position comparison) | `relevant_chunk_ranked_above` |
| Does generated text (e.g. coaching) only claim things the retrieved context supports? | **LLM-as-judge** -- this genuinely needs semantic judgment, no keyword check can catch "the model added a plausible-sounding fact that isn't in the source" | `judge_faithfulness` |

That third one is the "second layer" mentioned above, now actually built:
a structured LLM call (`FaithfulnessJudgment`, `evals/schemas.py`) whose
only job is to compare generated text against its source context and flag
any unsupported claims. Same pattern as every other LLM call in this
app -- structured output, validated schema -- just aimed at judging output
quality instead of producing application data.

Run it the same way, since it's included in the main runner:
```bash
python evals/run_evals.py
```

The RAG cases seed a small, controlled corpus (one clearly relevant chunk,
one similar-sounding-but-irrelevant one) into a throwaway session before
each run, so results are reproducible and never depend on whatever happens
to already be sitting in your real Pinecone index.

## What this deliberately does NOT do (yet)

`evals/cases.py` still uses simple, deterministic checks on purpose (keyword
matching, score thresholds) for the extraction/quality evals — that's still
the right first tool, since it's free, instant, and every pass/fail is
fully explainable. LLM-as-judge (now built, see the RAG evaluation section
above) is genuinely the right *second* layer, reached for specifically
where deterministic checks can't tell you enough — like faithfulness,
which requires actually understanding whether a claim is supported.

Two things still missing if you want to go further:
- **Context recall** (of all the relevant chunks that exist, how many did
  retrieval actually find?) — needs a bigger labeled corpus than the
  two-chunk one here to be meaningful; worth building once you have real
  production data to draw a "ground truth" set from.
- **Answer relevancy** (is the generated coaching actually relevant to the
  original question, independent of faithfulness?) — a natural extension
  of `judge_faithfulness`'s pattern, just judging a different property.

#!/usr/bin/env python3
"""
Run the eval suite against your REAL, configured LLM (whatever's in .env --
Ollama, OpenAI, Groq, etc.). Unlike pytest's unit tests (which use a
FakeLLMService and check code logic), this exercises the actual model and
checks whether ITS OUTPUT is actually good.

Usage:
    python evals/run_evals.py

Exit code is 0 if everything passes, 1 if anything fails -- so this can be
wired into CI to catch a prompt change or model swap that silently makes
output quality worse, the same way pytest catches a code change that breaks
behavior.
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

from logging_config import configure_logging
configure_logging()

from llm.factory import get_llm_service
from evals.cases import ALL_CASES as EXTRACTION_AND_QUALITY_CASES
from evals.rag_cases import RAG_CASES

ALL_CASES = EXTRACTION_AND_QUALITY_CASES + RAG_CASES


def main():
    llm = get_llm_service()
    print(f"Running {len(ALL_CASES)} eval case(s) against {type(llm).__name__}...\n")

    results = []
    overall_start = time.monotonic()
    for case_fn in ALL_CASES:
        print(f"  running {case_fn.__name__}...", end=" ", flush=True)
        result = case_fn(llm)
        results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(f"{status} ({result.elapsed_s:.1f}s)")

    print(f"\n{'=' * 78}")
    print(f"{'CASE':<50} {'RESULT':<8} {'TIME':<8}")
    print("-" * 78)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"{r.name:<50} {status:<8} {r.elapsed_s:.1f}s")
    print("-" * 78)

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"{passed}/{total} passed in {time.monotonic() - overall_start:.1f}s total\n")

    failures = [r for r in results if not r.passed]
    if failures:
        print("FAILURE DETAILS:")
        for r in failures:
            print(f"\n  [{r.name}]")
            print(f"  {r.detail}")
        print()
        sys.exit(1)

    print("ALL EVALS PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()

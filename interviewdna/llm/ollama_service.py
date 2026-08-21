"""
Ollama-backed implementation of LLMService.

Runs against a local Ollama server (default http://localhost:11434) using an
open-source, Ollama-compatible model (default: "llama3.1", override via
OLLAMA_MODEL env var). No paid API keys required.

Only this file (and llm/base.py) know that "Ollama" exists. Every agent in
agents/*.py depends only on the LLMService interface.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import List, Dict, Type, TypeVar

import requests
from pydantic import BaseModel, ValidationError

from llm.base import LLMService, LLMStructuredOutputError, Message, looks_empty

logger = logging.getLogger("interviewdna.llm.ollama")

T = TypeVar("T", bound=BaseModel)


class OllamaLLMService(LLMService):
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        request_timeout: int | None = None,
    ):
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1")
        # Was hardcoded at 120s with no way to raise it without editing code.
        # CPU-only inference -- especially under Docker Desktop's default
        # (often conservative) CPU/RAM allocation on Mac -- can genuinely
        # need longer than 120s for a 7B-class model on some hardware. Set
        # OLLAMA_REQUEST_TIMEOUT in .env to raise this without touching code.
        self.request_timeout = request_timeout or int(os.getenv("OLLAMA_REQUEST_TIMEOUT", "120"))
        logger.info(
            "OllamaLLMService initialized (base_url=%s, model=%s, timeout=%ss)",
            self.base_url, self.model, self.request_timeout,
        )

    def _build_options(self, temperature: float, num_predict: int | None = None) -> dict:
        """Shared Ollama `options` builder. On CPU-only setups Ollama can
        under-use available cores unless told otherwise -- set
        OLLAMA_NUM_THREAD (e.g. to your physical core count) to fix that.
        Left unset by default so Ollama auto-detects."""
        options: dict = {"temperature": temperature}
        if num_predict is not None:
            options["num_predict"] = num_predict
        num_thread = os.getenv("OLLAMA_NUM_THREAD")
        if num_thread:
            options["num_thread"] = int(num_thread)
        return options

    # ------------------------------------------------------------------ #
    # Free text
    # ------------------------------------------------------------------ #
    def invoke(self, messages: List[Message], temperature: float = 0.4) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": self._build_options(temperature),
        }
        logger.info("invoke() -> %s | %d message(s), ~%d chars",
                    self.model, len(messages), sum(len(m.get("content", "")) for m in messages))
        start = time.monotonic()
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat", json=payload, timeout=self.request_timeout
            )
            resp.raise_for_status()
        except requests.RequestException:
            logger.exception(
                "invoke() FAILED after %.1fs -- is Ollama running at %s and is model "
                "'%s' pulled? (`ollama list` / `ollama pull %s`)",
                time.monotonic() - start, self.base_url, self.model, self.model,
            )
            raise
        elapsed = time.monotonic() - start
        data = resp.json()
        text = data.get("message", {}).get("content", "").strip()
        logger.info("invoke() done in %.1fs, %d chars returned", elapsed, len(text))
        return text

    # ------------------------------------------------------------------ #
    # Structured (JSON, validated against a Pydantic schema)
    # ------------------------------------------------------------------ #
    def invoke_structured(
        self,
        messages: List[Message],
        schema: Type[T],
        temperature: float = 0.2,
        max_retries: int = 2,
    ) -> T:
        # Deliberately a `user` message, not another `system` message -- many
        # smaller Ollama-served models handle multiple stacked system prompts
        # poorly (they get treated as lower priority / ignored), while a
        # trailing user instruction reliably gets followed.
        structured_messages = list(messages) + [
            {
                "role": "user",
                "content": (
                    "Now respond with ONLY a single valid JSON object -- no markdown "
                    "fences, no commentary, no explanation before or after.\n\n"
                    "IMPORTANT: base your answer on the actual text provided above. "
                    "If that text contains relevant skills, projects, requirements, "
                    "etc., you MUST include them -- do not return empty lists/fields "
                    "when the source text clearly contains that information."
                ),
            }
        ]

        logger.info(
            "invoke_structured() -> %s | schema=%s | %d message(s) | max_retries=%d",
            self.model, schema.__name__, len(messages), max_retries,
        )
        overall_start = time.monotonic()
        last_error: Exception | None = None
        # Ollama supports true grammar-constrained decoding: passing the
        # actual JSON schema (not just the string "json") to `format` forces
        # the model to only emit tokens that keep the output schema-valid.
        # This is far more reliable than loose format="json" + a text hint,
        # especially for smaller models. Requires a reasonably recent Ollama
        # (>= 0.5); falls back automatically to loose mode if the server
        # rejects a dict-valued `format`.
        use_schema_format = True
        for attempt in range(max_retries + 1):
            payload = {
                "model": self.model,
                "messages": structured_messages,
                "stream": False,
                "format": schema.model_json_schema() if use_schema_format else "json",
                # num_predict guards against truncated JSON on models/configs
                # with a low default completion budget, which would otherwise
                # silently produce a short-but-valid (e.g. mostly empty) JSON
                # object rather than an obvious error.
                "options": self._build_options(
                    temperature, num_predict=int(os.getenv("OLLAMA_NUM_PREDICT", "2048"))
                ),
            }
            attempt_start = time.monotonic()
            try:
                resp = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=self.request_timeout,
                )
                if use_schema_format and resp.status_code == 400:
                    logger.warning(
                        "Ollama rejected schema-constrained format (likely an older "
                        "Ollama version) -- falling back to loose format='json' mode."
                    )
                    use_schema_format = False
                    payload["format"] = "json"
                    resp = requests.post(
                        f"{self.base_url}/api/chat",
                        json=payload,
                        timeout=self.request_timeout,
                    )
                resp.raise_for_status()
                raw = resp.json().get("message", {}).get("content", "")
                logger.debug("invoke_structured() raw model output (schema=%s): %s",
                             schema.__name__, raw[:2000])
                cleaned = _strip_code_fences(raw)
                parsed = json.loads(cleaned)
                result = schema.model_validate(parsed)
                elapsed_total = time.monotonic() - overall_start
                logger.info(
                    "invoke_structured() -> %s done in %.1fs total (attempt %d/%d, %.1fs)",
                    schema.__name__, elapsed_total, attempt + 1, max_retries + 1,
                    time.monotonic() - attempt_start,
                )
                if looks_empty(parsed):
                    logger.warning(
                        "invoke_structured() -> %s parsed successfully but EVERY field is "
                        "empty/default. This is valid JSON, not a bug in parsing -- the model "
                        "itself extracted nothing. Set LOG_LEVEL=DEBUG to see the raw model "
                        "output and verify the input text it was given wasn't blank. Common "
                        "causes: (1) input text was empty/near-empty, (2) the model is too "
                        "small/weak to follow structured-extraction instructions under JSON "
                        "mode, (3) response got truncated. Try a larger model (e.g. llama3.1:8b "
                        "or qwen2.5:7b) if this persists.",
                        schema.__name__,
                    )
                return result
            except requests.exceptions.Timeout:
                last_error = TimeoutError(
                    f"Ollama did not respond within {self.request_timeout}s"
                )
                logger.error(
                    "invoke_structured() attempt %d/%d TIMED OUT after %.1fs "
                    "(schema=%s, model=%s) -- Ollama may be overloaded, still "
                    "loading the model into memory, or the prompt is too long "
                    "for this hardware",
                    attempt + 1, max_retries + 1, time.monotonic() - attempt_start,
                    schema.__name__, self.model,
                )
            except (json.JSONDecodeError, ValidationError, requests.RequestException) as exc:
                last_error = exc
                logger.warning(
                    "invoke_structured() attempt %d/%d failed after %.1fs (schema=%s): %s",
                    attempt + 1, max_retries + 1, time.monotonic() - attempt_start,
                    schema.__name__, exc,
                )
                structured_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "That response was not valid JSON matching the schema. "
                            f"Error: {exc}. Return ONLY corrected valid JSON."
                        ),
                    }
                )

        logger.error(
            "invoke_structured() -> %s EXHAUSTED all %d attempts after %.1fs total",
            schema.__name__, max_retries + 1, time.monotonic() - overall_start,
        )
        raise LLMStructuredOutputError(
            f"Could not obtain valid '{schema.__name__}' JSON after {max_retries + 1} attempts: {last_error}"
        )


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    return text.strip()

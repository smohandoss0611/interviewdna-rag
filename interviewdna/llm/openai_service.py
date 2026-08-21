"""
OpenAI-compatible implementation of LLMService.

OPTIONAL, opt-in provider. The project's default is still Ollama (free,
local, no API key) per the original "don't require a paid API" requirement
-- this exists purely because the LLMService abstraction (llm/base.py) was
built specifically so a different provider could be swapped in later without
touching agents/interview_graph.py or any other orchestration code. This is
that swap.

This isn't just OpenAI -- it works with ANY provider that speaks the OpenAI
chat-completions wire format, which covers most hosted open-model APIs.
Point OPENAI_BASE_URL at the provider and set OPENAI_API_KEY /
OPENAI_MODEL accordingly:

    Provider    Base URL                                  Notes
    ----------  ----------------------------------------  ------------------------------
    OpenAI      (leave unset, defaults to api.openai.com)  gpt-4o-mini is a solid default
    Groq        https://api.groq.com/openai/v1             very fast, generous free tier;
                                                             openai/gpt-oss-20b is current
                                                             (Groq deprecated its Llama
                                                             chat models in 2026). Note:
                                                             gpt-oss models are REASONING
                                                             models -- this file sends
                                                             Groq's reasoning_format=hidden
                                                             automatically to reduce
                                                             reasoning-content leaking into
                                                             JSON responses.
    Together    https://api.together.xyz/v1                wide open-model selection
    Fireworks   https://api.fireworks.ai/inference/v1       fast, pay-per-token
    DeepSeek    https://api.deepseek.com/v1                 very cheap; deepseek-chat

To use it:
    pip install openai
    # in .env:
    LLM_PROVIDER=openai
    OPENAI_API_KEY=<provider's api key>
    OPENAI_MODEL=<provider's model name>
    OPENAI_BASE_URL=<provider's base url, or leave unset for real OpenAI>

Nothing else in the codebase changes -- llm/factory.py is the only other
file that knows this exists. Native structured-outputs mode
(client.chat.completions.parse) is tried first regardless of provider; if
the provider doesn't support it, this falls back automatically to JSON mode
+ manual validation, so non-OpenAI providers still work even without full
structured-outputs support.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import List, Type, TypeVar

from pydantic import BaseModel, ValidationError

from llm.base import LLMService, LLMStructuredOutputError, Message, looks_empty

logger = logging.getLogger("interviewdna.llm.openai")

T = TypeVar("T", bound=BaseModel)


class OpenAILLMService(LLMService):
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        request_timeout: int = 120,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "LLM_PROVIDER=openai but OPENAI_API_KEY is not set. Add it to .env "
                "(with OPENAI_BASE_URL if you're pointing at Groq/Together/etc. "
                "instead of real OpenAI), or switch LLM_PROVIDER back to 'ollama' "
                "for the free local default."
            )
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or None
        self.request_timeout = request_timeout
        self._client = None
        logger.info(
            "OpenAILLMService initialized (model=%s, base_url=%s, timeout=%ss)",
            self.model, self.base_url or "api.openai.com (default)", self.request_timeout,
        )

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.api_key, base_url=self.base_url, timeout=self.request_timeout
            )
        return self._client

    def _reasoning_extra_body(self) -> dict:
        """Some hosted models (notably Groq's GPT-OSS family) are reasoning
        models that mix internal 'thinking' content into the response by
        default, which breaks strict JSON validation -- this is a known,
        fairly common failure mode (Groq returns 'json_validate_failed').
        Groq documents `reasoning_format: "hidden"` specifically to
        suppress that. Only sent when talking to Groq, since this is a
        Groq-specific parameter that other OpenAI-compatible providers may
        not recognize."""
        if self.base_url and "groq.com" in self.base_url:
            return {"reasoning_format": "hidden"}
        return {}

    # ------------------------------------------------------------------ #
    # Free text
    # ------------------------------------------------------------------ #
    def invoke(self, messages: List[Message], temperature: float = 0.4) -> str:
        client = self._get_client()
        logger.info("invoke() -> %s | %d message(s), ~%d chars",
                    self.model, len(messages), sum(len(m.get("content", "")) for m in messages))
        start = time.monotonic()
        try:
            resp = client.chat.completions.create(
                model=self.model, messages=messages, temperature=temperature,
                extra_body=self._reasoning_extra_body() or None,
            )
        except Exception:
            logger.exception("invoke() FAILED after %.1fs (model=%s)", time.monotonic() - start, self.model)
            raise
        text = (resp.choices[0].message.content or "").strip()
        logger.info("invoke() done in %.1fs, %d chars returned", time.monotonic() - start, len(text))
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
        client = self._get_client()

        # OpenAI's native structured-outputs mode (response_format=<pydantic
        # model>) grammar-constrains generation to match the schema exactly
        # -- the most reliable option, but requires openai>=1.40 and a
        # structured-outputs-capable model. We try it first and fall back to
        # plain JSON mode + validation (matching the Ollama implementation's
        # approach) if that API isn't available, so this keeps working across
        # SDK/model versions.
        logger.info(
            "invoke_structured() -> %s | schema=%s | %d message(s) | max_retries=%d",
            self.model, schema.__name__, len(messages), max_retries,
        )
        overall_start = time.monotonic()

        if hasattr(client.chat.completions, "parse"):
            try:
                return self._invoke_structured_native(client, messages, schema, temperature, overall_start)
            except Exception as exc:
                logger.warning(
                    "Native structured-outputs call failed (%s) -- falling back to "
                    "JSON mode + manual validation for this call.", exc,
                )

        return self._invoke_structured_json_mode(client, messages, schema, temperature, max_retries, overall_start)

    def _invoke_structured_native(self, client, messages, schema, temperature, overall_start):
        resp = client.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=schema,
            temperature=temperature,
            extra_body=self._reasoning_extra_body() or None,
        )
        result = resp.choices[0].message.parsed
        if result is None:
            raise LLMStructuredOutputError(
                f"OpenAI returned a refusal or empty parse for schema '{schema.__name__}'"
            )
        elapsed = time.monotonic() - overall_start
        logger.info("invoke_structured() (native) -> %s done in %.1fs", schema.__name__, elapsed)
        if looks_empty(result.model_dump()):
            logger.warning(
                "invoke_structured() -> %s parsed successfully but EVERY field is "
                "empty/default. The call succeeded -- check that the input text "
                "(resume/JD) actually contains extractable content.",
                schema.__name__,
            )
        return result

    def _invoke_structured_json_mode(self, client, messages, schema, temperature, max_retries, overall_start):
        working_messages = list(messages) + [
            {
                "role": "user",
                "content": (
                    "Now respond with ONLY a single valid JSON object -- no markdown "
                    "fences, no commentary, no explanation, and no reasoning/thinking "
                    "text before or after it. Output the JSON object and NOTHING else.\n\n"
                    "IMPORTANT: base your answer on the actual text provided above. "
                    "If that text contains relevant skills, projects, requirements, "
                    "etc., you MUST include them -- do not return empty lists/fields "
                    "when the source text clearly contains that information."
                ),
            }
        ]

        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            attempt_start = time.monotonic()
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=working_messages,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                    extra_body=self._reasoning_extra_body() or None,
                )
                raw = resp.choices[0].message.content or ""
                logger.debug("invoke_structured() (json mode) raw output (schema=%s): %s",
                             schema.__name__, raw[:2000])
                # Defensive: even with reasoning suppressed, some models still
                # wrap the JSON in stray text (markdown fences, a leading
                # "Here is the JSON:", etc.) -- extract just the outermost
                # {...} block rather than assuming the whole response is
                # pure JSON.
                json_text = _extract_json_object(raw)
                parsed = json.loads(json_text)
                result = schema.model_validate(parsed)
                logger.info(
                    "invoke_structured() (json mode) -> %s done in %.1fs total (attempt %d/%d)",
                    schema.__name__, time.monotonic() - overall_start, attempt + 1, max_retries + 1,
                )
                if looks_empty(parsed):
                    logger.warning(
                        "invoke_structured() -> %s parsed successfully but EVERY field "
                        "is empty/default -- check the input text actually had content.",
                        schema.__name__,
                    )
                return result
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                logger.warning(
                    "invoke_structured() attempt %d/%d failed after %.1fs (schema=%s): %s",
                    attempt + 1, max_retries + 1, time.monotonic() - attempt_start, schema.__name__, exc,
                )
                working_messages.append(
                    {"role": "user", "content": f"That was not valid JSON matching the schema. Error: {exc}. Return ONLY corrected valid JSON, with no reasoning or explanation text."}
                )
            except Exception as exc:
                # Covers server-side rejections too (e.g. Groq's
                # "json_validate_failed" 400 error) -- these previously fell
                # through with NO correction added to working_messages, so
                # every retry sent the exact same request and failed
                # identically each time. Now we still nudge the model on
                # retry instead of blindly repeating.
                last_error = exc
                logger.exception("invoke_structured() attempt %d/%d raised an API error", attempt + 1, max_retries + 1)
                working_messages.append(
                    {"role": "user", "content": (
                        f"That request failed with an error: {exc}. Respond again with ONLY a "
                        "single valid JSON object matching the schema, no reasoning or "
                        "explanation text before or after it."
                    )}
                )

        raise LLMStructuredOutputError(
            f"Could not obtain valid '{schema.__name__}' JSON after {max_retries + 1} attempts: {last_error}"
        )


def _extract_json_object(text: str) -> str:
    """Extract the outermost {...} block from `text`, tolerating stray text
    (markdown fences, leading/trailing commentary, leaked reasoning tokens)
    around it. Falls back to the original text unchanged if no clear JSON
    object boundaries are found, so json.loads() can still raise its normal,
    informative error rather than this function masking the problem."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start:end + 1]
    return stripped

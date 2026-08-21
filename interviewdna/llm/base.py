"""
LLM provider abstraction.

Every agent/service in InterviewDNA talks to the model ONLY through this
interface. This is what makes it possible to swap Ollama for a different
Ollama-compatible model, or a different provider entirely, later, WITHOUT
touching agents/interview_graph.py or any LangGraph routing logic.

    LangGraph Node
          |
          v
      LLMService            <-- this file defines the contract
          |
          v
    OllamaLLMService (llm/ollama_service.py)
          |
          v
        Ollama
          |
          v
      Local LLM
"""
from __future__ import annotations

import abc
from typing import List, Dict, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

Message = Dict[str, str]  # {"role": "system"|"user"|"assistant", "content": "..."}


class LLMService(abc.ABC):
    """Abstract LLM runtime interface.

    Concrete implementations (e.g. OllamaLLMService) are responsible for:
      - turning `messages` into a provider-specific request
      - for invoke_structured: forcing/validating JSON output against `schema`
        and retrying on malformed output
    """

    @abc.abstractmethod
    def invoke(self, messages: List[Message], temperature: float = 0.4) -> str:
        """Free-text completion. Returns the raw text response."""
        raise NotImplementedError

    @abc.abstractmethod
    def invoke_structured(
        self,
        messages: List[Message],
        schema: Type[T],
        temperature: float = 0.2,
        max_retries: int = 2,
    ) -> T:
        """Structured completion. Returns a validated instance of `schema`.

        Implementations MUST validate the model output against `schema` and
        retry (with a corrective follow-up message) on failure, raising
        LLMStructuredOutputError if it still cannot produce valid output.
        """
        raise NotImplementedError


class LLMStructuredOutputError(RuntimeError):
    """Raised when the LLM cannot be coerced into valid structured output."""


def looks_empty(parsed: dict) -> bool:
    """True if every value in a parsed structured-output dict is empty/falsy
    (empty list/dict/string, None, or 0) -- i.e. the model returned a
    schema-valid but content-free object. Shared across provider
    implementations purely for a diagnostic log warning; never used to fail
    a call, since a genuinely empty resume section is a valid outcome too."""
    if not isinstance(parsed, dict) or not parsed:
        return True
    for v in parsed.values():
        if isinstance(v, (list, dict, str)) and len(v) > 0:
            return False
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v != 0:
            return False
        if isinstance(v, bool):
            return False
    return True

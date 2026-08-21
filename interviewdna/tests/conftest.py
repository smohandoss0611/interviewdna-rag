import sys
import os
from typing import List, Type, TypeVar

import pytest
from pydantic import BaseModel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm.base import LLMService

T = TypeVar("T", bound=BaseModel)


class FakeLLMService(LLMService):
    """Deterministic stand-in for llm.base.LLMService used across tests, so
    no test depends on a running Ollama server."""

    def __init__(self, structured_responses: dict[str, BaseModel] | None = None):
        self.structured_responses = structured_responses or {}
        self.calls = []

    def invoke(self, messages, temperature: float = 0.4) -> str:
        self.calls.append(("invoke", messages))
        return "fake response"

    def invoke_structured(self, messages, schema: Type[T], temperature: float = 0.2, max_retries: int = 2) -> T:
        self.calls.append(("invoke_structured", schema.__name__, messages))
        if schema.__name__ in self.structured_responses:
            return self.structured_responses[schema.__name__]
        raise AssertionError(f"No fake response registered for schema {schema.__name__}")


@pytest.fixture
def fake_llm():
    return FakeLLMService()

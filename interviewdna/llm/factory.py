"""
Single place that decides WHICH concrete LLMService implementation is used.

To swap providers later (per spec: "Keep the LLM provider behind an
abstraction so a different model/provider could be substituted later without
changing the LangGraph workflow"), only this file needs to change -- add a
new LLM_PROVIDER branch and a new llm/<provider>_service.py implementing
llm.base.LLMService.

Set LLM_PROVIDER in .env:
    LLM_PROVIDER=ollama   (default -- free, local, no API key)
    LLM_PROVIDER=openai   (opt-in -- requires OPENAI_API_KEY, paid)
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from llm.base import LLMService

logger = logging.getLogger("interviewdna.llm.factory")

_llm: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    global _llm
    if _llm is None:
        provider = os.getenv("LLM_PROVIDER", "ollama").lower()
        logger.info("Initializing LLM provider: %s", provider)
        if provider == "ollama":
            from llm.ollama_service import OllamaLLMService

            _llm = OllamaLLMService()
        elif provider == "openai":
            from llm.openai_service import OpenAILLMService

            _llm = OpenAILLMService()
        else:
            raise ValueError(
                f"Unknown LLM_PROVIDER: {provider!r} -- expected 'ollama' or 'openai'"
            )
    return _llm

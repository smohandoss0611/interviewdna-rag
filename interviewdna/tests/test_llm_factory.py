import os
import pytest

import llm.factory as factory_module


@pytest.fixture(autouse=True)
def _reset_llm_singleton(monkeypatch):
    """Each test gets a clean factory singleton so provider selection is
    actually exercised rather than cached from a previous test."""
    factory_module._llm = None
    yield
    factory_module._llm = None


def test_factory_defaults_to_ollama(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    llm = factory_module.get_llm_service()
    assert type(llm).__name__ == "OllamaLLMService"


def test_factory_selects_openai_when_configured(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key")
    llm = factory_module.get_llm_service()
    assert type(llm).__name__ == "OpenAILLMService"


def test_openai_service_accepts_custom_base_url_for_compatible_providers(monkeypatch):
    """Confirms the OpenAI-compatible provider can point at Groq/Together/
    Fireworks/DeepSeek/etc. via OPENAI_BASE_URL, not just real OpenAI."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "gsk-fake-groq-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setenv("OPENAI_MODEL", "llama-3.3-70b-versatile")

    llm = factory_module.get_llm_service()
    assert llm.base_url == "https://api.groq.com/openai/v1"
    assert llm.model == "llama-3.3-70b-versatile"


def test_factory_openai_without_api_key_raises_clear_error(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        factory_module.get_llm_service()


def test_factory_rejects_unknown_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "not_a_real_provider")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        factory_module.get_llm_service()
